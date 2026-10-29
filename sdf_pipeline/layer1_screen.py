"""Layer 1 screen: an implausibility check on the dealt hands, before any planning.

The deal (and its deal-rule repair) produces one hand per document: a card from
each variable's column. This stage shows the model the hands, ten at a time,
rendered as the variable lines the planner will see, and asks which are extremely
implausible: no realistic document could satisfy every dealt value, judged
against how the world actually is. It answers with local numbers 1..N, never
hand ids; this module maps the numbers back. Removed hands are never planned.

Removed hands are redealt so the dealt shares stay at the weights. Their cards
go back into a per-column pool, each column is reshuffled with a seed derived
from the run seed and the round number, the pool is dealt into as many new
hands as were removed, the deal-rule repair swaps cards only among those new
hands (hands that already passed are never touched), and the new hands are
screened in turn, for up to ``sdf.screen_redeal_rounds`` redeal rounds. Cards
still removed after the last round are dropped and reported (the column counts
of the final hands then fall short of the dealt counts by exactly those cards).

The screening logic is importable on its own: ``screen_hands`` takes the
rendered variable lines of one batch of hands and returns one judgment per hand,
using the same prompt files, answer schema and validation as the stage, so
offline validation scripts can reuse it unchanged. ``build_screen_prompt`` is
the matching prompt builder. The model's answer is schema-constrained JSON, one
entry per hand (reasoning first, then the decision); every entry is kept in
screen_judgments.jsonl so keeps are auditable.

Config (``sdf`` section): ``screen_enabled`` (default true), ``screen_model``
(falls back to ``plan_model``, then the global model), ``screen_batch_size``
(default 10), ``screen_redeal_rounds`` (default 3).

Files in the layer 1 directory (every round's deal and judgments are saved
before the next round starts, so a resumed run repeats identical rounds):
- screen_hands.jsonl: every hand ever dealt, {prompt_id, round, variables}.
  Round 0 is the original deal; hand ids are the ids the surviving hands keep
  as prompt ids, so the example species the planner is shown is the one the
  screen saw.
- screen_rounds.jsonl: one record per round, {round, prompt_ids, note}. A round
  exists once its record is written; hands saved without one (an interrupted deal
  or redeal) are ignored and dealt again, and only hands of a recorded round are
  ever accepted.
- screen_batches.jsonl: the batches of a round, written before its first call; a
  round whose saved batches do not hold exactly its hands is planned again.
- screen_judgments.jsonl / screen_rejections.jsonl: every judgment; the removals
  with their reason, round, variables and the lines the model saw.
- _screen_checkpoint.json: one checkpoint per batch. A malformed, truncated,
  miscounted, duplicate or out-of-range answer is NOT checkpointed, so --resume
  retries exactly that batch. A batch's judgments and rejections are appended
  before its checkpoint, so a crash between the two re-screens the batch, and the
  new answer replaces the batch's earlier records in both files.
- screened_hands.jsonl + screen_summary.json: written when every round is done;
  their presence makes a resumed run return at once with zero calls.
"""

import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, deal_constraints, utils
from shared.matrix import split_sections
from sdf_pipeline import compose_prompts as cp

STAGE_TAG = "layer1_screen"
TEMPLATE_FILENAME = "layer1_screen.txt"
HAND_BLOCK_FILENAME = "layer1_screen_specification_block.txt"
SCHEMA_FILENAME = "layer1_screen_answer_schema.json"
HANDS_FILENAME = "screen_hands.jsonl"
ROUNDS_FILENAME = "screen_rounds.jsonl"
BATCHES_FILENAME = "screen_batches.jsonl"
JUDGMENTS_FILENAME = "screen_judgments.jsonl"
REJECTIONS_FILENAME = "screen_rejections.jsonl"
CHECKPOINT_FILENAME = "_screen_checkpoint.json"
SCREENED_HANDS_FILENAME = "screened_hands.jsonl"
SUMMARY_FILENAME = "screen_summary.json"
DEFAULT_BATCH_SIZE = 10
DEFAULT_REDEAL_ROUNDS = 3
MAX_REDEAL_ATTEMPTS = 25  # reshuffles tried per round before the removed cards are given up on
_MAX_TOKENS = 8000  # reasoning plus a decision for each of 10 hands; a batch used 1,819 and another hit 2,000 on 2026-10-22

# Values an evaluation or report flags: a variable value removed at more than
# this multiple of the corpus-wide removal rate (with at least MIN_REMOVED_TO_FLAG
# removals, so a single removal in a tiny cell is not a finding).
FLAG_RATE_MULTIPLE = 2.0
MIN_REMOVED_TO_FLAG = 2


class ScreenAnswerError(ValueError):
    """The model's screen answer was truncated, malformed, or out of range."""


@dataclass(frozen=True)
class ScreenJudgment:
    number: int  # the local 1-based number the model gave
    reasoning: str
    remove: bool
    reason: str  # one-sentence reason; non-empty whenever remove is true

    @property
    def index(self) -> int:
        return self.number - 1


# ---------------------------------------------------------------- prompt and parsing

@dataclass(frozen=True)
class ScreenPrompts:
    """Every model-facing string of the screen, read from prompts/sdf/*.txt (and the
    answer schema from prompts/sdf/layer1_screen_answer_schema.json)."""
    template: str
    hand_block: str
    answer_schema: dict


def load_screen_prompts(prompts_dir: Path) -> ScreenPrompts | None:
    """The screen's prompt files, or None for a run whose snapshot predates the screen."""
    directory = Path(prompts_dir)
    if not (directory / TEMPLATE_FILENAME).is_file():
        return None
    read = lambda name: (directory / name).read_text(encoding="utf-8")  # noqa: E731
    return ScreenPrompts(
        template=read(TEMPLATE_FILENAME),
        hand_block=read(HAND_BLOCK_FILENAME),
        answer_schema=json.loads(read(SCHEMA_FILENAME)),
    )


def build_answer_schema(prompts: ScreenPrompts, hand_count: int) -> dict:
    """The answer schema with the ``number`` enum filled for this batch (1..N)."""
    schema = copy.deepcopy(prompts.answer_schema)
    schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] = list(
        range(1, hand_count + 1)
    )
    return schema


def build_screen_prompt(prompts: ScreenPrompts, rendered_hands: list[str]) -> tuple[str | None, str]:
    """Render (system, user) for one batch. ``rendered_hands`` holds each hand's
    variable lines as the planner sees them (``compose_prompts.render_variable_lines``)."""
    blocks = [
        prompts.hand_block.format(number=number, variable_lines=lines.strip())
        for number, lines in enumerate(rendered_hands, 1)
    ]
    rendered = prompts.template.format(hand_count=len(rendered_hands), hands="\n\n".join(blocks))
    return split_sections(rendered)


def parse_screen_answer(raw: str, hand_count: int) -> list[ScreenJudgment]:
    """One judgment per hand, ordered by number. Raises ScreenAnswerError unless
    the answer is JSON of the schema's shape with exactly hand_count entries,
    numbers 1..N each exactly once, and a non-empty reason on every entry marked
    for removal."""
    try:
        data = json.loads(raw or "")
        entries = data["specifications"]
    except (ValueError, KeyError, TypeError) as e:
        raise ScreenAnswerError(f"answer is not the expected JSON ({type(e).__name__}: {e})") from e
    if not isinstance(entries, list) or len(entries) != hand_count:
        raise ScreenAnswerError(
            f"expected {hand_count} entries, got "
            f"{len(entries) if isinstance(entries, list) else type(entries).__name__}")
    judgments: dict[int, ScreenJudgment] = {}
    for entry in entries:
        try:
            number, reasoning, remove, reason = (
                entry["number"], entry["reasoning"], entry["remove"], entry["reason"])
        except (KeyError, TypeError) as e:
            raise ScreenAnswerError(f"entry missing a field: {entry!r}") from e
        if (isinstance(number, bool) or not isinstance(number, int)
                or not isinstance(reasoning, str) or not isinstance(remove, bool)
                or not isinstance(reason, str)):
            raise ScreenAnswerError(f"entry has a wrongly typed field: {entry!r}")
        if not 1 <= number <= hand_count:
            raise ScreenAnswerError(f"number {number} is outside 1..{hand_count}")
        if number in judgments:
            raise ScreenAnswerError(f"number {number} listed twice")
        if remove and not reason.strip():
            raise ScreenAnswerError(f"number {number} is marked for removal without a reason")
        judgments[number] = ScreenJudgment(
            number=number, reasoning=reasoning.strip(), remove=remove, reason=reason.strip())
    return [judgments[n] for n in range(1, hand_count + 1)]


def screen_hands(
    rendered_hands: list[str],
    *,
    prompts: ScreenPrompts,
    model: str | None = None,
    item_id: str | None = None,
) -> list[ScreenJudgment]:
    """Screen one batch of rendered hands and return a judgment for every one
    (reasoning included, so keeps are auditable); the removals are the entries
    with ``remove`` true.

    The call is schema-constrained (number enum 1..N). Raises ScreenAnswerError
    for a truncated or invalid answer; API errors propagate. Uses the global
    temperature and tags the call ``layer1_screen`` for the cost log."""
    system, user = build_screen_prompt(prompts, rendered_hands)
    raw, stop_reason = api.call_claude(
        user,
        system_prompt=system or "",
        model=model,
        max_tokens=_MAX_TOKENS,
        stage=STAGE_TAG,
        item_id=item_id,
        return_stop_reason=True,
        json_schema=build_answer_schema(prompts, len(rendered_hands)),
    )
    if stop_reason != "end_turn":
        raise ScreenAnswerError(f"truncated answer (stop_reason={stop_reason})")
    return parse_screen_answer(raw, len(rendered_hands))


# ---------------------------------------------------------------- batching and redealing

def plan_round_batches(
    output_dir: Path, round_number: int, prompt_ids: list[str], batch_size: int, seed: int
) -> list[dict]:
    """The batches of one round, persisted before any call of that round.

    A seeded shuffle (of the sorted ids, so input order is irrelevant) cut into
    batch_size chunks. On resume the round's saved batches are reused verbatim when
    together they hold exactly the round's hands; a partial set is planned again."""
    path = output_dir / BATCHES_FILENAME
    every_batch = utils.load_jsonl(path)
    saved = [b for b in every_batch if b["round"] == round_number]
    saved_ids = [pid for b in saved for pid in b["prompt_ids"]]
    if saved and sorted(saved_ids) == sorted(prompt_ids):
        return saved
    if saved:  # an interrupted write left only some of the round's batches; plan the round again
        utils.save_jsonl([b for b in every_batch if b["round"] != round_number], path)
    shuffled = sorted(prompt_ids)
    random.Random(f"{seed}:{STAGE_TAG}:round{round_number}").shuffle(shuffled)
    batches = []
    for start in range(0, len(shuffled), batch_size):
        batch = {
            "batch_id": f"screen_round{round_number}_batch{len(batches):04d}",
            "round": round_number,
            "prompt_ids": shuffled[start:start + batch_size],
        }
        batches.append(batch)
        utils.append_jsonl(batch, path)
    return batches


def redeal_removed_hands(
    removed_hands: list[dict],
    round_number: int,
    seed: int,
    deal_rules: list[deal_constraints.DealRule],
) -> list[dict[str, str]] | None:
    """New hands (variables only) dealt from the removed hands' cards, or None.

    Each column of the pool is reshuffled with a seed derived from the run seed
    and the round number, and the shuffled columns are zipped into
    len(removed_hands) new hands, so every column keeps exactly its cards. The
    deal-rule repair then swaps cards among these new hands only. A reshuffle
    that cannot be repaired, or that just reproduces the removed hands, is
    retried with the next attempt's seed; after MAX_REDEAL_ATTEMPTS the cards are
    given up on (None)."""
    axes = list(removed_hands[0]["variables"])
    removed_as_dealt = sorted(json.dumps(h["variables"], sort_keys=True) for h in removed_hands)
    for attempt in range(MAX_REDEAL_ATTEMPTS):
        rng = random.Random(f"{seed}:{STAGE_TAG}:redeal{round_number}:attempt{attempt}")
        columns = {}
        for axis in axes:
            column = [hand["variables"][axis] for hand in removed_hands]
            rng.shuffle(column)
            columns[axis] = column
        new_hands = [{axis: columns[axis][i] for axis in axes} for i in range(len(removed_hands))]
        try:
            new_hands, _ = deal_constraints.swap_repair(new_hands, deal_rules, rng)
        except deal_constraints.InfeasibleDealConstraintError:
            continue
        if sorted(json.dumps(h, sort_keys=True) for h in new_hands) == removed_as_dealt:
            continue  # the same hands again would only be removed again
        return new_hands
    return None


# ---------------------------------------------------------------- stage

def load_rejections(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: rejection record}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / REJECTIONS_FILENAME)}


def _drop_batch_records(path: Path, batch_id: str) -> None:
    """Remove a batch's earlier records from a jsonl file, so that screening the
    batch again replaces its verdicts instead of adding to them."""
    records = utils.load_jsonl(path)
    kept = [r for r in records if r["batch_id"] != batch_id]
    if len(kept) != len(records):
        utils.save_jsonl(kept, path)


def _next_hand_index(hands: list[dict]) -> int:
    return max((int(h["prompt_id"].rsplit("_", 1)[1]) for h in hands), default=-1) + 1


def run(config: dict, prompts_dir: Path, output_dir: Path) -> list[dict] | None:
    """Deal the hands, screen them with redeal rounds, and return the accepted
    hand records ({"prompt_id", "variables"}) for planning. Returns None, making
    no calls, when the screen is disabled or the run's prompt snapshot has no
    screen template (the planner then deals its own hands as before)."""
    sdf = config["sdf"]
    if not sdf.get("screen_enabled", True):
        print("  Screen disabled (sdf.screen_enabled: false)")
        return None
    prompts = load_screen_prompts(prompts_dir)
    if prompts is None:
        print(f"  No {TEMPLATE_FILENAME} in this run's prompt snapshot — screen skipped")
        return None

    accepted_path = output_dir / SCREENED_HANDS_FILENAME
    if accepted_path.exists():
        accepted = utils.load_jsonl(accepted_path)
        print(f"  Screen already complete: {len(accepted)} hands accepted (no calls)")
        return accepted

    seed = sdf.get("seed", 0)
    batch_size = int(sdf.get("screen_batch_size", DEFAULT_BATCH_SIZE))
    redeal_rounds = int(sdf.get("screen_redeal_rounds", DEFAULT_REDEAL_ROUNDS))
    model = sdf.get("screen_model") or sdf.get("plan_model")
    render_variable_lines = cp.make_variable_line_renderer(
        prompts_dir, sdf.get("entity_pool_seed", cp.DEFAULT_ENTITY_SEED))
    _, _, value_ids = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
    deal_rules = cp.load_deal_rules(prompts_dir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
    checkpoint = utils.Checkpoint(output_dir / CHECKPOINT_FILENAME)
    hands_path = output_dir / HANDS_FILENAME
    rounds_path = output_dir / ROUNDS_FILENAME
    judgments_path = output_dir / JUDGMENTS_FILENAME
    rejections_path = output_dir / REJECTIONS_FILENAME

    hands = utils.load_jsonl(hands_path)
    rounds = utils.load_jsonl(rounds_path)
    if not rounds:  # round 0: the original deal, saved before anything is screened
        template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
        values, weights, _ = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
        dealt = cp.hand_records(
            cp.deal_hands(template, values, weights, sdf["n_prompts"], seed, deal_rules))
        hands = [{"prompt_id": h["prompt_id"], "round": 0, "variables": h["variables"]} for h in dealt]
        utils.save_jsonl(hands, hands_path)   # replaces any partial write from an interrupted deal
        rounds = [{"round": 0, "prompt_ids": [h["prompt_id"] for h in hands], "note": "original deal"}]
        utils.append_jsonl(rounds[0], rounds_path)
    recorded_ids = {pid for r in rounds for pid in r["prompt_ids"]}
    hands = [h for h in hands if h["prompt_id"] in recorded_ids]   # hands saved before their round record are dealt again
    hands_by_id = {h["prompt_id"]: h for h in hands}

    started_all = time.monotonic()
    workers = config.get("workers", 1)
    round_number = 0
    while True:
        round_record = rounds[round_number]
        round_ids = round_record["prompt_ids"]
        batches = plan_round_batches(output_dir, round_number, round_ids, batch_size, seed)
        pending = [b for b in batches if not checkpoint.is_done(b["batch_id"])]
        print(f"  Screen round {round_number}: {len(round_ids)} hands in {len(batches)} batches of up to "
              f"{batch_size} ({len(batches) - len(pending)} already done)")

        def screen_one(batch: dict):
            members = [hands_by_id[pid] for pid in batch["prompt_ids"]]
            shown = [render_variable_lines(m) for m in members]
            try:
                started = time.monotonic()
                judgments = screen_hands(shown, prompts=prompts, model=model, item_id=batch["batch_id"])
                return members, shown, judgments, time.monotonic() - started, None
            except ScreenAnswerError as e:
                return members, shown, None, 0.0, f"unusable answer ({e})"
            except Exception as e:  # one failing call must not discard its siblings' work
                return members, shown, None, 0.0, f"API call failed ({type(e).__name__}: {e})"

        failed = 0
        round_started = time.monotonic()
        for done, (batch, (members, shown, judgments, seconds, problem)) in enumerate(
            zip(pending, utils.parallel_map(screen_one, pending, workers)), 1
        ):
            if problem:
                failed += 1
                print(f"  {batch['batch_id']}: {problem} — will retry on resume")
                continue
            _drop_batch_records(judgments_path, batch["batch_id"])
            _drop_batch_records(rejections_path, batch["batch_id"])
            for judgment in judgments:
                member, lines = members[judgment.index], shown[judgment.index]
                utils.append_jsonl({
                    "prompt_id": member["prompt_id"], "round": round_number,
                    "batch_id": batch["batch_id"], "number": judgment.number,
                    "reasoning": judgment.reasoning, "remove": judgment.remove,
                    "reason": judgment.reason,
                }, judgments_path)
                if judgment.remove:
                    utils.append_jsonl({
                        "prompt_id": member["prompt_id"], "round": round_number,
                        "batch_id": batch["batch_id"], "number": judgment.number,
                        "reason": judgment.reason, "variables": member["variables"],
                        "variable_lines": lines,
                    }, rejections_path)
            checkpoint.mark_done(batch["batch_id"])
            print(f"  Screened {batch['batch_id']} ({done}/{len(pending)}): {len(members)} hands, "
                  f"{sum(j.remove for j in judgments)} extremely implausible, {seconds:.1f}s")

        if pending and failed == len(pending):
            raise SystemExit(
                "layer1_screen: every pending screen call failed — this is systemic "
                "(auth, backend, or network), not per-batch; fix and --resume."
            )
        if failed:
            raise SystemExit(
                f"layer1_screen: {failed} batch(es) of round {round_number} failed; the next "
                "round cannot be dealt until the round is complete — --resume retries them."
            )

        rejected = load_rejections(output_dir)
        removed_ids = [pid for pid in round_ids if pid in rejected]
        print(f"  Round {round_number} removed {len(removed_ids)} of {len(round_ids)} hands "
              f"({time.monotonic() - round_started:.1f}s; {time.monotonic() - started_all:.1f}s total)")
        if not removed_ids or round_number >= redeal_rounds:
            break
        if round_number + 1 < len(rounds):  # the next round was dealt before an interruption
            round_number += 1
            continue
        new_hands = redeal_removed_hands(
            [hands_by_id[pid] for pid in removed_ids], round_number + 1, seed, deal_rules)
        if new_hands is None:
            note = (f"{len(removed_ids)} removed hands could not be redealt into different, "
                    "rule-satisfying hands")
            print(f"  {note}; their cards are dropped")
            rounds.append({"round": round_number + 1, "prompt_ids": [], "note": note})
            utils.append_jsonl(rounds[-1], rounds_path)
            break
        first_index = _next_hand_index(hands)
        new_records = [
            {"prompt_id": record["prompt_id"], "round": round_number + 1, "variables": record["variables"]}
            for record in cp.hand_records(new_hands, first_index)
        ]
        for record in new_records:
            hands_by_id[record["prompt_id"]] = record
        hands.extend(new_records)
        utils.save_jsonl(hands, hands_path)   # replaces any hands from an earlier interrupted redeal
        rounds.append({"round": round_number + 1, "prompt_ids": [r["prompt_id"] for r in new_records],
                       "note": f"redeal of {len(removed_ids)} removed hands"})
        utils.append_jsonl(rounds[-1], rounds_path)
        round_number += 1

    return _finish(output_dir, sdf, hands, rounds, started_all)


def _finish(output_dir: Path, sdf: dict, hands: list[dict], rounds: list[dict], started_all: float) -> list[dict]:
    """Write the accepted hands and the summary; return the accepted hand records."""
    rejected = load_rejections(output_dir)
    last_round = max(r["round"] for r in rounds if r["prompt_ids"])
    last_ids = {pid for r in rounds if r["round"] == last_round for pid in r["prompt_ids"]}
    leftover_ids = sorted(pid for pid in rejected if pid in last_ids)
    dealt_ids = {pid for r in rounds for pid in r["prompt_ids"]}
    accepted = [
        {"prompt_id": h["prompt_id"], "variables": h["variables"]}
        for h in hands if h["prompt_id"] in dealt_ids and h["prompt_id"] not in rejected
    ]
    utils.save_jsonl(accepted, output_dir / SCREENED_HANDS_FILENAME)
    summary = {
        "dealt": len(rounds[0]["prompt_ids"]),
        "accepted": len(accepted),
        "leftover_dropped": len(leftover_ids),
        "leftover_prompt_ids": leftover_ids,
        "rounds": [
            {"round": r["round"], "hands": len(r["prompt_ids"]),
             "removed": sum(pid in rejected for pid in r["prompt_ids"]), "note": r["note"]}
            for r in rounds
        ],
    }
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Screen done: {summary['accepted']} of {summary['dealt']} dealt hands accepted over "
          f"{len(rounds)} round(s); {len(leftover_ids)} dropped after the last round "
          f"({time.monotonic() - started_all:.1f}s)")
    return accepted


# ---------------------------------------------------------------- reporting

def load_screen_outputs(run_dir: Path) -> dict | None:
    """The screen's results for a run, or None when it never ran (older runs,
    which screened planned descriptions and have no screen_hands.jsonl).

    Returns {"hands": every hand ever dealt, "rounds", "removed": {prompt_id:
    rejection record}, "screened_ids": ids in a checkpointed batch,
    "accepted_ids": set or None until the screen finishes, "leftover_ids",
    "dealt_hands": the round-0 hands, "batches_total", "batches_done"}."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
    hands = utils.load_jsonl(stage_dir / HANDS_FILENAME)
    if not hands:
        return None
    batches = utils.load_jsonl(stage_dir / BATCHES_FILENAME)
    checkpoint = utils.Checkpoint(stage_dir / CHECKPOINT_FILENAME)
    done = [b for b in batches if checkpoint.is_done(b["batch_id"])]
    accepted = utils.load_jsonl(stage_dir / SCREENED_HANDS_FILENAME)
    summary_path = stage_dir / SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "hands": hands,
        "rounds": utils.load_jsonl(stage_dir / ROUNDS_FILENAME),
        "dealt_hands": [h for h in hands if h["round"] == 0],
        "screened_ids": {pid for b in done for pid in b["prompt_ids"]},
        "removed": load_rejections(stage_dir),
        "accepted_ids": {h["prompt_id"] for h in accepted} if summary else None,
        "leftover_ids": set(summary["leftover_prompt_ids"]) if summary else set(),
        "batches_total": len(batches),
        "batches_done": len(done),
    }


def removal_rate_rows(screen: dict, axes: list[str] | None = None) -> list[dict]:
    """Removal rate per variable value over every screened hand, counting every
    round (a redealt hand is a fresh hand to screen).

    Each row: variable, value, screened, removed, rate, corpus_rate, flagged
    (rate above FLAG_RATE_MULTIPLE x the corpus rate with at least
    MIN_REMOVED_TO_FLAG removals). Sorted flagged first, then by rate."""
    screened = [h for h in screen["hands"] if h["prompt_id"] in screen["screened_ids"]]
    removed_ids = set(screen["removed"])
    corpus_rate = (sum(h["prompt_id"] in removed_ids for h in screened) / len(screened)) if screened else 0.0
    counts: dict[tuple[str, str], list[int]] = {}
    for hand in screened:
        for axis, value in hand["variables"].items():
            if axes is not None and axis not in axes:
                continue
            cell = counts.setdefault((axis, value), [0, 0])
            cell[0] += 1
            cell[1] += hand["prompt_id"] in removed_ids
    rows = []
    for (axis, value), (n_screened, n_removed) in counts.items():
        rate = n_removed / n_screened
        rows.append({
            "variable": axis, "value": value, "screened": n_screened, "removed": n_removed,
            "rate": rate, "corpus_rate": corpus_rate,
            "flagged": n_removed >= MIN_REMOVED_TO_FLAG and rate > FLAG_RATE_MULTIPLE * corpus_rate,
        })
    rows.sort(key=lambda r: (not r["flagged"], -r["rate"], r["variable"], r["value"]))
    return rows


def round_rows(screen: dict) -> list[dict]:
    """One row per round: hands screened, hands removed."""
    removed_ids = set(screen["removed"])
    return [
        {"round": r["round"], "hands": len(r["prompt_ids"]),
         "removed": sum(pid in removed_ids for pid in r["prompt_ids"]), "note": r.get("note", "")}
        for r in screen["rounds"]
    ]


def share_shift_rows(screen: dict) -> list[dict]:
    """Per variable value, its share of the dealt hands against its share of the
    accepted hands: the shift caused by the leftover hands dropped after the last
    round. Empty until the screen has finished. Sorted by absolute shift."""
    if screen["accepted_ids"] is None:
        return []
    dealt = screen["dealt_hands"]
    accepted = [h for h in screen["hands"] if h["prompt_id"] in screen["accepted_ids"]]
    counts: dict[tuple[str, str], list[int]] = {}
    for position, group in enumerate((dealt, accepted)):
        for hand in group:
            for axis, value in hand["variables"].items():
                counts.setdefault((axis, value), [0, 0])[position] += 1
    rows = []
    for (axis, value), (n_dealt, n_accepted) in counts.items():
        dealt_share = n_dealt / len(dealt) if dealt else 0.0
        final_share = n_accepted / len(accepted) if accepted else 0.0
        rows.append({
            "variable": axis, "value": value, "dealt": n_dealt, "accepted": n_accepted,
            "dealt_share": dealt_share, "final_share": final_share, "shift": final_share - dealt_share,
        })
    rows.sort(key=lambda r: (-abs(r["shift"]), r["variable"], r["value"]))
    return rows
