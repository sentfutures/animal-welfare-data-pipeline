"""Layer 1 screen: a batch incoherence check on the planned specifications.

Runs after every layer-1 plan has finished and before drafting. The planner
sees one deal at a time, so a contradictory specification can slip through its
own INCOHERENT gate; the screen shows the model a batch of numbered
specifications (default 10) and asks which could not be written as a realistic
document. It answers with local numbers 1..N, never prompt ids; this module
maps the numbers back. Removed specifications are written to
screen_rejections.jsonl with the model's reason and are left out downstream,
like INCOHERENT plans.

The screening logic is importable on its own: ``screen_descriptions`` takes a
list of descriptions (and optionally their dealt variables) and returns one
judgment per description (the removals are those with ``remove`` true), using
the same prompt files, answer schema and validation as the pipeline stage, so
offline validation scripts can reuse it unchanged. The model's answer is
schema-constrained JSON, one entry per specification (reasoning first, then the
decision); every entry is kept in screen_judgments.jsonl so keeps are auditable.

Config (``sdf`` section): ``screen_enabled`` (default true), ``screen_model``
(falls back to ``plan_model``, then the global model), ``screen_batch_size``
(default 10), ``screen_show_variables`` (default false: the screen sees only
the descriptions, which is all that travels downstream; true also shows each
specification's dealt variables).

Checkpoint semantics (protects paid work):
- screen_batches.jsonl is written BEFORE any call, so a resumed run screens
  identical batches. Specifications planned after it was written (plans
  retried on resume) get new batches appended; existing batches never change.
- One checkpoint per batch. A malformed, truncated, miscounted, duplicate or
  out-of-range answer is NOT checkpointed, so --resume retries exactly that batch.
- A batch's rejections are appended before its checkpoint, so a crash between
  the two re-screens the batch; readers de-duplicate by prompt_id.
"""

import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from shared.matrix import split_sections

STAGE_TAG = "layer1_screen"
TEMPLATE_FILENAME = "layer1_screen.txt"
VARIABLES_NOTE_FILENAME = "layer1_screen_variables_note.txt"
DEALT_VARIABLES_FILENAME = "layer1_screen_dealt_variables.txt"
SPECIFICATION_BLOCK_FILENAME = "layer1_screen_specification_block.txt"
SCHEMA_FILENAME = "layer1_screen_answer_schema.json"
JUDGMENTS_FILENAME = "screen_judgments.jsonl"
BATCHES_FILENAME = "screen_batches.jsonl"
REJECTIONS_FILENAME = "screen_rejections.jsonl"
CHECKPOINT_FILENAME = "_screen_checkpoint.json"
DEFAULT_BATCH_SIZE = 10
_MAX_TOKENS = 2000  # a short list of one-line reasons; bounds api-backend cost

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
    variables_note: str
    dealt_variables: str
    specification_block: str
    answer_schema: dict


def load_screen_prompts(prompts_dir: Path) -> ScreenPrompts | None:
    """The screen's prompt files, or None for a run whose snapshot predates the screen."""
    directory = Path(prompts_dir)
    if not (directory / TEMPLATE_FILENAME).is_file():
        return None
    read = lambda name: (directory / name).read_text(encoding="utf-8")  # noqa: E731
    return ScreenPrompts(
        template=read(TEMPLATE_FILENAME),
        variables_note=read(VARIABLES_NOTE_FILENAME),
        dealt_variables=read(DEALT_VARIABLES_FILENAME),
        specification_block=read(SPECIFICATION_BLOCK_FILENAME),
        answer_schema=json.loads(read(SCHEMA_FILENAME)),
    )


def build_answer_schema(prompts: ScreenPrompts, specification_count: int) -> dict:
    """The answer schema with the ``number`` enum filled for this batch (1..N)."""
    schema = copy.deepcopy(prompts.answer_schema)
    schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] = list(
        range(1, specification_count + 1)
    )
    return schema


def build_screen_prompt(
    prompts: ScreenPrompts,
    descriptions: list[str],
    variables_list: list[dict] | None = None,
) -> tuple[str | None, str]:
    """Render (system, user) for one batch. ``variables_list`` (parallel to
    descriptions) is shown only when given."""
    if variables_list is not None and len(variables_list) != len(descriptions):
        raise ValueError("variables_list must be parallel to descriptions")
    blocks = []
    for number, description in enumerate(descriptions, 1):
        dealt = ""
        if variables_list is not None:
            lines = "\n".join(f"- {name}: {value}" for name, value in variables_list[number - 1].items())
            dealt = prompts.dealt_variables.format(variables=lines)
        blocks.append(prompts.specification_block.format(
            number=number, description=description.strip(), dealt_variables=dealt))
    rendered = prompts.template.format(
        specification_count=len(descriptions),
        variables_note=prompts.variables_note if variables_list is not None else "",
        specifications="\n\n".join(blocks),
    )
    return split_sections(rendered)


def parse_screen_answer(raw: str, specification_count: int) -> list[ScreenJudgment]:
    """One judgment per specification, ordered by number. Raises ScreenAnswerError
    unless the answer is JSON of the schema's shape with exactly
    specification_count entries, numbers 1..N each exactly once, and a non-empty
    reason on every entry marked for removal."""
    try:
        data = json.loads(raw or "")
        entries = data["specifications"]
    except (ValueError, KeyError, TypeError) as e:
        raise ScreenAnswerError(f"answer is not the expected JSON ({type(e).__name__}: {e})") from e
    if not isinstance(entries, list) or len(entries) != specification_count:
        raise ScreenAnswerError(
            f"expected {specification_count} entries, got "
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
        if not 1 <= number <= specification_count:
            raise ScreenAnswerError(f"number {number} is outside 1..{specification_count}")
        if number in judgments:
            raise ScreenAnswerError(f"number {number} listed twice")
        if remove and not reason.strip():
            raise ScreenAnswerError(f"number {number} is marked for removal without a reason")
        judgments[number] = ScreenJudgment(
            number=number, reasoning=reasoning.strip(), remove=remove, reason=reason.strip())
    return [judgments[n] for n in range(1, specification_count + 1)]


def screen_descriptions(
    descriptions: list[str],
    variables_list: list[dict] | None = None,
    *,
    prompts: ScreenPrompts,
    model: str | None = None,
    item_id: str | None = None,
) -> list[ScreenJudgment]:
    """Screen one batch of descriptions and return a judgment for every one
    (reasoning included, so keeps are auditable); the removals are the entries
    with ``remove`` true.

    The call is schema-constrained (number enum 1..N). Raises ScreenAnswerError
    for a truncated or invalid answer; API errors propagate. Uses the global
    temperature and tags the call ``layer1_screen`` for the cost log."""
    system, user = build_screen_prompt(prompts, descriptions, variables_list)
    raw, stop_reason = api.call_claude(
        user,
        system_prompt=system or "",
        model=model,
        max_tokens=_MAX_TOKENS,
        stage=STAGE_TAG,
        item_id=item_id,
        return_stop_reason=True,
        json_schema=build_answer_schema(prompts, len(descriptions)),
    )
    if stop_reason != "end_turn":
        raise ScreenAnswerError(f"truncated answer (stop_reason={stop_reason})")
    return parse_screen_answer(raw, len(descriptions))


# ---------------------------------------------------------------- batching

def plan_batches(
    output_dir: Path, planned_ids: list[str], batch_size: int, seed: int
) -> list[dict]:
    """The batches for this run, persisted before any call.

    A seeded shuffle (of the sorted ids, so input order is irrelevant) cut into
    batch_size chunks. On resume the saved batches are reused verbatim; ids
    planned since then are batched and appended."""
    path = output_dir / BATCHES_FILENAME
    batches = utils.load_jsonl(path)
    batched = {pid for batch in batches for pid in batch["prompt_ids"]}
    unbatched = sorted(pid for pid in planned_ids if pid not in batched)
    if not unbatched:
        return batches
    random.Random(f"{seed}:{STAGE_TAG}:{len(batches)}").shuffle(unbatched)
    for start in range(0, len(unbatched), batch_size):
        batch = {
            "batch_id": f"screen_batch_{len(batches):04d}",
            "prompt_ids": unbatched[start:start + batch_size],
        }
        batches.append(batch)
        utils.append_jsonl(batch, path)
    return batches


# ---------------------------------------------------------------- stage

def load_rejections(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: rejection record}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / REJECTIONS_FILENAME)}


def drop_rejected(config: dict, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans minus those a completed or partial screen already removed (no calls).
    Used when a run resumes past layer 1 and plans are re-read from disk."""
    if not config["sdf"].get("screen_enabled", True):
        return plans
    rejected = load_rejections(output_dir)
    return [p for p in plans if p["prompt_id"] not in rejected]


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Screen the planned (non-INCOHERENT) specifications; return ``plans`` minus
    the removed ones. A disabled screen, or a run whose prompt snapshot has no
    screen template, makes no calls and returns ``plans`` unchanged."""
    sdf = config["sdf"]
    if not sdf.get("screen_enabled", True):
        print("  Screen disabled (sdf.screen_enabled: false)")
        return plans
    prompts = load_screen_prompts(prompts_dir)
    if prompts is None:
        print(f"  No {TEMPLATE_FILENAME} in this run's prompt snapshot — screen skipped")
        return plans

    show_variables = bool(sdf.get("screen_show_variables", False))
    batch_size = int(sdf.get("screen_batch_size", DEFAULT_BATCH_SIZE))
    model = sdf.get("screen_model") or sdf.get("plan_model")
    checkpoint = utils.Checkpoint(output_dir / CHECKPOINT_FILENAME)
    rejections_path = output_dir / REJECTIONS_FILENAME
    judgments_path = output_dir / JUDGMENTS_FILENAME

    by_id = {p["prompt_id"]: p for p in plans if p.get("description")}
    batches = plan_batches(output_dir, list(by_id), batch_size, sdf.get("seed", 0))
    # Batches can name ids that are no longer planned (plans file edited); screen what exists.
    pending = [b for b in batches if not checkpoint.is_done(b["batch_id"])]
    print(f"  Screening {len(by_id)} specifications in {len(batches)} batches of up to "
          f"{batch_size} ({len(batches) - len(pending)} already done)")

    def screen_one(batch: dict):
        members = [by_id[pid] for pid in batch["prompt_ids"] if pid in by_id]
        try:
            started = time.monotonic()
            judgments = screen_descriptions(
                [m["description"] for m in members],
                [m["variables"] for m in members] if show_variables else None,
                prompts=prompts, model=model, item_id=batch["batch_id"],
            )
            return members, judgments, time.monotonic() - started, None
        except ScreenAnswerError as e:
            return members, None, 0.0, f"unusable answer ({e})"
        except Exception as e:  # one failing call must not discard its siblings' work
            return members, None, 0.0, f"API call failed ({type(e).__name__}: {e})"

    workers = config.get("workers", 1)
    failed = 0
    removed_total = 0
    started_all = time.monotonic()
    for done, (batch, (members, judgments, seconds, problem)) in enumerate(
        zip(pending, utils.parallel_map(screen_one, pending, workers)), 1
    ):
        if problem:
            failed += 1
            print(f"  {batch['batch_id']}: {problem} — will retry on resume")
            continue
        for judgment in judgments:
            member = members[judgment.index]
            utils.append_jsonl({
                "prompt_id": member["prompt_id"],
                "batch_id": batch["batch_id"],
                "number": judgment.number,
                "reasoning": judgment.reasoning,
                "remove": judgment.remove,
                "reason": judgment.reason,
            }, judgments_path)
            if judgment.remove:
                utils.append_jsonl({
                    "prompt_id": member["prompt_id"],
                    "batch_id": batch["batch_id"],
                    "number": judgment.number,
                    "reason": judgment.reason,
                    "description": member["description"],
                }, rejections_path)
        removed = sum(j.remove for j in judgments)
        checkpoint.mark_done(batch["batch_id"])
        removed_total += removed
        print(f"  Screened {batch['batch_id']} ({done}/{len(pending)}): {len(members)} specifications, "
              f"{removed} removed in {seconds:.1f}s")

    if pending and failed == len(pending):
        raise SystemExit(
            "layer1_screen: every pending screen call failed — this is systemic "
            "(auth, backend, or network), not per-batch; fix and --resume."
        )
    rejected = load_rejections(output_dir)
    print(f"  Screen removed {len(rejected)} of {len(by_id)} specifications "
          f"({removed_total} this session, {time.monotonic() - started_all:.1f}s)")
    return [p for p in plans if p["prompt_id"] not in rejected]


# ---------------------------------------------------------------- reporting

def load_screen_outputs(run_dir: Path) -> dict | None:
    """The screen's results for a run, or None when it never ran (older runs).

    Returns {"screened_ids": set, "removed": {prompt_id: record}, "batches_total",
    "batches_done"}; only specifications in a checkpointed batch count as screened."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
    batches = utils.load_jsonl(stage_dir / BATCHES_FILENAME)
    if not batches:
        return None
    checkpoint = utils.Checkpoint(stage_dir / CHECKPOINT_FILENAME)
    done = [b for b in batches if checkpoint.is_done(b["batch_id"])]
    return {
        "screened_ids": {pid for b in done for pid in b["prompt_ids"]},
        "removed": load_rejections(stage_dir),
        "batches_total": len(batches),
        "batches_done": len(done),
    }


def removal_rate_rows(
    plans: list[dict], screen: dict, axes: list[str] | None = None
) -> list[dict]:
    """Removal rate per variable value over the screened specifications.

    Each row: variable, value, screened, removed, rate, corpus_rate, flagged
    (rate above FLAG_RATE_MULTIPLE x the corpus rate with at least
    MIN_REMOVED_TO_FLAG removals). Sorted flagged first, then by rate."""
    screened = [p for p in plans if p["prompt_id"] in screen["screened_ids"]]
    removed_ids = set(screen["removed"])
    corpus_rate = (sum(p["prompt_id"] in removed_ids for p in screened) / len(screened)) if screened else 0.0
    counts: dict[tuple[str, str], list[int]] = {}
    for plan in screened:
        for axis, value in (plan.get("variables") or {}).items():
            if axes is not None and axis not in axes:
                continue
            cell = counts.setdefault((axis, value), [0, 0])
            cell[0] += 1
            cell[1] += plan["prompt_id"] in removed_ids
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
