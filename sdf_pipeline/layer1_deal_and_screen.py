"""Layer 1 (deal and screen): deal the hands, screen them for extreme implausibility, redeal the removed ones.

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

The stage is three update steps, one module each in ``sdf_pipeline/steps/``: ``DealStep`` (``deal.py``, corpus step,
``Created`` for each hand of round 0), ``ScreenStep`` (``screen.py``, batch step, run by ``BatchStepRunner``) and
``RedealStep`` (``redeal.py``, corpus step, ``Removed`` for the screened-out hands and ``Created`` for the new
ones). This module holds the layer's driver, ``run``, the loop over the three. What the steps share, and the
readers of the screen's files used by the audit and the report, are in ``layer1_deal_and_screen_shared.py``.

Config (``sdf`` section): ``screen_enabled`` (default true), ``screen_model``
(falls back to ``plan_model``, then the global model), ``screen_batch_size``
(default 10), ``screen_redeal_rounds`` (default 3).

Files in the layer 1 folder (``layer1_deal_and_screen``) (every round's deal and judgments are saved
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

import json
import time
from pathlib import Path

from shared import utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline.dealt_value_identifiers import run_records_value_ids, value_ids_field_of
from sdf_pipeline.layer1_deal_and_screen_shared import *  # noqa: F401,F403  (the file names, readers and row builders)
from sdf_pipeline.layer1_deal_and_screen_shared import (
    CHECKPOINT_FILENAME, DEFAULT_BATCH_SIZE, DEFAULT_REDEAL_ROUNDS, HANDS_FILENAME,
    JUDGMENTS_FILENAME, REJECTIONS_FILENAME, ROUNDS_FILENAME, SCREENED_HANDS_FILENAME, SUMMARY_FILENAME,
    load_rejections)
from sdf_pipeline.steps.deal import DealStep
from sdf_pipeline.steps.redeal import RedealStep
from sdf_pipeline.steps.screen import *  # noqa: F401,F403  (the screen's prompt, parsing and request helpers)
from sdf_pipeline.steps.screen import ScreenBatch, ScreenStep, load_screen_prompts, plan_round_batches, reason_screen_is_inactive
from sdf_pipeline.step_runner import BatchStepRunner
from sdf_pipeline.update_steps import Created


def _drop_batch_records(path: Path, batch_id: str) -> None:
    """Remove a batch's earlier records from a jsonl file, so that screening the
    batch again replaces its verdicts instead of adding to them."""
    records = utils.load_jsonl(path)
    kept = [r for r in records if r["batch_id"] != batch_id]
    if len(kept) != len(records):
        utils.save_jsonl(kept, path)


def run(config: dict, prompts_dir: Path, output_dir: Path) -> list[dict] | None:
    """Deal the hands, screen them with redeal rounds, and return the accepted
    hand records ({"prompt_id", "variables"}) for planning. Returns None, making
    no calls, when the screen is disabled or the run's prompt snapshot has no
    screen template (the planner then deals its own hands as before)."""
    sdf = config["sdf"]
    reason_inactive = reason_screen_is_inactive(config, prompts_dir)
    if reason_inactive is not None:
        print(f"  {reason_inactive}")
        return None
    prompts = load_screen_prompts(prompts_dir)

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
    record_value_ids = run_records_value_ids(output_dir)   # decided per run, by its manifest
    screen_step = ScreenStep(prompts, model, batch_size, render_variable_lines)

    hands = utils.load_jsonl(hands_path)
    rounds = utils.load_jsonl(rounds_path)
    if not rounds:  # round 0: the original deal, saved before anything is screened
        template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
        values, weights, _ = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
        deal_outcomes = DealStep(
            template, values, weights, sdf["n_prompts"], seed, deal_rules, value_ids, record_value_ids).apply([])
        hands = [outcome.new_state for outcome in deal_outcomes.values()]
        utils.save_jsonl(hands, hands_path)   # replaces any partial write from an interrupted deal
        rounds = [{"round": 0, "prompt_ids": [h["prompt_id"] for h in hands], "note": "original deal"}]
        utils.append_jsonl(rounds[0], rounds_path)
    recorded_ids = {pid for r in rounds for pid in r["prompt_ids"]}
    hands = [h for h in hands if h["prompt_id"] in recorded_ids]   # hands saved before their round record are dealt again
    hands_by_id = {h["prompt_id"]: h for h in hands}

    def write_batch_outcomes(batch: ScreenBatch, outcomes: dict) -> None:
        judgment_records, rejection_records = screen_step.stage_file_records(batch, outcomes)
        _drop_batch_records(judgments_path, batch.batch_id)
        _drop_batch_records(rejections_path, batch.batch_id)
        for record in judgment_records:
            utils.append_jsonl(record, judgments_path)
        for record in rejection_records:
            utils.append_jsonl(record, rejections_path)

    runner = BatchStepRunner(
        screen_step, checkpoint, write_batch_outcomes, workers=config.get("workers", 1))

    started_all = time.monotonic()
    round_number = 0
    while True:
        round_record = rounds[round_number]
        round_ids = round_record["prompt_ids"]
        batch_records = plan_round_batches(output_dir, round_number, round_ids, batch_size, seed)
        pending_records = [b for b in batch_records if not checkpoint.is_done(b["batch_id"])]
        print(f"  Screen round {round_number}: {len(round_ids)} hands in {len(batch_records)} batches of up to "
              f"{batch_size} ({len(batch_records) - len(pending_records)} already done)")

        round_started = time.monotonic()
        runner.run([screen_step.batch_from_record(record, hands_by_id) for record in pending_records])

        rejected = load_rejections(output_dir)
        removed_ids = [pid for pid in round_ids if pid in rejected]
        print(f"  Round {round_number} removed {len(removed_ids)} of {len(round_ids)} hands "
              f"({time.monotonic() - round_started:.1f}s; {time.monotonic() - started_all:.1f}s total)")
        if not removed_ids or round_number >= redeal_rounds:
            break
        if round_number + 1 < len(rounds):  # the next round was dealt before an interruption
            round_number += 1
            continue
        redeal_outcomes = RedealStep(
            round_number, removed_ids, {pid: rejected[pid]["reason"] for pid in removed_ids}, seed, deal_rules,
            value_ids, record_value_ids
        ).apply(hands)
        new_records = [outcome.new_state for outcome in redeal_outcomes.values() if isinstance(outcome, Created)]
        if not new_records:
            note = (f"{len(removed_ids)} removed hands could not be redealt into different, "
                    "rule-satisfying hands")
            print(f"  {note}; their cards are dropped")
            rounds.append({"round": round_number + 1, "prompt_ids": [], "note": note})
            utils.append_jsonl(rounds[-1], rounds_path)
            break
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
        {"prompt_id": h["prompt_id"], "variables": h["variables"], **value_ids_field_of(h)}
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
