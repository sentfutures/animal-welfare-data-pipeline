"""Layer 1 (deal variables and screen): the driver ``run`` loops the deal (``deal_variables.py``), the screen (``screen.py``) and the redeal (``redeal.py``).
Every round's deal and judgments are saved before the next round, so a resumed run repeats identical rounds; the stage's files and
config are described in ``sdf_pipeline/CLAUDE.md``. What the steps share is in ``shared_by_this_layers_steps.py``."""

import json
import time
from pathlib import Path

from shared import utils
from shared.sdf_run_layouts import DEAL_VARIABLES_AND_SCREEN
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import run_records_value_ids, value_ids_field_of
from sdf_pipeline.shared_step_machinery.step_record import Run
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import *  # noqa: F401,F403  (the file names, readers and row builders)
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import (
    DEFAULT_BATCH_SIZE, DEFAULT_REDEAL_ROUNDS, HANDS_FILENAME, ROUNDS_FILENAME, SCREENED_HANDS_FILENAME,
    SUMMARY_FILENAME, load_rejections)
from sdf_pipeline.layer1_deal_variables_and_screen.deal_variables import deal_original_hands
from sdf_pipeline.layer1_deal_variables_and_screen.redeal import redeal_screened_out_hands
from sdf_pipeline.layer1_deal_variables_and_screen.screen import *  # noqa: F401,F403  (the screen's prompt, parsing and request helpers)
from sdf_pipeline.layer1_deal_variables_and_screen import screen
from sdf_pipeline.layer1_deal_variables_and_screen.screen import load_screen_prompts, reason_screen_is_inactive
from sdf_pipeline.shared_step_machinery.step_runner import find_pending, run_step


def run(config: dict, prompts_dir: Path, output_dir: Path) -> list[dict] | None:
    """The accepted hand records ({"prompt_id", "variables"}) for the description write, or None, making no calls, when the screen is off or the snapshot has no screen template."""
    sdf = config["sdf"]
    reason_inactive = reason_screen_is_inactive(config, prompts_dir)
    if reason_inactive is not None:
        print(f"  {reason_inactive}")
        return None
    accepted_path = output_dir / SCREENED_HANDS_FILENAME
    if accepted_path.exists():
        accepted = utils.load_jsonl(accepted_path)
        print(f"  Screen already complete: {len(accepted)} hands accepted (no calls)")
        return accepted

    seed = sdf.get("seed", 0)
    batch_size = int(sdf.get("screen_batch_size", DEFAULT_BATCH_SIZE))
    redeal_rounds = int(sdf.get("screen_redeal_rounds", DEFAULT_REDEAL_ROUNDS))
    _, _, value_ids = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
    deal_rules = cp.load_deal_rules(prompts_dir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
    hands_path = output_dir / HANDS_FILENAME
    rounds_path = output_dir / ROUNDS_FILENAME
    record_value_ids = run_records_value_ids(output_dir)   # decided per run, by its manifest
    run_of_screen = Run(config, prompts_dir, {DEAL_VARIABLES_AND_SCREEN: output_dir})

    hands = utils.load_jsonl(hands_path)
    rounds = utils.load_jsonl(rounds_path)
    if not rounds:  # round 0: the original deal, saved before anything is screened
        template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
        values, weights, _ = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
        hands = deal_original_hands(
            template, values, weights, sdf["n_prompts"], seed, deal_rules, value_ids, record_value_ids)
        utils.save_jsonl(hands, hands_path)   # replaces any partial write from an interrupted deal
        rounds = [{"round": 0, "prompt_ids": [h["prompt_id"] for h in hands], "note": "original deal"}]
        utils.append_jsonl(rounds[0], rounds_path)
    recorded_ids = {pid for r in rounds for pid in r["prompt_ids"]}
    hands = [h for h in hands if h["prompt_id"] in recorded_ids]   # hands saved before their round record are dealt again
    hands_by_id = {h["prompt_id"]: h for h in hands}

    started_all = time.monotonic()
    round_number = 0
    while True:
        round_record = rounds[round_number]
        round_ids = round_record["prompt_ids"]
        round_hands = [hands_by_id[prompt_id] for prompt_id in round_ids]
        batches, pending_batches = find_pending(screen.STEP, run_of_screen, round_hands)   # saves the round's batches
        print(f"  Screen round {round_number}: {len(round_ids)} hands in {len(batches)} batches of up to "
              f"{batch_size} ({len(batches) - len(pending_batches)} already done)")

        round_started = time.monotonic()
        run_step(screen.STEP, run_of_screen, round_hands)

        rejected = load_rejections(output_dir)
        removed_ids = [pid for pid in round_ids if pid in rejected]
        print(f"  Round {round_number} removed {len(removed_ids)} of {len(round_ids)} hands "
              f"({time.monotonic() - round_started:.1f}s; {time.monotonic() - started_all:.1f}s total)")
        if not removed_ids or round_number >= redeal_rounds:
            break
        if round_number + 1 < len(rounds):  # the next round was dealt before an interruption
            round_number += 1
            continue
        new_records = redeal_screened_out_hands(
            hands, removed_ids, round_number, seed, deal_rules, value_ids, record_value_ids)
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
