"""Layer 2 step: compose the prompt matrix offline, then plan each document (a description spec or INCOHERENT; only the description travels on).
prompts.jsonl is composed once and reloaded on ``--resume`` (the run's snapshot is authority); malformed output is not recorded, so ``--resume`` retries it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import utils
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import (
    announce_hands_with_a_value_that_has_no_unique_id, deal_hand_records_with_value_ids, run_records_value_ids, value_ids_field_of, value_ids_of_hands)
from sdf_pipeline.layer1_deal_variables_and_screen.screen import reason_screen_is_inactive
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import PROMPTS, SCREENED_HANDS
from sdf_pipeline.shared_step_machinery.step_record import Run, StageFile, Step, description_write_prompt_files
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, Removed, StepProgress,
    line_for_document_left_to_retry)
from shared.sdf_run_layouts import DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE


PLANS = StageFile(DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, "plans.jsonl", "prompt_id")


def build_request(documents: list, run: Run) -> ModelRequest:
    """The prompt the composer already wrote for the prompt record (``prompt_id``, ``variables``, ``prompt``, ``system``)."""
    (prompt_record,) = documents
    return STEP.model_request(prompt_record, prompt_record["prompt"], prompt_record.get("system"), run)


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """A reply that declares the combination INCOHERENT is ``Removed``; its record is still written to plans.jsonl."""
    (prompt_record,) = documents
    if reply.stop_reason != "end_turn":
        outcome = Failed(FailureKind.UNUSABLE_ANSWER, f"truncated description write (stop_reason={reply.stop_reason})")
    else:
        description = cp.extract_description(reply.text)
        if cp.is_incoherent(reply.text):
            outcome = Removed("incoherent", {"plan": reply.text, "description": description})
        elif description is None:
            outcome = Failed(FailureKind.UNUSABLE_ANSWER, "no DOCUMENT DESCRIPTION found")
        else:
            outcome = Accepted({"plan": reply.text, "description": description})
    return {prompt_record["prompt_id"]: outcome}


def _plan_record(prompt_record: dict, plan: str, description, incoherent: bool) -> dict:
    return {
        "prompt_id": prompt_record["prompt_id"],
        "variables": prompt_record["variables"],
        **value_ids_field_of(prompt_record),
        "plan": plan,
        "description": description,
        "incoherent": incoherent,
    }


def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
    """An accepted description write, and an INCOHERENT one (a finished one whose record says so), are one record each."""
    (prompt_record,) = documents
    outcome = outcomes[prompt_record["prompt_id"]]
    if isinstance(outcome, Accepted):
        return OutcomeRecords(((PLANS.file_name, _plan_record(
            prompt_record, outcome.update["plan"], outcome.update["description"], False)),))
    if isinstance(outcome, Removed):
        return OutcomeRecords(((PLANS.file_name, _plan_record(
            prompt_record, outcome.detail["plan"], outcome.detail["description"], True)),))
    return OutcomeRecords()


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (prompt_record,), (outcome,) = documents, outcomes.values()
    prompt_id = prompt_record["prompt_id"]
    if isinstance(outcome, Accepted):
        return f"  Wrote description for {prompt_id}"
    if isinstance(outcome, Removed):
        return f"  Wrote description for {prompt_id} (INCOHERENT)"
    if outcome.kind is FailureKind.THROWN_CALL:
        return line_for_document_left_to_retry(prompt_id, f"API call failed ({outcome.reason})")
    return line_for_document_left_to_retry(prompt_id, outcome.reason)


def _screened_hands_or_prompts(run: Run) -> StageFile:
    """What seeding selects from: planning composes prompts from the screened hands, or reads the composed ones."""
    return PROMPTS if reason_screen_is_inactive(run.config, run.prompts_directory) else SCREENED_HANDS


STEP = Step(
    name="description_write", folder_key=DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, model_setting_keys=("description_write_model",),
    required_prompt_files=description_write_prompt_files,
    reads=lambda run: run.records(PROMPTS), seeds_from=_screened_hands_or_prompts,
    build_request=build_request, read_reply=read_reply, records=records, writes=(PLANS,), progress_line=progress_line)


def run(
    config: dict, prompts_dir: Path, output_dir: Path, hands: list[dict] | None = None
) -> list[dict]:
    """Plan every hand; without ``hands`` (screen disabled) the composer deals its own."""
    prompts_path = output_dir / PROMPTS.file_name
    sdf = config["sdf"]

    if prompts_path.exists():
        prompts = utils.load_jsonl(prompts_path)
        print(f"  Reusing {len(prompts)} composed prompts from this run")
    else:
        template = utils.sdf_template_path(prompts_dir, 1).read_text(encoding="utf-8")
        values, weights, value_ids = cp.load_variables(prompts_dir / "variables.txt")
        preamble = (prompts_dir / "preamble.txt").read_text(encoding="utf-8")
        deal_rules = cp.load_deal_rules(prompts_dir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
        if hands is None and sdf["n_prompts"] is not None:
            # Screen off: deal with the same function the screen's deal step calls, then compose
            # exactly those hands.
            dealt_hands, ids_by_hand, notes_by_hand = deal_hand_records_with_value_ids(
                template, values, weights, sdf["n_prompts"], sdf.get("seed", 0), deal_rules, value_ids)
            announce_hands_with_a_value_that_has_no_unique_id(notes_by_hand, len(dealt_hands))
            if run_records_value_ids(output_dir):
                dealt_hands = [{**hand, "variable_value_ids": ids_by_hand[hand["prompt_id"]]} for hand in dealt_hands]
            hands = dealt_hands
        prompts = list(cp.compose_records(
            template, values, weights, preamble,
            n_prompts=sdf["n_prompts"],
            seed=sdf.get("seed", 0),
            entity_seed=sdf.get("entity_pool_seed", 137),
            deal_rules=deal_rules,
            hands=hands,
            value_ids=value_ids,
        ))
        if hands is None and run_records_value_ids(output_dir):   # full matrix: no deal, but each hand's ids
            ids_by_hand, notes_by_hand = value_ids_of_hands(prompts, value_ids)
            announce_hands_with_a_value_that_has_no_unique_id(notes_by_hand, len(prompts))
            prompts = [{"prompt_id": r["prompt_id"], "variables": r["variables"],
                        "variable_value_ids": ids_by_hand[r["prompt_id"]], "system": r["system"], "prompt": r["prompt"]}
                       for r in prompts]
        for rec in prompts:
            utils.append_jsonl(rec, prompts_path)
        print(f"  Composed {len(prompts)} prompts (offline, $0.00)")

    results = run_step(STEP, Run(config, prompts_dir, {DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE: output_dir}), prompts).records_in_input_order
    incoherent_n = sum(1 for r in results if r.get("incoherent"))
    if incoherent_n:
        print(f"  {incoherent_n} combination(s) declared INCOHERENT — dropped")
    return results
