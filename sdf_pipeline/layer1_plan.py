"""Layer 1: compose the prompt matrix (offline), then plan each document.

The composition half replaces the old LLM layers entirely: a deck sample
of the variables matrix (prompts/sdf/variables.txt x layer1.txt) yields one
fully-specified prompt per planned document, at zero API cost. The plan half
sends each composed prompt to the model, which works through the template's
questions and emits a self-contained DOCUMENT DESCRIPTION spec (or declares
the combination INCOHERENT). Only the extracted description travels
downstream; the working notes are scaffolding.

Checkpoint semantics (protects paid work):
- The implausibility screen (layer1_screen.py) runs before this module and hands
  it the accepted hands; removed hands are never planned.
- prompts.jsonl is composed once per run and reloaded on --resume (the deck
  sample is seeded, but reusing the file guarantees identity even if the
  variables file later changes in the repo — the run's snapshot is authority).
- A plan is checkpointed when it yields a description OR a deliberate
  INCOHERENT; malformed output (no tags, truncation) is NOT checkpointed, so
  --resume retries exactly the failed calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline.dealt_value_identifiers import (
    deal_hand_records_with_value_ids, run_records_value_ids, value_ids_field_of, value_ids_of_hands)
from sdf_pipeline.step_runner import StepRunner
from sdf_pipeline.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, Removed,
    StepPolicy, StepProgress, line_for_document_left_to_retry)


class PlanStep(DocumentStep):
    """Plan one composed prompt. The document is the prompt record (``prompt_id``, ``variables``,
    ``prompt``, ``system``). The step sends the prompt the composer already wrote; a plan that
    declares the combination INCOHERENT is a ``Removed`` outcome whose record is still written to
    plans.jsonl and checkpointed, as it always was."""

    step_name = "plan"
    stage_file_name = "plans.jsonl"
    stage_file_id_field = "prompt_id"
    # Only a thrown call counts towards the systemic stop; an unusable answer is left behind.
    policy = StepPolicy()
    stage_tag = "layer1_plan"

    def __init__(self, config: dict):
        self.plan_model = config["sdf"].get("plan_model")

    def build_request(self, prompt_record: dict) -> ModelRequest:
        return self.model_request(
            prompt_record, prompt_record["prompt"], prompt_record.get("system"), self.plan_model)

    def interpret(self, prompt_record: dict, reply: ModelReply):
        if reply.stop_reason != "end_turn":
            return Failed(FailureKind.UNUSABLE_ANSWER, f"truncated plan (stop_reason={reply.stop_reason})")
        description = cp.extract_description(reply.text)
        if cp.is_incoherent(reply.text):
            return Removed("incoherent", {"plan": reply.text, "description": description})
        if description is None:
            return Failed(FailureKind.UNUSABLE_ANSWER, "no DOCUMENT DESCRIPTION found")
        return Accepted({"plan": reply.text, "description": description})

    def _record(self, prompt_record: dict, plan: str, description, incoherent: bool) -> dict:
        return {
            "prompt_id": prompt_record["prompt_id"],
            "variables": prompt_record["variables"],
            **value_ids_field_of(prompt_record),
            "plan": plan,
            "description": description,
            "incoherent": incoherent,
        }

    def stage_file_record(self, prompt_record: dict, update: dict) -> dict:
        return self._record(prompt_record, update["plan"], update["description"], False)

    def outcome_records(self, prompt_record: dict, outcome) -> OutcomeRecords:
        """An INCOHERENT plan is a finished plan whose record says so; everything else as the default."""
        if isinstance(outcome, Removed):
            return OutcomeRecords(((
                self.stage_file_name,
                self._record(prompt_record, outcome.detail["plan"], outcome.detail["description"], True)),), True)
        return super().outcome_records(prompt_record, outcome)

    def progress_line(self, prompt_record: dict, outcome, progress: StepProgress) -> str | None:
        prompt_id = prompt_record["prompt_id"]
        if isinstance(outcome, Accepted):
            return f"  Planned {prompt_id}"
        if isinstance(outcome, Removed):
            return f"  Planned {prompt_id} (INCOHERENT)"
        if outcome.kind is FailureKind.THROWN_CALL:
            return line_for_document_left_to_retry(prompt_id, f"API call failed ({outcome.reason})")
        return line_for_document_left_to_retry(prompt_id, outcome.reason)


def run(
    config: dict, prompts_dir: Path, output_dir: Path, hands: list[dict] | None = None
) -> list[dict]:
    """Plan every hand. ``hands`` are the hand records the implausibility screen
    accepted ({"prompt_id", "variables"}); without them (screen disabled) the
    composer deals its own hands."""
    prompts_path = output_dir / "prompts.jsonl"
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
            dealt_hands, ids_by_hand, _ = deal_hand_records_with_value_ids(
                template, values, weights, sdf["n_prompts"], sdf.get("seed", 0), deal_rules, value_ids)
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
            ids_by_hand, _ = value_ids_of_hands(prompts, value_ids)
            prompts = [{"prompt_id": r["prompt_id"], "variables": r["variables"],
                        "variable_value_ids": ids_by_hand[r["prompt_id"]], "system": r["system"], "prompt": r["prompt"]}
                       for r in prompts]
        for rec in prompts:
            utils.append_jsonl(rec, prompts_path)
        print(f"  Composed {len(prompts)} prompts (offline, $0.00)")

    runner = StepRunner(PlanStep(config), output_dir, workers=config.get("workers", 1))
    results = runner.run(prompts).records_in_input_order
    incoherent_n = sum(1 for r in results if r.get("incoherent"))
    if incoherent_n:
        print(f"  {incoherent_n} combination(s) declared INCOHERENT — dropped")
    return results
