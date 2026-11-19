"""Layer 4: score each rewrite and gate the final corpus. The judge scores alignment, realism and spec_conformance (1-10 each); the gate keeps documents
with alignment AND realism >= ``sdf.min_score_threshold`` (spec_conformance is advisory), then a near-duplicate removal runs. A response that fails to parse
is recorded as 5/5/5 "Parse error." and kept (a re-run would re-bill without new information); the score report flags these for review."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import constitution_loader, textstats, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.step_record import (
    LAYER4, Run, StageFile, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, ModelReply, ModelRequest, StepProgress, line_for_document_left_to_retry,
    reason_without_error_prefix)

PARSE_ERROR_SCORES = {"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."}

SCORES = StageFile(LAYER4, "scores.jsonl", "doc_id")


@dataclass(frozen=True)
class ScoreInputs:
    """What every score request is built from, loaded once per run."""
    constitution_claude: str
    template_path: Path
    render_variable_lines: Callable


@loaded_once_per_run
def _inputs(run: Run) -> ScoreInputs:
    template_path = utils.sdf_template_path(run.prompts_directory, 4)
    return ScoreInputs(
        constitution_claude=constitution_loader.load_constitution_claude(
            utils.resolve_constitution_dir(run.prompts_directory)),
        template_path=template_path,
        render_variable_lines=cp.render_variable_lines_for_template(
            template_path.read_text(encoding="utf-8"), run.prompts_directory, run.config))


def build_request(documents: list, run: Run) -> ModelRequest:
    (rewrite,) = documents
    inputs = _inputs(run)
    system, user = cp.split_sections(utils.load_prompt(
        inputs.template_path,
        constitution_claude=inputs.constitution_claude,
        document_description=rewrite["description"],
        variable_lines=inputs.render_variable_lines({"prompt_id": rewrite["doc_id"], "variables": rewrite["variables"]}),
        improved_document=rewrite["content"],
    ))
    return STEP.model_request(
        rewrite, user, system, run,
        cache_system=True,  # constitution + rubric are identical across scoring calls
    )


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """An unparseable answer is accepted as 5/5/5 "Parse error."; only a thrown call is left behind (and counts
    towards the systemic stop)."""
    (rewrite,) = documents
    try:
        scores = utils.extract_json_object(reply.text)
    except json.JSONDecodeError:
        scores = PARSE_ERROR_SCORES
    return {rewrite["doc_id"]: Accepted({
        "alignment": scores.get("alignment", 0),
        "realism": scores.get("realism", 0),
        "spec_conformance": scores.get("spec_conformance", 0),
        "notes": scores.get("notes", ""),
    })}


def record_of(rewrite: dict, update: dict, run: Run) -> dict:
    return {"doc_id": rewrite["doc_id"], "variables": rewrite["variables"], "scores": update}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (rewrite,), (outcome,) = documents, outcomes.values()
    doc_id = rewrite["doc_id"]
    if isinstance(outcome, Failed):
        return line_for_document_left_to_retry(
            doc_id, f"API call failed ({reason_without_error_prefix(outcome.reason)})")
    s = outcome.update
    return f"  Scored {doc_id} A{s['alignment']} R{s['realism']} S{s['spec_conformance']}"


STEP = Step(
    name="score", folder_key=LAYER4, stage_tag="layer4_score", model_setting_keys=("score_model",),
    build_request=build_request, read_reply=read_reply,
    records=one_record_per_accepted_document(SCORES, record_of), writes=(SCORES,),
    checkpoint_file_name="_checkpoint.json", progress_line=progress_line)


def keep_documents_at_or_above_threshold(score_records: list[dict], threshold) -> list[dict]:
    """The score gate (no model call): alignment AND realism at or above the threshold."""
    return [record for record in score_records
            if record["scores"]["alignment"] >= threshold and record["scores"]["realism"] >= threshold]


def remove_near_duplicates(records: list[dict], near_duplicate_threshold) -> tuple[list[dict], list[dict]]:
    """``(records kept, records dropped as {"doc_id", "kept_doc_id", "similarity"})``, keeping the first of each pair in the
    order given; a falsy threshold removes nothing."""
    if not near_duplicate_threshold or len(records) < 2:
        return records, []
    _, dropped_entries = textstats.near_dup_filter([record["content"] for record in records], near_duplicate_threshold)
    dropped_by_index = {entry["index"]: entry for entry in dropped_entries}
    kept = [record for index, record in enumerate(records) if index not in dropped_by_index]
    dropped = [{"doc_id": records[index]["doc_id"], "kept_doc_id": records[entry["kept_index"]]["doc_id"],
                "similarity": entry["similarity"]}
               for index, entry in sorted(dropped_by_index.items())]
    return kept, dropped


def run(
    config: dict,
    prompts_dir: Path,
    output_dir: Path,
    final_dir: Path,
    rewrites: list[dict],
) -> list[dict]:
    """Score, gate and cull the rewrites and write the final corpus; returns its records."""
    final_path = final_dir / "sdf_corpus.jsonl"
    sdf = config["sdf"]
    threshold = sdf["min_score_threshold"]

    results = run_step(STEP, Run(config, prompts_dir, {LAYER4: output_dir}), rewrites).records_in_input_order
    rewrites_by_id = {rw["doc_id"]: rw for rw in rewrites}

    passed = keep_documents_at_or_above_threshold(results, threshold)
    failed = len(results) - len(passed)
    if failed:
        print(f"  {failed} document(s) below threshold {threshold} — dropped")

    passed_with_content = [{**r, "content": rewrites_by_id[r["doc_id"]]["content"]} for r in passed]
    passed, dropped = remove_near_duplicates(passed_with_content, sdf.get("near_dup_threshold"))
    if dropped:
        utils.save_jsonl(dropped, output_dir / "near_dup_dropped.jsonl")
        print(f"  {len(dropped)} near-duplicate(s) culled")

    # Final corpus records carry the eval-facing fields (audit_sdf/diversity
    # read content/type/language/register) alongside the full matrix lineage.
    corpus = []
    for r in passed:
        rw = rewrites_by_id[r["doc_id"]]
        v = rw["variables"]
        corpus.append({
            "doc_id": r["doc_id"],
            "type_id": v.get("document_type", ""),
            "type_name": v.get("document_type", ""),
            "language": cp.derive_language(v.get("culture", "")),
            "register": v.get("tone", ""),
            "variables": v,
            "description": rw["description"],
            "considerations_spec": rw.get("considerations_spec", ""),
            "content": rw["content"],
            "scores": r["scores"],
        })
    utils.save_jsonl(corpus, final_path)

    low_spec = [r for r in results if r["scores"]["spec_conformance"] < threshold]
    if low_spec:
        print(f"  Note: {len(low_spec)} document(s) scored <{threshold} on spec_conformance (advisory, not gated)")
    return corpus
