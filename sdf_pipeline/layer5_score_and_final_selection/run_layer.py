"""Layer 5: the driver ``run`` scores each rewrite (``score.py``), gates (``score_gate.py``), removes near-duplicates (``near_duplicate_removal.py``) and writes the final corpus.
Documents reach the removal in scored order (earlier runs' scores in ``rewrites`` order, then this run's), and the removal keeps the first of each pair."""

from pathlib import Path

from shared import utils
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.step_record import Run
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.layer5_score_and_final_selection.near_duplicate_removal import remove_near_duplicates
from sdf_pipeline.layer5_score_and_final_selection import score
from sdf_pipeline.layer5_score_and_final_selection.score import *  # noqa: F401,F403
from sdf_pipeline.layer5_score_and_final_selection.score import PARSE_ERROR_SCORES  # noqa: F401
from sdf_pipeline.layer5_score_and_final_selection.score_gate import keep_documents_at_or_above_threshold
from shared.sdf_run_layouts import SCORE


def run(
    config: dict,
    prompts_dir: Path,
    output_dir: Path,
    final_dir: Path,
    rewrites: list[dict],
) -> list[dict]:
    """Score, gate and remove near-duplicates from the rewrites and write the final corpus; returns its records."""
    final_path = final_dir / "sdf_corpus.jsonl"
    sdf = config["sdf"]
    threshold = sdf["min_score_threshold"]

    results = run_step(score.STEP, Run(config, prompts_dir, {SCORE: output_dir}), rewrites).records_in_input_order
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
