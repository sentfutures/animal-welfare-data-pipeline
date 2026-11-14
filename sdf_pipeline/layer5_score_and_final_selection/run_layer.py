"""Layer 5 (score and final selection): score each rewrite, gate, remove near-duplicates and write the final corpus.

The judge scores alignment, realism, and spec_conformance (1-10 each). The gate keeps documents with alignment AND
realism >= sdf.min_score_threshold; spec_conformance is recorded and reported but does not gate. Survivors pass a
deterministic near-duplicate removal before the final corpus is written.

The layer is three steps, one module each in ``sdf_pipeline/layer5_score_and_final_selection/``: ``ScoreStep`` (``score.py``, a
``DocumentStep`` run by ``sdf_pipeline.shared_step_machinery.step_runner.StepRunner``), ``ScoreGateStep`` (``score_gate.py``) and
``NearDuplicateRemovalStep`` (``near_duplicate_removal.py``), both ``CorpusStep``s run after it by
``run_corpus_step``. This module holds the layer's driver, ``run``, called by ``sdf_pipeline/run.py`` and
``experiments/experiment.py``: it runs the three steps and writes the final corpus.

Document order before the removal is the order of ``results`` in ``run``: scores finished in earlier runs (in
``rewrites`` order) followed by the scores written in this run (in ``rewrites`` order). An uninterrupted run is
therefore in ``rewrites`` order; a resumed run puts a document scored late after the ones scored before the
interruption. The removal keeps the first of each near-duplicate pair in that order.
"""

from pathlib import Path

from shared import utils
from sdf_pipeline.shared_step_machinery import compose_prompts
from sdf_pipeline.shared_step_machinery.step_runner import StepRunner, run_corpus_step
from sdf_pipeline.layer5_score_and_final_selection.near_duplicate_removal import NearDuplicateRemovalStep
from sdf_pipeline.layer5_score_and_final_selection.score import *  # noqa: F401,F403
from sdf_pipeline.layer5_score_and_final_selection.score import PARSE_ERROR_SCORES, ScoreStep  # noqa: F401
from sdf_pipeline.layer5_score_and_final_selection.score_gate import ScoreGateStep
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Removed


def final_corpus_record(
        document_id: str, variables: dict, description: str, considerations_spec: str | None, content: str,
        scores: dict, variable_value_ids: dict | None = None) -> dict:
    """A record of ``final/sdf_corpus.jsonl``, with its field order. ``considerations_spec`` None leaves the
    field out (the matrix layout before the renumbering). ``variable_value_ids`` (the dealt value ids of the
    stage files) is written right after ``variables`` when given and left out when None."""
    record = {
        "doc_id": document_id,
        "type_id": variables.get("document_type", ""),
        "type_name": variables.get("document_type", ""),
        "language": compose_prompts.derive_language(variables.get("culture", "")),
        "register": variables.get("tone", ""),
        "variables": variables,
    }
    if variable_value_ids is not None:
        record["variable_value_ids"] = variable_value_ids
    record["description"] = description
    if considerations_spec is not None:
        record["considerations_spec"] = considerations_spec
    record["content"] = content
    record["scores"] = scores
    return record


def run(
    config: dict,
    prompts_dir: Path,
    output_dir: Path,
    final_dir: Path,
    rewrites: list[dict],
) -> list[dict]:
    final_path = final_dir / "sdf_corpus.jsonl"
    sdf = config["sdf"]
    threshold = sdf["min_score_threshold"]

    step = ScoreStep(config, prompts_dir)
    runner = StepRunner(step, output_dir, workers=config.get("workers", 1))
    results = runner.run(rewrites).records_in_input_order
    rewrites_by_id = {rw["doc_id"]: rw for rw in rewrites}

    # Gate on alignment + realism; spec_conformance is advisory (reported below).
    gate_outcomes = run_corpus_step(ScoreGateStep(threshold), results)
    passed = [r for r in results if isinstance(gate_outcomes[r["doc_id"]], Accepted)]
    failed = len(results) - len(passed)
    if failed:
        print(f"  {failed} document(s) below threshold {threshold} — dropped")

    # Deterministic near-duplicate removal over survivors (greedy keep-first).
    passed_with_content = [{**r, "content": rewrites_by_id[r["doc_id"]]["content"]} for r in passed]
    removal_outcomes = run_corpus_step(
        NearDuplicateRemovalStep(sdf.get("near_dup_threshold")), passed_with_content)
    dropped = [
        {"doc_id": r["doc_id"], **removal_outcomes[r["doc_id"]].detail}
        for r in passed if isinstance(removal_outcomes[r["doc_id"]], Removed)]
    if dropped:
        utils.save_jsonl(dropped, output_dir / "near_dup_dropped.jsonl")
        print(f"  {len(dropped)} near-duplicate(s) culled")
    passed = [r for r in passed if isinstance(removal_outcomes[r["doc_id"]], Accepted)]

    # Final corpus records carry the eval-facing fields (audit_sdf/diversity
    # read content/type/language/register) alongside the full matrix lineage.
    corpus = []
    for r in passed:
        rw = rewrites_by_id[r["doc_id"]]
        corpus.append(final_corpus_record(
            r["doc_id"], rw["variables"], rw["description"], rw.get("considerations_spec", ""),
            rw["content"], r["scores"], rw.get("variable_value_ids")))
    utils.save_jsonl(corpus, final_path)

    low_spec = [r for r in results if r["scores"]["spec_conformance"] < threshold]
    if low_spec:
        print(f"  Note: {len(low_spec)} document(s) scored <{threshold} on spec_conformance (advisory, not gated)")
    return corpus
