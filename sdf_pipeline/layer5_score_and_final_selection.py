"""Layer 5 (score and final selection): score each rewrite, gate, remove near-duplicates and write the final corpus.

The judge scores alignment, realism, and spec_conformance (1-10 each). The gate keeps documents with alignment AND
realism >= sdf.min_score_threshold; spec_conformance is recorded and reported but does not gate. Survivors pass a
deterministic near-duplicate removal before the final corpus is written.

The layer is three steps, one module each in ``sdf_pipeline/steps/``: ``ScoreStep`` (``score.py``, a
``DocumentStep`` run by ``sdf_pipeline.step_runner.StepRunner``), ``ScoreGateStep`` (``score_gate.py``) and
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
from sdf_pipeline.document_state_final_corpus import final_corpus_record
from sdf_pipeline.step_runner import StepRunner, run_corpus_step
from sdf_pipeline.steps.near_duplicate_removal import NearDuplicateRemovalStep
from sdf_pipeline.steps.score import *  # noqa: F401,F403
from sdf_pipeline.steps.score import PARSE_ERROR_SCORES, ScoreStep  # noqa: F401
from sdf_pipeline.steps.score_gate import ScoreGateStep
from sdf_pipeline.update_steps import Accepted, Removed


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
