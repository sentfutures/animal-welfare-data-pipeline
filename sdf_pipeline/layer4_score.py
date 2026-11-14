"""Layer 4: score each rewrite and gate the final corpus.

The judge scores alignment, realism, and spec_conformance (1-10 each) against
the constitution and the document's generating spec. The gate keeps documents
with alignment AND realism >= sdf.min_score_threshold; spec_conformance is
recorded and reported but does not gate (drift diagnostics belong to humans
while the dimension is young). Survivors pass a deterministic near-duplicate
cull before the final corpus is written.

A scoring response that fails to parse is recorded as 5/5/5 "Parse error."
and checkpointed — a re-run would re-bill without new information; the score
report flags these for manual review instead.

Scoring is a ``DocumentStep`` (``ScoreStep``) run by ``sdf_pipeline.step_runner.StepRunner``;
the gate (``ScoreGateStep``) and the near-duplicate cull (``NearDuplicateCullStep``) are
``CorpusStep``s run after it by ``run_corpus_step``. ``run`` is the layer's
entry point, called by ``sdf_pipeline/run.py`` and ``experiments/experiment.py``: it runs the three steps
and writes the final corpus.

Document order before the cull is the order of ``results`` in ``run``: scores finished in earlier
runs (in ``rewrites`` order) followed by the scores written in this run (in ``rewrites`` order).
An uninterrupted run is therefore in ``rewrites`` order; a resumed run puts a document scored late
after the ones scored before the interruption. The cull keeps the first of each near-duplicate
pair in that order.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import constitution_loader, textstats, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline.document_state_final_corpus import final_corpus_record
from sdf_pipeline.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.step_runner import StepRunner, run_corpus_step
from sdf_pipeline.update_steps import (
    Accepted, CorpusStep, DocumentStep, Failed, ModelReply, ModelRequest, Removed, StepPolicy,
    line_for_document_left_to_retry, reason_without_error_prefix)

PARSE_ERROR_SCORES = {"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."}


class ScoreStep(DocumentStep):
    """Score one rewrite. The document is the rewrite record (``doc_id``, ``description``,
    ``variables``, ``content``). An unparseable answer is accepted as 5/5/5 "Parse error."; only a
    thrown call is left behind (and counts towards the systemic stop)."""

    step_name = "score"
    stage_file_name = "scores.jsonl"
    policy = StepPolicy()
    systemic_stop_message = (
        "layer4_score: every pending API call failed — this is systemic "
        "(auth, backend, network, or an error building the prompt), not per-document; fix and --resume."
    )

    def __init__(self, config: dict, prompts_dir: Path):
        self.score_model = config["sdf"].get("score_model")
        constitution_dir = utils.resolve_constitution_dir(prompts_dir)
        self.constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
        self.template_path = utils.sdf_template_path(prompts_dir, 4)
        self.render_variable_lines = cp.render_variable_lines_for_template(
            self.template_path.read_text(encoding="utf-8"), prompts_dir, config)

    def document_id_of(self, rewrite: dict) -> str:
        return rewrite["doc_id"]

    def applies_to(self, rewrite: dict) -> bool:
        return True

    def build_request(self, rewrite: dict) -> ModelRequest:
        system, user = cp.split_sections(utils.load_prompt(
            self.template_path,
            constitution_claude=self.constitution_claude,
            document_description=rewrite["description"],
            variable_lines=self.render_variable_lines(
                {"prompt_id": rewrite["doc_id"], "variables": rewrite["variables"]}),
            improved_document=rewrite["content"],
        ))
        return ModelRequest(
            user_message=user,
            system_prompt=system or "",
            model=self.score_model,
            stage="layer4_score",
            item_id=rewrite["doc_id"],
            cache_system=True,  # constitution + rubric are identical across scoring calls
        )

    def interpret(self, rewrite: dict, reply: ModelReply):
        try:
            scores = utils.extract_json_object(reply.text)
        except json.JSONDecodeError:
            scores = PARSE_ERROR_SCORES
        return Accepted({
            "alignment": scores.get("alignment", 0),
            "realism": scores.get("realism", 0),
            "spec_conformance": scores.get("spec_conformance", 0),
            "notes": scores.get("notes", ""),
        })

    def stage_file_record(self, rewrite: dict, update: dict) -> dict:
        return {"doc_id": rewrite["doc_id"], "variables": rewrite["variables"], **value_ids_field_of(rewrite),
                "scores": update}

    def progress_line(self, rewrite: dict, outcome) -> str | None:
        doc_id = rewrite["doc_id"]
        if isinstance(outcome, Failed):
            return line_for_document_left_to_retry(
                doc_id, f"API call failed ({reason_without_error_prefix(outcome.reason)})")
        s = outcome.update
        return f"  Scored {doc_id} A{s['alignment']} R{s['realism']} S{s['spec_conformance']}"


class ScoreGateStep(CorpusStep):
    """Keep documents with alignment AND realism >= the threshold. The documents are score records
    (``doc_id``, ``scores``); spec_conformance is advisory and does not gate."""

    step_name = "gate"

    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, score_records: list) -> dict:
        outcomes = {}
        for record in score_records:
            scores = record["scores"]
            if scores["alignment"] >= self.threshold and scores["realism"] >= self.threshold:
                outcomes[record["doc_id"]] = Accepted(None)
            else:
                outcomes[record["doc_id"]] = Removed(
                    "below_score_threshold",
                    f"alignment {scores['alignment']}, realism {scores['realism']}, threshold {self.threshold}")
        return outcomes


class NearDuplicateCullStep(CorpusStep):
    """Greedy keep-first near-duplicate cull over the documents given, in the order given. The
    documents are score records carrying ``content``. A falsy threshold, or fewer than two
    documents, culls nothing. A removal's ``detail`` is {"kept_doc_id", "similarity"}."""

    step_name = "near_duplicate_cull"

    def __init__(self, near_duplicate_threshold):
        self.near_duplicate_threshold = near_duplicate_threshold

    def apply(self, records: list) -> dict:
        outcomes = {record["doc_id"]: Accepted(None) for record in records}
        if self.near_duplicate_threshold and len(records) > 1:
            _, dropped = textstats.near_dup_filter(
                [record["content"] for record in records], self.near_duplicate_threshold)
            for entry in dropped:
                outcomes[records[entry["index"]]["doc_id"]] = Removed(
                    "near_duplicate",
                    {"kept_doc_id": records[entry["kept_index"]]["doc_id"], "similarity": entry["similarity"]})
        return outcomes


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

    # Deterministic near-duplicate cull over survivors (greedy keep-first).
    passed_with_content = [{**r, "content": rewrites_by_id[r["doc_id"]]["content"]} for r in passed]
    cull_outcomes = run_corpus_step(
        NearDuplicateCullStep(sdf.get("near_dup_threshold")), passed_with_content)
    dropped = [
        {"doc_id": r["doc_id"], **cull_outcomes[r["doc_id"]].detail}
        for r in passed if isinstance(cull_outcomes[r["doc_id"]], Removed)]
    if dropped:
        utils.save_jsonl(dropped, output_dir / "near_dup_dropped.jsonl")
        print(f"  {len(dropped)} near-duplicate(s) culled")
    passed = [r for r in passed if isinstance(cull_outcomes[r["doc_id"]], Accepted)]

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
