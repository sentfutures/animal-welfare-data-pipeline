"""Score as a step, gate and near-duplicate cull as plain functions.

Unit tests call ``read_reply``, the gate and the cull with no stubbed API and no files. The layer tests
pin the request, the document order before the cull and resume.
``score_and_final_selection_layer.run`` is the layer's entry point (scripts import it).
"""

import json

import pytest

from sdf_pipeline.layer5_score_and_final_selection import run_layer as score_and_final_selection_layer
from sdf_pipeline.layer5_score_and_final_selection import score
from sdf_pipeline.layer5_score_and_final_selection.near_duplicate_removal import remove_near_duplicates
from sdf_pipeline.layer5_score_and_final_selection.score_gate import keep_documents_at_or_above_threshold
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, FailureKind, Failed, ModelReply, StepProgress
from sdf_step_and_command_line_test_helpers import outcome_of, records_of, run_of
from shared import utils
from shared.sdf_run_layouts import SCORE
from sdf_plan_draft_rewrite_and_score_test_builders import SCORE_OK, make_rewrite

LONG_TEXT = ("Le même document mot pour mot, assez long pour produire des shingles "
             "stables et un cosinus de similarité fiable entre les deux copies.")


def scores_reply(alignment, realism, spec_conformance=8, notes="n"):
    return json.dumps({"alignment": alignment, "realism": realism,
                       "spec_conformance": spec_conformance, "notes": notes})


@pytest.fixture
def final_dir(tmp_path):
    directory = tmp_path / "final"
    directory.mkdir()
    return directory


@pytest.fixture
def score_run(tiny_config, prompts_sdf, stage_dir):
    return run_of(tiny_config, prompts_sdf, SCORE, stage_dir)


# ---------------------------------------------------------------- reading the score reply

def test_read_accepts_a_parsed_answer_with_missing_fields_defaulted(score_run):
    rewrite = make_rewrite()
    outcome = outcome_of(score.STEP, rewrite, ModelReply(scores_reply(9, 8, 7, "fine"), "end_turn"), score_run)
    assert outcome == Accepted({"alignment": 9, "realism": 8, "spec_conformance": 7, "notes": "fine"})
    outcome = outcome_of(score.STEP, rewrite, ModelReply('{"alignment": 6}', "end_turn"), score_run)
    assert outcome == Accepted({"alignment": 6, "realism": 0, "spec_conformance": 0, "notes": ""})
    ((file_name, record),) = records_of(score.STEP, rewrite, outcome, score_run).records
    assert file_name == "scores.jsonl" and record == {
        "doc_id": "matrix_000000", "variables": rewrite["variables"], "scores": outcome.update}


@pytest.mark.parametrize("text", ["utter nonsense, no json", "[8, 7, 9]", ""])
def test_read_accepts_an_unparseable_answer_as_five_five_five(score_run, text):
    outcome = outcome_of(score.STEP, make_rewrite(), ModelReply(text, "end_turn"), score_run)
    assert outcome == Accepted({"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."})


def test_a_thrown_call_is_left_behind_and_prints_todays_line(score_run):
    failed = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    rewrite = make_rewrite()
    assert score.STEP.progress_line([rewrite], {"matrix_000000": failed}, StepProgress(), score_run) == (
        "  matrix_000000: API call failed (RuntimeError: boom) — will retry on resume")
    assert score.STEP.progress_line([rewrite], {"matrix_000000": Accepted(
        {"alignment": 9, "realism": 8, "spec_conformance": 7, "notes": ""})}, StepProgress(), score_run) == "  Scored matrix_000000 A9 R8 S7"
    assert FailureKind.THROWN_CALL in score.STEP.policy.failure_kinds_counting_towards_the_systemic_stop
    assert FailureKind.UNUSABLE_ANSWER not in score.STEP.policy.failure_kinds_counting_towards_the_systemic_stop


# ---------------------------------------------------------------- gate and cull as plain functions

def score_record(doc_id, alignment, realism, content="x"):
    return {"doc_id": doc_id, "scores": {"alignment": alignment, "realism": realism, "spec_conformance": 1}, "content": content}


def test_gate_boundaries_at_the_threshold():
    records = [score_record("at_both", 7, 7), score_record("alignment_below", 6, 10),
               score_record("realism_below", 10, 6), score_record("above", 8, 9)]
    kept = keep_documents_at_or_above_threshold(records, 7)
    assert [record["doc_id"] for record in kept] == ["at_both", "above"]


def test_gate_ignores_spec_conformance():
    assert keep_documents_at_or_above_threshold([score_record("a", 9, 9)], 7) != []   # spec_conformance is 1


def test_gate_runs_again_with_any_threshold():
    records = [score_record("keep", 9, 9), score_record("drop", 5, 9)]
    assert [r["doc_id"] for r in keep_documents_at_or_above_threshold(records, 7)] == ["keep"]
    assert [r["doc_id"] for r in keep_documents_at_or_above_threshold(records, 5)] == ["keep", "drop"]


def test_cull_keeps_the_first_of_a_near_duplicate_pair_in_the_order_given():
    records = [score_record("second_in_deal", 9, 9, LONG_TEXT), score_record("other", 9, 9, "Tout autre chose, un texte sans rapport."),
               score_record("first_in_deal", 9, 9, LONG_TEXT)]
    kept, dropped = remove_near_duplicates(records, 0.90)
    assert [r["doc_id"] for r in kept] == ["second_in_deal", "other"]
    ((dropped_record),) = dropped
    assert dropped_record["doc_id"] == "first_in_deal" and dropped_record["kept_doc_id"] == "second_in_deal"
    assert list(dropped_record) == ["doc_id", "kept_doc_id", "similarity"]
    # the order given decides: reversed, the other copy is kept
    _, reversed_dropped = remove_near_duplicates(list(reversed(records)), 0.90)
    assert reversed_dropped[0]["doc_id"] == "second_in_deal" and reversed_dropped[0]["kept_doc_id"] == "first_in_deal"


def test_cull_does_nothing_without_a_threshold_or_with_one_document():
    twins = [score_record("a", 9, 9, LONG_TEXT), score_record("b", 9, 9, LONG_TEXT)]
    assert remove_near_duplicates(twins, None) == (twins, [])
    assert remove_near_duplicates(twins, 0) == (twins, [])
    assert remove_near_duplicates(twins[:1], 0.9) == (twins[:1], [])


# ---------------------------------------------------------------- the layer: order, resume

def test_document_order_before_the_cull_is_input_order_uninterrupted_and_scored_first_when_resumed(
        tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    ids = ["matrix_000000", "matrix_000001", "matrix_000002"]
    rewrites = [make_rewrite(i, LONG_TEXT) for i in ids]

    stub_claude(lambda user_message, **kw: SCORE_OK)
    corpus = score_and_final_selection_layer.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert [r["doc_id"] for r in corpus] == ["matrix_000000"]  # uninterrupted: rewrites order, first kept

    # resumed: the first document's call throws, so the later ones are scored before it
    stage_dir_2 = stage_dir.parent / "resumed" / stage_dir.name
    final_dir_2 = stage_dir.parent / "resumed" / "final"
    utils.ensure_dir(stage_dir_2), utils.ensure_dir(final_dir_2)

    def first_attempt(user_message, **kw):
        if kw["item_id"] == "matrix_000000":
            raise RuntimeError("thrown")
        return SCORE_OK

    stub_claude(first_attempt)
    score_and_final_selection_layer.run(config, prompts_sdf, stage_dir_2, final_dir_2, rewrites)
    stub_claude(lambda user_message, **kw: SCORE_OK)
    corpus = score_and_final_selection_layer.run(config, prompts_sdf, stage_dir_2, final_dir_2, rewrites)
    assert [r["doc_id"] for r in utils.load_jsonl(stage_dir_2 / "scores.jsonl")] == [
        "matrix_000001", "matrix_000002", "matrix_000000"]
    assert [r["doc_id"] for r in corpus] == ["matrix_000001"]  # the resumed run keeps a different twin (risk 11)


def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    rewrites = [make_rewrite("matrix_000000", LONG_TEXT), make_rewrite("matrix_000001", LONG_TEXT),
                make_rewrite("matrix_000002", "Un autre texte tout à fait différent des deux premiers.")]
    stub_claude(lambda user_message, **kw: SCORE_OK)
    score_and_final_selection_layer.run(config, prompts_sdf, stage_dir, final_dir, rewrites)

    def snapshot():
        return {str(p.relative_to(stage_dir.parent)): p.read_bytes() for p in stage_dir.parent.rglob("*") if p.is_file()}

    before = snapshot()
    calls = stub_claude([])
    corpus = score_and_final_selection_layer.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert calls == [] and [r["doc_id"] for r in corpus] == ["matrix_000000", "matrix_000002"]
    assert snapshot() == before  # no stage file changes


def test_printed_lines_are_todays(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude, capsys):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    stub_claude(lambda user_message, **kw: scores_reply(4, 9, 3) if kw["item_id"] == "matrix_000002" else scores_reply(9, 8, 3))
    rewrites = [make_rewrite("matrix_000000", LONG_TEXT), make_rewrite("matrix_000001", LONG_TEXT),
                make_rewrite("matrix_000002", "Texte à part.")]
    score_and_final_selection_layer.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert capsys.readouterr().out.splitlines() == [
        "  Scored matrix_000000 A9 R8 S3", "  Scored matrix_000001 A9 R8 S3", "  Scored matrix_000002 A4 R9 S3",
        "  1 document(s) below threshold 7 — dropped", "  1 near-duplicate(s) culled",
        "  Note: 3 document(s) scored <7 on spec_conformance (advisory, not gated)"]


