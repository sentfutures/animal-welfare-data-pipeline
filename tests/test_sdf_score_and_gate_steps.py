"""Score as a DocumentStep, gate and near-duplicate cull as CorpusSteps.

Step unit tests call ``interpret`` and ``apply`` with no stubbed API and no files. The layer tests
pin the request, the document order before the cull and resume.
``layer4_score.run`` keeps its signature and behaviour (scripts import it).
"""

import json

import pytest

from sdf_pipeline import layer4_score
from sdf_pipeline.layer4_score import NearDuplicateCullStep, ScoreGateStep, ScoreStep
from sdf_pipeline.step_runner import run_corpus_step
from sdf_pipeline.update_steps import Accepted, FailureKind, Failed, ModelReply, Removed, StepProgress
from shared import utils
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
def score_step(tiny_config, prompts_sdf):
    return ScoreStep(tiny_config, prompts_sdf)


# ---------------------------------------------------------------- ScoreStep.interpret

def test_interpret_accepts_a_parsed_answer_with_missing_fields_defaulted(score_step):
    rewrite = make_rewrite()
    outcome = score_step.interpret(rewrite, ModelReply(scores_reply(9, 8, 7, "fine"), "end_turn"))
    assert outcome == Accepted({"alignment": 9, "realism": 8, "spec_conformance": 7, "notes": "fine"})
    outcome = score_step.interpret(rewrite, ModelReply('{"alignment": 6}', "end_turn"))
    assert outcome == Accepted({"alignment": 6, "realism": 0, "spec_conformance": 0, "notes": ""})
    assert score_step.stage_file_record(rewrite, outcome.update) == {
        "doc_id": "matrix_000000", "variables": rewrite["variables"], "scores": outcome.update}


@pytest.mark.parametrize("text", ["utter nonsense, no json", "[8, 7, 9]", ""])
def test_interpret_accepts_an_unparseable_answer_as_five_five_five(score_step, text):
    outcome = score_step.interpret(make_rewrite(), ModelReply(text, "end_turn"))
    assert outcome == Accepted({"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."})


def test_a_truncated_but_parseable_answer_is_accepted_as_today(score_step):
    # the layer never looked at the stop reason
    assert isinstance(score_step.interpret(make_rewrite(), ModelReply(scores_reply(9, 9), "max_tokens")), Accepted)


def test_a_thrown_call_is_left_behind_and_prints_todays_line(score_step):
    failed = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    assert score_step.progress_line(make_rewrite(), failed, StepProgress()) == (
        "  matrix_000000: API call failed (RuntimeError: boom) — will retry on resume")
    assert score_step.progress_line(make_rewrite(), Accepted(
        {"alignment": 9, "realism": 8, "spec_conformance": 7, "notes": ""}), StepProgress()) == "  Scored matrix_000000 A9 R8 S7"
    assert FailureKind.THROWN_CALL in score_step.policy.failure_kinds_counting_towards_the_systemic_stop
    assert FailureKind.UNUSABLE_ANSWER not in score_step.policy.failure_kinds_counting_towards_the_systemic_stop


def test_request_sent_has_exactly_the_arguments_the_layer_always_passed(
        tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    calls = stub_claude(lambda user_message, **kw: SCORE_OK)
    layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
    (call,) = calls
    assert call["max_tokens"] is None and call["stage"] == "layer4_score" and call["item_id"] == "matrix_000000"
    assert call["cache_system"] is True and call["temperature"] is None and call["json_schema"] is None
    assert call["model"] == tiny_config["sdf"].get("score_model")
    assert call["system_prompt"] and call["user_message"]


# ---------------------------------------------------------------- gate and cull as corpus steps

def score_record(doc_id, alignment, realism, content="x"):
    return {"doc_id": doc_id, "scores": {"alignment": alignment, "realism": realism, "spec_conformance": 1}, "content": content}


def test_gate_boundaries_at_the_threshold():
    records = [score_record("at_both", 7, 7), score_record("alignment_below", 6, 10),
               score_record("realism_below", 10, 6), score_record("above", 8, 9)]
    outcomes = ScoreGateStep(7).apply(records)
    assert {d: isinstance(o, Accepted) for d, o in outcomes.items()} == {
        "at_both": True, "alignment_below": False, "realism_below": False, "above": True}
    assert outcomes["alignment_below"].reason_code == "below_score_threshold"


def test_gate_ignores_spec_conformance():
    assert isinstance(ScoreGateStep(7).apply([score_record("a", 9, 9)])["a"], Accepted)


def test_cull_keeps_the_first_of_a_near_duplicate_pair_in_the_order_given():
    records = [score_record("second_in_deal", 9, 9, LONG_TEXT), score_record("other", 9, 9, "Tout autre chose, un texte sans rapport."),
               score_record("first_in_deal", 9, 9, LONG_TEXT)]
    outcomes = NearDuplicateCullStep(0.90).apply(records)
    assert isinstance(outcomes["second_in_deal"], Accepted) and isinstance(outcomes["other"], Accepted)
    assert outcomes["first_in_deal"].reason_code == "near_duplicate"
    assert outcomes["first_in_deal"].detail["kept_doc_id"] == "second_in_deal"
    # the order given decides: reversed, the other copy is kept
    reversed_outcomes = NearDuplicateCullStep(0.90).apply(list(reversed(records)))
    assert reversed_outcomes["second_in_deal"].detail["kept_doc_id"] == "first_in_deal"


def test_cull_does_nothing_without_a_threshold_or_with_one_document():
    twins = [score_record("a", 9, 9, LONG_TEXT), score_record("b", 9, 9, LONG_TEXT)]
    assert all(isinstance(o, Accepted) for o in NearDuplicateCullStep(None).apply(twins).values())
    assert all(isinstance(o, Accepted) for o in NearDuplicateCullStep(0).apply(twins).values())
    assert all(isinstance(o, Accepted) for o in NearDuplicateCullStep(0.9).apply(twins[:1]).values())


def test_run_corpus_step_returns_the_outcomes_of_the_step_each_time_it_runs(tmp_path):
    records = [score_record("keep", 9, 9), score_record("drop", 5, 9)]
    outcomes = run_corpus_step(ScoreGateStep(7), records)
    assert isinstance(outcomes["keep"], Accepted) and outcomes["drop"].reason_code == "below_score_threshold"
    assert run_corpus_step(ScoreGateStep(7), records) == outcomes   # run again, nothing changes
    assert isinstance(run_corpus_step(ScoreGateStep(5), records)["drop"], Accepted)  # a lower threshold lets it through


def test_run_corpus_step_refuses_a_step_that_returns_a_failure():
    class FailingStep(ScoreGateStep):
        def apply(self, states):
            return {"x": Failed(FailureKind.THROWN_CALL, "no")}
    with pytest.raises(TypeError, match="Failed"):
        run_corpus_step(FailingStep(7), [score_record("x", 9, 9)])


# ---------------------------------------------------------------- the layer: order, resume

def test_document_order_before_the_cull_is_input_order_uninterrupted_and_scored_first_when_resumed(
        tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    ids = ["matrix_000000", "matrix_000001", "matrix_000002"]
    rewrites = [make_rewrite(i, LONG_TEXT) for i in ids]

    stub_claude(lambda user_message, **kw: SCORE_OK)
    corpus = layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
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
    layer4_score.run(config, prompts_sdf, stage_dir_2, final_dir_2, rewrites)
    stub_claude(lambda user_message, **kw: SCORE_OK)
    corpus = layer4_score.run(config, prompts_sdf, stage_dir_2, final_dir_2, rewrites)
    assert [r["doc_id"] for r in utils.load_jsonl(stage_dir_2 / "scores.jsonl")] == [
        "matrix_000001", "matrix_000002", "matrix_000000"]
    assert [r["doc_id"] for r in corpus] == ["matrix_000001"]  # the resumed run keeps a different twin (risk 11)


def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    rewrites = [make_rewrite("matrix_000000", LONG_TEXT), make_rewrite("matrix_000001", LONG_TEXT),
                make_rewrite("matrix_000002", "Un autre texte tout à fait différent des deux premiers.")]
    stub_claude(lambda user_message, **kw: SCORE_OK)
    layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)

    def snapshot():
        return {str(p.relative_to(stage_dir.parent)): p.read_bytes() for p in stage_dir.parent.rglob("*") if p.is_file()}

    before = snapshot()
    calls = stub_claude([])
    corpus = layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert calls == [] and [r["doc_id"] for r in corpus] == ["matrix_000000", "matrix_000002"]
    assert snapshot() == before  # no stage file changes


def test_resume_retries_exactly_the_failed_documents(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    ids = [f"matrix_00000{i}" for i in range(4)]
    rewrites = [make_rewrite(i, f"Texte numéro {i}, distinct des autres.") for i in ids]

    def first_attempt(user_message, **kw):
        if kw["item_id"] in ("matrix_000001", "matrix_000003"):
            raise RuntimeError("thrown")
        return SCORE_OK

    stub_claude(first_attempt)
    corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert [r["doc_id"] for r in corpus] == ["matrix_000000", "matrix_000002"]  # the gate does not wait for them
    checkpoint = utils.Checkpoint(stage_dir / "_checkpoint.json")
    assert not checkpoint.is_done("matrix_000001") and not checkpoint.is_done("matrix_000003")

    calls = stub_claude(lambda user_message, **kw: SCORE_OK)
    corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert sorted(c["item_id"] for c in calls) == ["matrix_000001", "matrix_000003"]
    assert [r["doc_id"] for r in corpus] == ["matrix_000000", "matrix_000002", "matrix_000001", "matrix_000003"]


def test_every_call_failing_stops_the_run_with_todays_message(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    def always_throws(user_message, **kw):
        raise RuntimeError("down")

    stub_claude(always_throws)
    with pytest.raises(SystemExit, match="layer4_score: every pending API call failed"):
        layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
    assert not (final_dir / "sdf_corpus.jsonl").exists()


def test_printed_lines_are_todays(tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude, capsys):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    stub_claude(lambda user_message, **kw: scores_reply(4, 9, 3) if kw["item_id"] == "matrix_000002" else scores_reply(9, 8, 3))
    rewrites = [make_rewrite("matrix_000000", LONG_TEXT), make_rewrite("matrix_000001", LONG_TEXT),
                make_rewrite("matrix_000002", "Texte à part.")]
    layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    assert capsys.readouterr().out.splitlines() == [
        "  Scored matrix_000000 A9 R8 S3", "  Scored matrix_000001 A9 R8 S3", "  Scored matrix_000002 A4 R9 S3",
        "  1 document(s) below threshold 7 — dropped", "  1 near-duplicate(s) culled",
        "  Note: 3 document(s) scored <7 on spec_conformance (advisory, not gated)"]


def test_scores_hold_every_document_and_the_corpus_only_the_ones_through_the_gate_and_the_cull(
        tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
    stub_claude(lambda user_message, **kw: scores_reply(4, 9) if kw["item_id"] == "matrix_000002" else SCORE_OK)
    rewrites = [make_rewrite("matrix_000000", LONG_TEXT), make_rewrite("matrix_000001", LONG_TEXT),
                make_rewrite("matrix_000002", "Texte à part.")]
    corpus = layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
    scored = {r["doc_id"]: r for r in utils.load_jsonl(stage_dir / "scores.jsonl")}
    assert set(scored) == {"matrix_000000", "matrix_000001", "matrix_000002"}
    assert scored["matrix_000002"]["scores"]["alignment"] == 4   # below the gate
    # matrix_000002 fails the gate and matrix_000001 is the later copy of matrix_000000
    assert [r["doc_id"] for r in corpus] == ["matrix_000000"]
    assert [r["doc_id"] for r in utils.load_jsonl(final_dir / "sdf_corpus.jsonl")] == ["matrix_000000"]
