"""A prompt-building error is a recorded failure for that document in the draft and layer 3 steps (the
exception policy is declared once, in ``StepPolicy``): the other documents are still sent, the failure
is printed and retried on resume, and a template broken for every document ends the run through
the systemic stop with no model call made."""

import pytest

from sdf_pipeline.layer4_draft_critique_and_revise import draft as draft_step
from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from sdf_plan_draft_rewrite_and_score_test_builders import (
    make_draft, make_plan, single_call_config, split_config)

BROKEN_DOCUMENT_ID = "matrix_000001"


def break_template(monkeypatch, step_class, only_document_id=None):
    real_build_request = step_class.build_request

    def build_request(self, document):
        if only_document_id is None or self.document_id_of(document) == only_document_id:
            raise KeyError("missing_placeholder")
        return real_build_request(self, document)

    monkeypatch.setattr(step_class, "build_request", build_request)
    return real_build_request


def test_draft_step_records_the_failure_prints_it_and_a_resume_retries_it(
        tiny_config, prompts_sdf, stage_dir, stub_claude, monkeypatch, capsys):
    plans = [make_plan("matrix_000000"), make_plan(BROKEN_DOCUMENT_ID)]
    real_build_request = break_template(monkeypatch, draft_step.DraftStep, BROKEN_DOCUMENT_ID)
    calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
    records = draft_step.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert [r["doc_id"] for r in records] == ["matrix_000000"] and [c["item_id"] for c in calls] == ["matrix_000000"]
    printed = capsys.readouterr().out
    assert BROKEN_DOCUMENT_ID in printed and "error: KeyError: 'missing_placeholder'" in printed

    monkeypatch.setattr(draft_step.DraftStep, "build_request", real_build_request)
    calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
    records = draft_step.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert [c["item_id"] for c in calls] == [BROKEN_DOCUMENT_ID]
    assert sorted(r["doc_id"] for r in records) == ["matrix_000000", BROKEN_DOCUMENT_ID]


def test_draft_step_with_a_template_broken_for_every_document_stops_with_no_model_call(
        tiny_config, prompts_sdf, stage_dir, stub_claude, monkeypatch):
    break_template(monkeypatch, draft_step.DraftStep)
    calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
    with pytest.raises(SystemExit) as stop:
        draft_step.run(tiny_config, prompts_sdf, stage_dir, [make_plan("matrix_000000"), make_plan(BROKEN_DOCUMENT_ID)])
    assert str(stop.value) == (
        "layer4_draft: every pending API call failed — this is systemic "
        "(auth, backend, network, or an error building the prompt), not per-document; fix and --resume.")
    assert calls == []


@pytest.mark.parametrize("broken_step_name", ["CritiqueStep", "CritiqueAndReviseInOneCallStep"])
def test_layer_3_with_a_template_broken_for_every_document_stops_with_no_model_call(
        tiny_config, prompts_sdf, stage_dir, stub_claude, monkeypatch, broken_step_name):
    config = split_config(tiny_config) if broken_step_name == "CritiqueStep" else single_call_config(tiny_config)
    break_template(monkeypatch, getattr(draft_critique_and_revise_layer, broken_step_name))
    calls = stub_claude(lambda user_message, **kw: "<review>r</review>")
    with pytest.raises(SystemExit) as stop:
        draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [make_draft(), make_draft("matrix_000001")])
    assert "an error building the prompt" in str(stop.value)
    assert calls == []


def test_layer_3_revise_prompt_building_error_is_a_failure_that_a_resume_retries(
        tiny_config, prompts_sdf, stage_dir, stub_claude, monkeypatch):
    drafts = [make_draft(), make_draft("matrix_000001")]

    def replies(user_message, **kw):
        if kw.get("stage") == "layer4_critique":
            return "<review>Problem.</review>"
        return "<improved_document>Better.</improved_document>"

    real_build_request = break_template(monkeypatch, draft_critique_and_revise_layer.ReviseStep, "matrix_000001")
    stub_claude(replies)
    records = draft_critique_and_revise_layer.run_critique_and_revise(split_config(tiny_config), prompts_sdf, stage_dir, drafts)
    assert [r["doc_id"] for r in records] == ["matrix_000000"]
    monkeypatch.setattr(draft_critique_and_revise_layer.ReviseStep, "build_request", real_build_request)
    calls = stub_claude(replies)
    records = draft_critique_and_revise_layer.run_critique_and_revise(split_config(tiny_config), prompts_sdf, stage_dir, drafts)
    assert [(c["stage"], c["item_id"]) for c in calls] == [("layer4_revise", "matrix_000001")]
    assert sorted(r["doc_id"] for r in records) == ["matrix_000000", "matrix_000001"]


def test_the_final_corpus_record_has_one_definition_with_its_field_order_and_the_matrix_layout_omission():
    from sdf_pipeline.shared_step_machinery.document_state_final_corpus import final_corpus_record
    variables = {"document_type": "memo", "culture": "French", "tone": "dry"}
    record = final_corpus_record("matrix_000000", variables, "Spec.", "Consider.", "Text.", {"alignment": 5})
    assert list(record) == ["doc_id", "type_id", "type_name", "language", "register", "variables", "description",
                            "considerations_spec", "content", "scores"]
    assert (record["type_id"], record["type_name"], record["register"], record["considerations_spec"]) == (
        "memo", "memo", "dry", "Consider.")
    assert "considerations_spec" not in final_corpus_record("m", variables, "Spec.", None, "Text.", {})
