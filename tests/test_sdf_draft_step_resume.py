"""The draft layer as a DocumentStep: the request sent and resume behaviour.

``layer2_draft.run`` keeps its signature and behaviour (scripts import it); these tests pin the
request it sends and that a resume pays for exactly the unfinished work.
"""

import pytest

from sdf_pipeline import layer2_draft
from shared import utils
from sdf_plan_draft_rewrite_and_score_test_builders import make_plan


def plans_for(*ids):
    return [make_plan(i) for i in ids]


def test_request_sent_has_exactly_the_arguments_the_layer_always_passed(tiny_config, prompts_sdf, stage_dir, stub_claude):
    calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
    layer2_draft.run(tiny_config, prompts_sdf, stage_dir, [make_plan()])
    (call,) = calls
    assert call["max_tokens"] == 6000 and call["stage"] == "layer2_draft" and call["item_id"] == "matrix_000000"
    assert call["cache_system"] is True and call["temperature"] is None and call["json_schema"] is None
    assert call["model"] == tiny_config["sdf"].get("draft_model")
    assert call["injection"] == "" and call["system_prompt"] and call["user_message"]


def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(lambda user_message, **kw: "<document>D.</document>")
    plans = plans_for("matrix_000000", "matrix_000001")
    layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert calls == [] and len(records) == 2
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before


def test_resume_retries_exactly_the_failed_documents(tiny_config, prompts_sdf, stage_dir, stub_claude):
    def first_attempt(user_message, **kw):
        item = kw["item_id"]
        if item == "matrix_000001":
            raise RuntimeError("thrown")
        if item == "matrix_000002":
            return "<document>cut", "max_tokens"
        if item == "matrix_000003":
            return "no tags", "end_turn"
        return f"<document>Doc {item}.</document>"

    # call_claude returns (text, stop) when asked; the stub wraps plain strings as end_turn
    stub_claude(first_attempt)
    plans = plans_for(*(f"matrix_00000{i}" for i in range(5)))
    records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000004"]
    checkpoint = utils.Checkpoint(stage_dir / "_checkpoint.json")
    assert not any(checkpoint.is_done(f"matrix_00000{i}") for i in (1, 2, 3))

    calls = stub_claude(lambda user_message, **kw: f"<document>Recovered {kw['item_id']}.</document>")
    records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert sorted(c["item_id"] for c in calls) == ["matrix_000001", "matrix_000002", "matrix_000003"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_dir / "drafts.jsonl")] == [
        "matrix_000000", "matrix_000004", "matrix_000001", "matrix_000002", "matrix_000003"]


def test_a_document_that_always_fails_does_not_stop_the_others_until_it_is_alone(
        tiny_config, prompts_sdf, stage_dir, stub_claude):
    def dispatch(user_message, **kw):
        if kw["item_id"] == "matrix_000001":
            raise RuntimeError("always")
        return "<document>ok</document>"

    stub_claude(dispatch)
    plans = plans_for("matrix_000000", "matrix_000001")
    assert [r["doc_id"] for r in layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)] == ["matrix_000000"]
    # a resume with new work alongside it still finishes the new work
    plans = plans_for("matrix_000000", "matrix_000001", "matrix_000002")
    records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000002"]
    # alone, every pending call failed, which is today's systemic stop
    with pytest.raises(SystemExit, match="systemic"):
        layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
