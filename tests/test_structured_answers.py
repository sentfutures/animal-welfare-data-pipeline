"""The shared helpers for layer-1 stages that read a schema-constrained JSON answer, and
DocumentFacts.from_plan. Offline; no model calls."""

import json

import pytest

from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as select
from sdf_pipeline.shared_step_machinery import structured_answers


class StageError(ValueError):
    pass


def test_parse_json_answer_returns_the_parsed_answer_or_the_extracted_part():
    assert structured_answers.parse_json_answer('{"a": 1}', StageError) == {"a": 1}
    assert structured_answers.parse_json_answer('{"a": 1}', StageError, extract=lambda data: data["a"]) == 1


def test_parse_json_answer_raises_the_stage_error_with_the_default_message():
    with pytest.raises(StageError, match=r"answer is not the expected JSON \(JSONDecodeError: "):
        structured_answers.parse_json_answer("not json", StageError)
    with pytest.raises(StageError, match=r"answer is not the expected JSON \(KeyError: "):
        structured_answers.parse_json_answer("{}", StageError, extract=lambda data: data["missing"])
    with pytest.raises(StageError, match=r"answer is not the expected JSON \(JSONDecodeError: "):
        structured_answers.parse_json_answer(None, StageError)


def test_parse_json_answer_takes_a_custom_message_and_exception_set():
    with pytest.raises(StageError, match=r"^no usable JSON object \(Expecting value"):
        structured_answers.parse_json_answer(
            "", StageError, caught_exceptions=(ValueError,), failure_message="no usable JSON object ({error})")
    with pytest.raises(KeyError):
        structured_answers.parse_json_answer(
            "{}", StageError, extract=lambda data: data["missing"], caught_exceptions=(ValueError,))


def test_a_truncated_answer_raises_and_a_finished_one_does_not():
    structured_answers.raise_if_answer_was_truncated("end_turn", StageError)
    with pytest.raises(StageError, match=r"^truncated answer \(stop_reason=max_tokens\)$"):
        structured_answers.raise_if_answer_was_truncated("max_tokens", StageError)


def test_template_and_schema_are_loaded_together_and_a_missing_template_gives_none(tmp_path):
    assert structured_answers.load_template_and_answer_schema(tmp_path, "t.txt", "s.json") is None
    assert structured_answers.read_template_if_present(tmp_path, "t.txt") is None
    (tmp_path / "t.txt").write_text("template text", encoding="utf-8")
    (tmp_path / "s.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    assert structured_answers.read_template_if_present(tmp_path, "t.txt") == "template text"
    assert structured_answers.load_template_and_answer_schema(tmp_path, "t.txt", "s.json") == (
        "template text", {"type": "object"})


def test_document_facts_from_plan_reads_the_description_and_three_dealt_values():
    plan = {"description": "A scenario.", "variables": {"document_type": "a news article", "centrality": "central",
                                                       "sentient_category": "farmed fishes", "tone": "dry"}}
    assert select.DocumentFacts.from_plan(plan) == select.DocumentFacts(
        "A scenario.", "a news article", "central", "farmed fishes")
    assert select.DocumentFacts.from_plan({"description": "D", "variables": {}}) == select.DocumentFacts("D", "", "", "")
