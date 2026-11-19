"""Offline tests for convert_difficult_advice_answers_into_log_probability_corpus.py."""

import pytest

from evals.log_probability_persona_check import convert_difficult_advice_answers_into_log_probability_corpus as converter


def answer(document_id, stage="stage_2", **overrides):
    record = {"document_id": document_id, "stage": stage, "writer_model": "opus55", "dataset_group": "main_fifty_prompts",
              "prompt_gid": "P-1", "user_message": "Should I keep the hens?", "answer_text": "Keep them.",
              "planted_flaw": None, "rewritten_source_document_id": None}
    record.update(overrides)
    return record


def planted(document_id, flaw, source_id="source"):
    return answer(document_id, stage="planted_flaw_control", planted_flaw=flaw, edited_from_document_id=source_id,
                  answer_text="Edited.")


def test_corpus_extra_documents_and_subset():
    records = [answer("source"), answer("baseline", stage="baseline"), planted("a", "too_much"),
               planted("b", "unhelpful"), answer("pilot", dataset_group="pilot_prompts")]
    corpus, extra, subset_ids = converter.convert_records(records, "preachier", "register_shift")
    assert [document["doc_id"] for document in corpus] == ["source", "baseline", "pilot"]
    assert corpus[0]["content"] == "Keep them." and corpus[0]["variables"]["user_message"] == "Should I keep the hens?"
    assert set(converter.GROUPING_FIELDS) <= set(corpus[0]["variables"])
    assert [(document["document_id"], document["source_document_id"], document["rewrite_direction"])
            for document in extra] == [("a", "source", "preachier"), ("b", "source", "register_shift")]
    assert subset_ids == ["source"]


@pytest.mark.parametrize("bad_records, message", [
    ([answer("x", user_message=" padded ")], "whitespace"),
    ([answer("x"), answer("x")], "Duplicate"),
    ([planted("a", "too_much", "missing")], "not an ordinary answer"),
    ([answer("source", user_message="Other?"), planted("a", "too_much")], "different user message")])
def test_bad_inputs_fail_loudly(bad_records, message):
    with pytest.raises(ValueError, match=message):
        converter.convert_records(bad_records, "preachier", "register_shift")
