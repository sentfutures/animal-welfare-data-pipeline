"""Tests of the standard-library helpers in timed_progress_and_jsonl_helpers.py: reading and appending to completion
records that may end in a partly copied line, and timestamps that never overwrite an earlier output file."""

import json

import pytest

from evals.log_probability_persona_check import timed_progress_and_jsonl_helpers as helpers

COMPLETE_RECORDS = [{"document_id": "document_a", "condition_id": "generic"},
                    {"document_id": "document_a", "condition_id": "condition_one"}]
TRUNCATED_LAST_LINE = '{"document_id": "document_b", "condition_i'


def write_records_with_truncated_last_line(path):
    path.write_text("".join(json.dumps(record) + "\n" for record in COMPLETE_RECORDS) + TRUNCATED_LAST_LINE, encoding="utf-8")


def test_read_jsonl_ignores_a_truncated_final_line(tmp_path):
    path = tmp_path / "scoring_completion_records.jsonl"
    write_records_with_truncated_last_line(path)
    assert helpers.read_jsonl(path) == COMPLETE_RECORDS
    path.write_text(TRUNCATED_LAST_LINE, encoding="utf-8")
    assert helpers.read_jsonl(path) == []


def test_read_jsonl_still_refuses_a_damaged_line_that_is_not_the_last(tmp_path):
    path = tmp_path / "scoring_completion_records.jsonl"
    path.write_text(json.dumps(COMPLETE_RECORDS[0]) + "\n" + TRUNCATED_LAST_LINE + "\n" + json.dumps(COMPLETE_RECORDS[1]) + "\n",
                    encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        helpers.read_jsonl(path)


def test_find_completed_keys_ignores_a_truncated_final_line(tmp_path):
    path = tmp_path / "scoring_completion_records.jsonl"
    write_records_with_truncated_last_line(path)
    assert helpers.find_completed_keys(path, ["document_id", "condition_id"]) == {
        ("document_a", "generic"), ("document_a", "condition_one")}


def test_appending_after_a_truncated_final_line_does_not_glue_the_new_record_onto_it(tmp_path):
    path = tmp_path / "scoring_completion_records.jsonl"
    write_records_with_truncated_last_line(path)
    new_record = {"document_id": "document_b", "condition_id": "generic"}
    helpers.append_jsonl_record(path, new_record)
    assert helpers.read_jsonl(path) == COMPLETE_RECORDS + [new_record]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_appending_after_a_complete_final_record_without_a_newline_keeps_that_record(tmp_path):
    path = tmp_path / "scoring_completion_records.jsonl"
    path.write_text(json.dumps(COMPLETE_RECORDS[0]), encoding="utf-8")
    helpers.append_jsonl_record(path, COMPLETE_RECORDS[1])
    assert helpers.read_jsonl(path) == COMPLETE_RECORDS


def test_timestamp_unused_in_folder_keeps_the_usual_format_and_never_repeats_one_that_names_an_existing_file(
        tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "timestamp_now", lambda: "2026-10-27_16-36")
    templates = ["pod_cost_log_{timestamp}.json", "pod_progress_log_{timestamp}.txt"]
    first = helpers.timestamp_unused_in_folder(tmp_path, templates)
    assert first == "2026-10-27_16-36"
    (tmp_path / f"pod_progress_log_{first}.txt").write_text("first launch")  # any one of the files is enough
    second = helpers.timestamp_unused_in_folder(tmp_path, templates)
    assert second == "2026-10-27_16-36_2"
    (tmp_path / f"pod_cost_log_{second}.json").write_text("second launch")
    assert helpers.timestamp_unused_in_folder(tmp_path, templates) == "2026-10-27_16-36_3"
    assert helpers.timestamp_unused_in_folder(tmp_path, ["unrelated_{timestamp}.txt"]) == "2026-10-27_16-36"
