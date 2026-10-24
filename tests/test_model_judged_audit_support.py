"""Offline tests for evals/model_judged_audit_support.py, the shared helpers of the plausibility and
shoehorning audits. The model call
is stubbed at the network seam (conftest's stub_openai_responses)."""

import json
from types import SimpleNamespace

import pytest

from evals import model_judged_audit_support as support
from shared import api


def test_quote_verification_tolerates_whitespace_and_curly_quotes():
    document = 'She said, "The fish may feel pain,"  and\nthen left. Wir sollten Tiere nicht leiden lassen.'
    assert support.verify_quote("“The fish may feel pain,” and then left.", document)[0]
    assert support.verify_quote("and then   left.", document)[1]


def test_quote_verification_rejects_fabricated_quote():
    document = "She said the fish may feel pain and then left."
    found, exact, score, spans_break = support.verify_quote(
        "Octopuses deserve legal protection under the new directive.", document)
    assert not found and not exact and score < support.FUZZY_MATCH_THRESHOLD and not spans_break


def test_quote_spanning_paragraph_break_is_rejected_but_within_a_paragraph_passes():
    document = ("Hay dos preguntas. Ese beneficio es agronómico y es tuyo.\n\n"
                "La segunda es si el escarabajo puede sufrir.  \n \n\nOtro párrafo distinto.")
    quote = "Ese beneficio es agronómico y es tuyo. La segunda es si el escarabajo puede sufrir."
    found, exact, score, spans_break = support.verify_quote(quote, document)
    assert not found and not exact and spans_break
    found, exact, _, spans_break = support.verify_quote("Otro párrafo distinto.", document)
    assert found and exact and not spans_break


@pytest.mark.parametrize("document_quote, judge_quote", [
    ("the hens were moved indoors because the weather turned cold, and the farm kept records",
     "The hens were moved indoors because the weather turned cold, and the farm kept records"),
    ("the hens were moved indoors - because the weather turned cold - and the farm kept records",
     "the hens were moved indoors \u2014 because the weather turned cold \u2014 and the farm kept records")])
def test_a_one_character_difference_gives_the_same_verdict_at_every_offset_in_the_paragraph(
        monkeypatch, document_quote, judge_quote):
    monkeypatch.setattr(support, "rapidfuzz_module", None)
    verdicts = {}
    for offset in range(0, 45):
        document = f"{'w' * offset} that {document_quote} of every move."
        found, exact, score, _ = support.verify_quote(judge_quote, document)
        verdicts[offset] = (found, score >= support.FUZZY_MATCH_THRESHOLD)
    assert set(verdicts.values()) == {(True, True)}, verdicts


def test_a_quote_longer_than_the_paragraph_is_not_found_by_the_rapidfuzz_path_either(monkeypatch):
    class PartialRatioThatAlignsTheShorterStringInsideTheLonger:
        @staticmethod
        def partial_ratio(first_text, second_text):
            return 100.0

        @staticmethod
        def ratio(first_text, second_text):
            return 10.0

    monkeypatch.setattr(support, "rapidfuzz_module", PartialRatioThatAlignsTheShorterStringInsideTheLonger)
    document = "Housing\n\nThe hens were moved indoors because the weather turned cold."
    quote = "Housing: the hens were moved indoors because the weather turned cold, and the farm kept records."
    found, exact, score, spans_break = support.verify_quote(quote, document)
    assert not found and score < support.FUZZY_MATCH_THRESHOLD


def test_find_completed_keys_reads_back_appended_records(tmp_path):
    writer = support.AppendOnlyJsonlWriter()
    writer.append(tmp_path / "classifications_x_2026-01-01_00-00.jsonl", {"doc_id": "a", "pass": "p"})
    writer.append(tmp_path / "classifications_y_2026-01-01_00-00.jsonl", {"doc_id": "b", "pass": "q"})
    assert support.find_completed_keys(tmp_path, "classifications_*.jsonl", ["doc_id", "pass"]) == {
        ("a", "p"), ("b", "q")}


def test_resume_refuses_changed_file_hashes_and_folders_without_configuration(tmp_path):
    configuration = {"prompt_file_sha256": {"a.txt": "a" * 64, "b.txt": "b" * 64}}
    with pytest.raises(RuntimeError):
        support.check_resume_is_compatible(tmp_path, configuration)
    tampered = json.loads(json.dumps(configuration))
    tampered["prompt_file_sha256"]["a.txt"] = "0" * 64
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        support.check_resume_is_compatible(tmp_path, configuration)
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(configuration))
    support.check_resume_is_compatible(tmp_path, configuration)


def test_resume_refuses_a_changed_corpus_and_names_it(tmp_path):
    recorded = {"prompt_file_sha256": {"a.txt": "a" * 64}, "corpus_sha256": "1" * 64, "corpus_path": "old/corpus.jsonl"}
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(recorded))
    support.check_resume_is_compatible(tmp_path, recorded)
    changed = {**recorded, "corpus_sha256": "2" * 64, "corpus_path": "new/corpus.jsonl"}
    with pytest.raises(RuntimeError, match="corpus changed.*old/corpus.jsonl.*new/corpus.jsonl"):
        support.check_resume_is_compatible(tmp_path, changed)


def test_resume_refuses_a_changed_compared_setting_and_says_from_what_to_what(tmp_path):
    recorded = {"prompt_file_sha256": {"a.txt": "a" * 64}, "reasoning_effort": "low"}
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(recorded))
    support.check_resume_is_compatible(tmp_path, recorded, compared_settings=("reasoning_effort",))
    with pytest.raises(RuntimeError, match="reasoning_effort changed from 'low' to 'high'"):
        support.check_resume_is_compatible(tmp_path, {**recorded, "reasoning_effort": "high"},
                                           compared_settings=("reasoning_effort",))


def test_strict_schema_helpers_accept_and_reject():
    schema = support.build_strict_object_schema({
        "a": {"type": "string"},
        "b": support.build_array_schema(support.build_strict_object_schema({"c": {"type": "boolean"}}), 3)})
    support.assert_schema_is_strict_valid(schema)
    assert schema["properties"]["b"]["maxItems"] == 3
    loose = {"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False}
    assert support.find_strict_schema_violations(loose)


def test_timestamp_appended_exactly_once(tmp_path):
    folder, _ = support.choose_run_folder(tmp_path / "small_sample_checks", "smoke_test", None)
    assert len(support.TIMESTAMP_SUFFIX_PATTERN.findall(folder.name)) == 1
    assert folder.parent == tmp_path / "small_sample_checks" and folder.name.startswith("smoke_test_")
    already, _ = support.choose_run_folder(tmp_path, "named_2026-10-07_12-00", None)
    assert already.name == "named_2026-10-07_12-00"


def test_a_run_name_already_used_must_be_resumed_not_reused(tmp_path):
    support.choose_run_folder(tmp_path, "same_name", None)
    with pytest.raises(SystemExit, match="resume-folder"):
        support.choose_run_folder(tmp_path, "same_name", None)


def test_resume_folder_works_in_either_location(tmp_path):
    for location in (tmp_path / "top_2026-10-07_12-00", tmp_path / "small_sample_checks" / "s_2026-10-07_12-00"):
        location.mkdir(parents=True)
        folder, is_resume = support.choose_run_folder(tmp_path / "elsewhere", "x", str(location))
        assert folder == location and is_resume


@pytest.mark.parametrize("extra, partial", [
    ({}, False), ({"estimate_only": True}, True), ({"document_ids": ["a"]}, True), ({"document_limit": 5}, True)])
def test_every_partial_run_goes_to_small_sample_checks(tmp_path, extra, partial):
    arguments = SimpleNamespace(estimate_only=False, document_ids=None, document_limit=None)
    for name, value in extra.items():
        setattr(arguments, name, value)
    kind, parent = support.choose_runs_parent_directory(arguments, tmp_path)
    assert (parent == tmp_path / "small_sample_checks") is partial
    assert kind.startswith("partial run") is partial


def test_stratified_sample_is_deterministic_and_mixed():
    documents = [{"doc_id": f"d{i}", "language": "English" if i % 2 else "German", "content": "x" * (10 + i * 7)}
                 for i in range(12)]
    first = support.choose_stratified_sample(documents, 6)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in support.choose_stratified_sample(documents, 6)]
    assert {d["language"] == "English" for d in first} == {True, False}


def test_corpus_path_of_accepts_a_run_directory_or_a_file(tmp_path):
    (tmp_path / "final").mkdir()
    corpus = tmp_path / "final" / "sdf_corpus.jsonl"
    corpus.write_text("{}\n")
    assert support.corpus_path_of(tmp_path) == corpus and support.corpus_path_of(corpus) == corpus
    assert support.run_directory_of(corpus) == tmp_path and support.run_directory_of(tmp_path) == tmp_path
    with pytest.raises(SystemExit):
        support.corpus_path_of(tmp_path / "missing.jsonl")


def test_write_rows_to_csv_tolerates_unexpected_and_missing_fields(tmp_path):
    import csv
    path = tmp_path / "rows.csv"
    support.write_rows_to_csv(path, [{"b": 1, "new_field": {"x": 1}, "c": None, "d": ["p", "q"]}, {"a": 2, "b": 3}],
                              ["a", "b"])
    with open(path, encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0].keys()) == ["a", "b", "new_field", "c", "d"]
    assert csv_rows[0]["new_field"] == '{"x": 1}' and csv_rows[1]["new_field"] == ""
    assert csv_rows[0]["c"] == "" and csv_rows[0]["d"] == "p;q"


def test_usage_totals_cost_and_cache_rate(tmp_path):
    totals = support.RunningUsageTotals("gpt-5.6-luna")
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 500_000, "output_tokens": 1_000_000,
             "reasoning_tokens": 4}
    totals.add_usage(usage)
    assert totals.totals()["calls"] == 1 and totals.totals()["cache_rate"] == 0.5
    assert totals.totals()["cost_dollars"] == pytest.approx(0.1 + 0.01 + 1.2)
    log_path = tmp_path / "log.json"
    totals.write_log_file(log_path, "note")
    assert set(json.loads(log_path.read_text())) == {"model", "note", "totals"}


# ---- the structured-output request, through shared.openai_api ----

SCHEMA = support.build_strict_object_schema({"x": {"type": "string"}})


@pytest.fixture
def cost_log(tmp_path, monkeypatch):
    path = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(api, "_cost_log_path", path)
    return path


def ask(**extra):
    return support.request_structured_output("gpt-5.6-luna", "INSTRUCTIONS", "DOCUMENT", "schema_name", SCHEMA,
                                             "low", stage="eval_x", item_id="doc", **extra)


def test_structured_output_sends_the_documented_request(stub_openai_responses, openai_response, cost_log):
    calls = stub_openai_responses([openai_response(text='{"x": "y"}')])
    result = ask(prompt_cache_key="cache_key")
    assert result["parsed_result"] == {"x": "y"} and result["parse_attempts"] == 1
    request = calls[0]
    assert request["instructions"] == "INSTRUCTIONS" and request["input"] == "DOCUMENT"
    assert request["temperature"] == 1 and "top_p" not in request
    assert request["reasoning"] == {"effort": "low"} and request["prompt_cache_key"] == "cache_key"
    assert request["max_output_tokens"] == support.MAXIMUM_OUTPUT_TOKENS
    assert request["text"]["format"] == {"type": "json_schema", "name": "schema_name", "schema": SCHEMA, "strict": True}


def test_no_prompt_cache_key_is_sent_unless_asked_for(stub_openai_responses, openai_response, cost_log):
    calls = stub_openai_responses([openai_response(text='{"x": "y"}')])
    ask()
    assert "prompt_cache_key" not in calls[0]


def test_a_cut_off_answer_is_retried_and_usage_is_summed_over_attempts(
        stub_openai_responses, openai_response, cost_log):
    stub_openai_responses([
        openai_response(status="incomplete", incomplete_reason="max_output_tokens", text='{"x"',
                        input_tokens=100, output_tokens=40, cached_tokens=10, reasoning_tokens=30),
        openai_response(text='{"x": "y"}', input_tokens=100, output_tokens=5, cached_tokens=20,
                        reasoning_tokens=1)])
    result = ask()
    assert result["parse_attempts"] == 2
    assert result["usage"] == {"input_tokens": 200, "cached_input_tokens": 30, "output_tokens": 45,
                               "reasoning_tokens": 31}
    assert len(support.read_jsonl(cost_log)) == 2


def test_two_unusable_answers_raise_a_parse_failure_carrying_the_usage(
        stub_openai_responses, openai_response, cost_log):
    stub_openai_responses(lambda request: openai_response(text="nope", input_tokens=7, output_tokens=3,
                                                          cached_tokens=0, reasoning_tokens=0))
    with pytest.raises(support.StructuredOutputParseFailure) as failure:
        ask()
    assert failure.value.usage["input_tokens"] == 14 and failure.value.raw_output_text == "nope"
    assert "attempt 2" in failure.value.description


def test_the_validator_decides_what_counts_as_usable(stub_openai_responses, openai_response, cost_log):
    def reject(parsed):
        raise ValueError("no")
    stub_openai_responses(lambda request: openai_response(text='{"x": "y"}'))
    with pytest.raises(support.StructuredOutputParseFailure):
        ask(validate_parsed_result=reject)


# ---- additions to shared/openai_api.py and shared/api.py made for these audits ----

def test_openai_call_passes_the_cache_key_and_reports_usage(stub_openai_responses, openai_response, cost_log):
    from shared import openai_api
    calls = stub_openai_responses([openai_response(input_tokens=900, cached_tokens=300, output_tokens=80,
                                                   reasoning_tokens=60)])
    usage = {}
    openai_api.call_openai("u", "s", model="gpt-5.6-luna", max_output_tokens=1000, json_schema=SCHEMA,
                           prompt_cache_key="k", usage_out=usage)
    assert calls[0]["prompt_cache_key"] == "k"
    assert usage == {"input_tokens": 900, "cached_input_tokens": 300, "output_tokens": 80, "reasoning_tokens": 60}


def test_openai_call_without_a_cache_key_sends_none(stub_openai_responses, openai_response, cost_log):
    from shared import openai_api
    calls = stub_openai_responses([openai_response()])
    openai_api.call_openai("u", "s", model="gpt-6.1-sol", max_output_tokens=1000)
    assert "prompt_cache_key" not in calls[0]


def test_init_cost_log_only_needs_no_config_or_anthropic_key(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_cost_log_path", None)
    api.init_cost_log_only(tmp_path / "nested" / "cost_log.jsonl")
    assert (tmp_path / "nested").is_dir() and api._cost_log_path == tmp_path / "nested" / "cost_log.jsonl"


def test_init_cost_log_only_records_full_model_calls_beside_its_own_cost_log(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_cost_log_path", None)
    monkeypatch.setattr(api, "_record_model_calls", False)
    api.init_cost_log_only(tmp_path / "run" / "cost_log.jsonl")
    api._log_model_call("gpt-6.1-sol", "openai", "system text", ["user text"], response_text="answer")
    records = (tmp_path / "run" / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1 and json.loads(records[0])["model"] == "gpt-6.1-sol"


@pytest.mark.parametrize("module_name", ["plausibility_audit_sdf", "shoehorning_audit_sdf"])
def test_the_audits_accept_the_find_dotenv_flag(module_name):
    import importlib
    from shared import api_keys
    module = importlib.import_module(f"evals.{module_name}")
    default = module.parse_arguments(["--run-name", "x"])
    flagged = module.parse_arguments(["--run-name", "x", "--find-dotenv"])
    assert default.api_key_source_mode == api_keys.REPO_ROOT_DOTENV_ONLY
    assert flagged.api_key_source_mode == api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT
