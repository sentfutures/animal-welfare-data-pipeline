"""Offline tests for evals/plausibility_audit_sdf.py and its post-processor (the model call is
stubbed one level below shared.openai_api.call_openai; see conftest's stub_openai_responses).

Runs on a synthetic corpus. The shared helpers (run folders, quote
verification, CSV writing, structured-output retry) are covered in test_model_judged_audit_support.py.
"""

import copy
import csv as csv_module
import json
from pathlib import Path

import pytest

from evals import plausibility_audit_postprocess_sdf as postprocess
from evals import plausibility_audit_sdf as judge
from evals.model_judged_audit_support import assert_schema_is_strict_valid
from shared import openai_api, utils


@pytest.fixture(scope="module")
def definition():
    return judge.JudgeDefinition()


def valid_axis(score=5, applicable=True, problems=()):
    return {"applicable": applicable, "problems": list(problems), "score_reasoning": "r", "score": score}


def valid_parsed_result(axis_keys, overall=5):
    return {"axes": {key: valid_axis() for key in axis_keys}, "other_implausibilities": [],
            "overall_reasoning": "ok", "overall_plausibility": overall}


def problem(quote="The fish may feel pain.", severity="minor"):
    return {"quote": quote, "english_gloss": None, "explanation": "e", "severity": severity}


def synthetic_corpus(count=3):
    return [{"doc_id": f"matrix_{index:06d}", "content": f"Document number {index}. The fish may feel pain.",
             "description": "SECRET-DESCRIPTION-DO-NOT-LEAK", "language": "English" if index % 2 else "Spanish",
             "type_name": "essay", "variables": {"document_type": "essay", "tone": "dry"}}
            for index in range(count)]


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "sdf_corpus.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in synthetic_corpus()) + "\n", encoding="utf-8")
    return path


# ---- request content and schema ----

def test_request_contains_only_instructions_and_content(definition):
    for document in synthetic_corpus():
        arguments = judge.build_request_arguments(definition, document["content"], "medium")
        assert arguments["input"] == document["content"]
        assert arguments["instructions"] == definition.instructions_text
        assert arguments["temperature"] == 1 and "top_p" not in arguments
        assert arguments["reasoning"] == {"effort": "medium"}
        assert arguments["prompt_cache_key"] == judge.SCHEMA_NAME
        sent = json.dumps(arguments, ensure_ascii=False)
        for value in (document["doc_id"], document["description"], "essay", "dry"):
            if value not in document["content"] and value not in definition.instructions_text:
                assert value not in sent, f"leaked: {value}"


def test_axis_keys_match_prompt_headers_and_there_are_ten(definition):
    headers = [line[4:].strip() for line in definition.instructions_text.splitlines() if line.startswith("### axis_")]
    assert definition.axis_keys == headers
    assert len(headers) == 10 and len(set(headers)) == 10
    assert list(definition.schema["properties"]["axes"]["properties"].keys()) == headers


def test_the_judging_prompt_text_is_unchanged():
    # sha256 of prompts/tools/plausibility_audit_judge_instructions.txt; an intentional prompt change updates this hash.
    import hashlib
    digest = hashlib.sha256(judge.instructions_file_path().read_bytes()).hexdigest()
    assert digest == "44d74427d23ce8d382825badfc18b4e9970ce4da9451e8c729f8d41a4c35124a"


def test_schema_is_strict_valid_and_ordered(definition):
    assert_schema_is_strict_valid(definition.schema)
    assert list(definition.schema["properties"].keys()) == [
        "axes", "other_implausibilities", "overall_reasoning", "overall_plausibility"]
    axis = next(iter(definition.schema["properties"]["axes"]["properties"].values()))
    assert list(axis["properties"].keys()) == ["applicable", "problems", "score_reasoning", "score"]
    assert axis["properties"]["problems"]["maxItems"] == 3
    assert list(axis["properties"]["problems"]["items"]["properties"].keys()) == [
        "quote", "english_gloss", "explanation", "severity"]
    assert axis["properties"]["problems"]["items"]["properties"]["english_gloss"]["type"] == ["string", "null"]
    assert axis["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]
    other = definition.schema["properties"]["other_implausibilities"]
    assert other["maxItems"] == 3
    assert list(other["items"]["properties"].keys()) == [
        "suggested_axis_name", "description", "quote", "english_gloss", "explanation", "severity"]


# ---- validation rejects instead of storing ----

def test_valid_result_passes(definition):
    judge.validate_parsed_result(valid_parsed_result(definition.axis_keys), definition)


@pytest.mark.parametrize("bad_score", [0, 6, -1, 3.5, "4", None, True])
def test_out_of_range_axis_score_is_rejected(definition, bad_score):
    result = valid_parsed_result(definition.axis_keys)
    result["axes"][definition.axis_keys[3]]["score"] = bad_score
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)


@pytest.mark.parametrize("bad_score", [0, 6, None])
def test_out_of_range_overall_score_is_rejected(definition, bad_score):
    with pytest.raises(ValueError):
        judge.validate_parsed_result(valid_parsed_result(definition.axis_keys, overall=bad_score), definition)


def test_missing_or_extra_axis_is_rejected(definition):
    result = valid_parsed_result(definition.axis_keys)
    del result["axes"][definition.axis_keys[0]]
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)
    result = valid_parsed_result(definition.axis_keys)
    result["axes"]["axis_99_invented"] = valid_axis()
    with pytest.raises(ValueError):
        judge.validate_parsed_result(result, definition)


def test_bad_problem_entries_are_rejected(definition):
    for mutate in (lambda a: a["problems"].extend([problem()] * 4),
                   lambda a: a["problems"].append(problem(severity="catastrophic")),
                   lambda a: a["problems"].append({"quote": "x"})):
        result = valid_parsed_result(definition.axis_keys)
        mutate(result["axes"][definition.axis_keys[0]])
        with pytest.raises(ValueError):
            judge.validate_parsed_result(result, definition)


# ---- the model call, stubbed at the network seam ----

@pytest.fixture
def cost_log(tmp_path, monkeypatch):
    from shared import api
    path = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(api, "_cost_log_path", path)
    return path


def test_one_document_makes_one_call_and_keeps_usage_and_raw_text(
        definition, stub_openai_responses, openai_response, cost_log):
    answer = json.dumps(valid_parsed_result(definition.axis_keys))
    calls = stub_openai_responses([openai_response(text=answer, input_tokens=1000, cached_tokens=200,
                                                   output_tokens=300, reasoning_tokens=100)])
    record, error = judge.judge_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert error is None and len(calls) == 1
    assert record["doc_id"] == "d" and record["model"] == "gpt-5.6-luna" and record["reasoning_effort"] == "low"
    assert record["raw_output_text"] == answer and record["parse_attempts"] == 1
    assert record["usage"] == {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 300,
                               "reasoning_tokens": 100}
    request = calls[0]
    assert request["model"] == "gpt-5.6-luna" and request["input"] == "x"
    assert request["instructions"] == definition.instructions_text
    assert request["prompt_cache_key"] == judge.SCHEMA_NAME
    assert request["text"]["format"]["strict"] is True
    logged = utils.load_jsonl(cost_log)
    assert logged[0]["stage"] == judge.STAGE_TAG and logged[0]["item_id"] == "d" and logged[0]["backend"] == "openai"
    assert logged[0]["cost_usd"] == pytest.approx((800 * 0.20 + 200 * 0.02 + 300 * 1.20) / 1_000_000)


def test_invalid_response_becomes_error_record_not_stored_judgment(
        definition, stub_openai_responses, openai_response, cost_log):
    bad_result = valid_parsed_result(definition.axis_keys)
    bad_result["axes"][definition.axis_keys[0]]["score"] = 9
    calls = stub_openai_responses(lambda request: openai_response(
        text=json.dumps(bad_result), input_tokens=10, output_tokens=5, cached_tokens=0, reasoning_tokens=0))
    record, error = judge.judge_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert record is None and error["error_kind"] == "parse_or_schema_failure"
    assert error["usage"]["input_tokens"] == 20 and len(calls) == 2  # both attempts made and counted


def test_a_refused_answer_is_retried_once_then_recorded_as_a_failure(
        definition, stub_openai_responses, openai_response, cost_log):
    calls = stub_openai_responses(lambda request: openai_response(refusal="no"))
    record, error = judge.judge_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert record is None and error["error_kind"] == "parse_or_schema_failure" and len(calls) == 2
    assert "refusal" in error["description"]


def test_a_malformed_first_answer_is_retried_and_the_second_stored(
        definition, stub_openai_responses, openai_response, cost_log):
    replies = [openai_response(text="not json", input_tokens=10, output_tokens=1),
               openai_response(text=json.dumps(valid_parsed_result(definition.axis_keys)), input_tokens=10,
                               output_tokens=2)]
    stub_openai_responses(replies)
    record, error = judge.judge_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert error is None and record["parse_attempts"] == 2 and record["usage"]["input_tokens"] == 20


# ---- resume ----

def test_resume_skips_finished_documents(tmp_path, monkeypatch, definition):
    (tmp_path / "judgments_2026-01-01_00-00.jsonl").write_text(json.dumps({"doc_id": "a", "usage": {}}) + "\n")
    completed = judge.find_completed_document_ids(tmp_path)
    assert completed == {"a"}
    calls = []

    def fake_judge(judge_definition, document, effort):
        calls.append(document["doc_id"])
        return {"doc_id": document["doc_id"], "usage": {"input_tokens": 1, "cached_input_tokens": 0,
                                                        "output_tokens": 1, "reasoning_tokens": 0}}, None

    monkeypatch.setattr(judge, "judge_one_document", fake_judge)
    documents = [{"doc_id": "a", "content": "x"}, {"doc_id": "b", "content": "y"}]
    judge.run_judging(tmp_path, "t", definition, documents, "low", 2, completed, [])
    assert calls == ["b"]


def test_resume_refuses_changed_prompt_model_or_effort(tmp_path, corpus_path):
    configuration = judge.build_run_configuration("medium", corpus_path)
    tampered = copy.deepcopy(configuration)
    tampered["prompt_file_sha256"][judge.INSTRUCTIONS_FILE_NAME] = "0" * 64
    configuration_path = tmp_path / "run_configuration_2026-01-01_00-00.json"
    configuration_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        judge.check_resume_configuration_matches(tmp_path, configuration)
    configuration_path.write_text(json.dumps(configuration))
    judge.check_resume_configuration_matches(tmp_path, configuration)
    changed_effort = copy.deepcopy(configuration)
    changed_effort["reasoning_effort"] = "high"
    with pytest.raises(RuntimeError):
        judge.check_resume_configuration_matches(tmp_path, changed_effort)


# ---- whole run through main() with the model stubbed ----

def run_main(tmp_path, corpus_path, *extra):
    return judge.main(["--input", str(corpus_path), "--run-name", "synthetic_check",
                       "--runs-directory", str(tmp_path / "runs"), "--workers", "2", *extra])


def test_a_partial_run_writes_judgments_cost_log_and_model_call_log_then_resumes_with_no_calls(
        tmp_path, corpus_path, definition, stub_openai_responses, openai_response):
    answer = json.dumps(valid_parsed_result(definition.axis_keys))
    calls = stub_openai_responses(lambda request: openai_response(text=answer))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "2")
    assert run_folder.parent == tmp_path / "runs" / "small_sample_checks"
    assert len(calls) == 2
    judgments = list(run_folder.glob("judgments_*.jsonl"))
    assert len(judgments) == 1 and len(utils.load_jsonl(judgments[0])) == 2
    assert len(utils.load_jsonl(next(run_folder.glob("model_call_cost_log_*.jsonl")))) == 2
    assert len(utils.load_jsonl(run_folder / "model_calls.jsonl")) == 2
    assert next(run_folder.glob("instructions_sent_*.txt")).read_text() == definition.instructions_text
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text())
    assert configuration["model"] == "gpt-5.6-luna" and configuration["axis_keys"] == definition.axis_keys

    calls.clear()
    judge.main(["--input", str(corpus_path), "--run-name", "synthetic_check", "--runs-directory",
                str(tmp_path / "runs"), "--document-limit", "2", "--resume-folder", str(run_folder)])
    assert calls == []


def test_a_full_run_goes_to_the_runs_directory_itself(
        tmp_path, corpus_path, definition, stub_openai_responses, openai_response):
    answer = json.dumps(valid_parsed_result(definition.axis_keys))
    calls = stub_openai_responses(lambda request: openai_response(text=answer))
    run_folder = run_main(tmp_path, corpus_path)
    assert run_folder.parent == tmp_path / "runs" and len(calls) == 3


def test_the_default_runs_directory_is_the_audit_folder_of_the_run(tmp_path):
    assert judge.default_runs_directory(tmp_path) == tmp_path / "audit" / "plausibility_audit_runs"
    corpus = tmp_path / "final" / "sdf_corpus.jsonl"
    assert judge.default_runs_directory(corpus) == tmp_path / "final" / "audit" / "plausibility_audit_runs"


def test_estimate_only_extrapolates_from_the_sample_usage(
        tmp_path, corpus_path, definition, stub_openai_responses, openai_response):
    answer = json.dumps(valid_parsed_result(definition.axis_keys))
    stub_openai_responses(lambda request: openai_response(text=answer, input_tokens=1000, cached_tokens=0,
                                                          output_tokens=100, reasoning_tokens=0))
    run_folder = run_main(tmp_path, corpus_path, "--estimate-only", "--estimate-sample-size", "2")
    estimate = json.loads(next(run_folder.glob("estimate_*.json")).read_text())
    assert estimate["corpus_document_count"] == 3 and len(estimate["sample_doc_ids"]) == 2
    assert estimate["total_cost_dollars"] > 0


def test_unknown_document_ids_exit_with_error(tmp_path, corpus_path):
    with pytest.raises(SystemExit) as exit_info:
        judge.main(["--input", str(corpus_path), "--run-name", "unknown_ids_check",
                    "--runs-directory", str(tmp_path / "runs"),
                    "--document-ids", "matrix_000000", "missing_one", "--estimate-only"])
    assert "missing_one" in str(exit_info.value)


def test_the_model_has_a_price_in_the_shared_table():
    assert judge.MODEL_NAME in openai_api._PRICING
    assert openai_api.is_openai_model(judge.MODEL_NAME)


# ---- post-processing ----

def test_variables_parse_from_string_form():
    assert postprocess.parse_variables("{'tone': 'dry', 'domain': \"it's\", 'n': 3}") == {
        "tone": "dry", "domain": "it's", "n": 3}
    assert postprocess.parse_variables({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        postprocess.parse_variables("__import__('os').getcwd()")
    with pytest.raises(ValueError):
        postprocess.parse_variables("[1, 2]")


def test_quote_spanning_paragraph_break_is_flagged():
    document = {"content": "First paragraph ends here.\n\nSecond paragraph starts here.", "language": "English",
                "variables": {"document_type": "essay"}}
    spanning = {"quote": "ends here. Second paragraph starts", "english_gloss": None, "explanation": "e",
                "severity": "minor"}
    within = problem("First paragraph ends here.")
    row_spanning = postprocess.build_quoted_row("d", document, {}, {}, spanning)
    row_within = postprocess.build_quoted_row("d", document, {}, {}, within)
    assert not row_spanning["quote_found_in_document"] and row_spanning["quote_spans_paragraph_break"]
    assert row_within["quote_found_in_document"] and not row_within["quote_spans_paragraph_break"]


def build_synthetic_run(tmp_path, definition):
    axes = definition.axis_keys
    corpus = [
        {"doc_id": "good_doc", "content": "The fish may feel pain.\n\nA second paragraph.", "language": "English",
         "variables": "{'document_type': 'essay', 'tone': 'dry'}"},
        {"doc_id": "other_doc", "content": "Texto en español.", "language": "Spanish",
         "variables": {"document_type": "blog", "tone": "dry"}},
        {"doc_id": "failed_doc", "content": "x", "language": "English",
         "variables": {"document_type": "essay", "tone": "warm"}}]
    corpus_file = tmp_path / "synthetic_corpus_for_postprocessing.jsonl"
    corpus_file.write_text("\n".join(json.dumps(d) for d in corpus) + "\n")
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"corpus_path": str(corpus_file), "axis_keys": axes}))
    first = valid_parsed_result(axes, overall=2)
    first["axes"][axes[0]] = valid_axis(score=2, problems=[problem(severity="minor"),
                                                         problem(severity="breaks_believability")])
    first["axes"][axes[1]] = valid_axis(score=1, applicable=True)
    first["axes"][axes[5]] = valid_axis(score=5, applicable=False)
    first["other_implausibilities"] = [{"suggested_axis_name": "n", "description": "d",
                                        "quote": "A second paragraph.", "english_gloss": None,
                                        "explanation": "e", "severity": "noticeable"}]
    second = valid_parsed_result(axes, overall=4)
    second["axes"][axes[1]] = valid_axis(score=3)
    second["axes"][axes[5]] = valid_axis(score=5, applicable=False)
    records = [{"doc_id": "good_doc", "parsed_result": first, "usage": {}},
               {"doc_id": "other_doc", "parsed_result": second, "usage": {}}]
    (run_folder / "judgments_2026-01-01_00-00.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_folder / "errors_2026-01-01_00-00.jsonl").write_text(
        json.dumps({"doc_id": "failed_doc", "error_kind": "parse_or_schema_failure", "usage": {}}) + "\n")
    return run_folder, axes


def test_failed_documents_have_empty_cells_and_inapplicable_axes_are_excluded(tmp_path, definition):
    run_folder, axes = build_synthetic_run(tmp_path, definition)
    paths, rows = postprocess.postprocess_run_folder(run_folder)
    for path in paths.values():
        assert Path(path).exists(), path
    with open(paths["per_document_scores_csv"], encoding="utf-8") as handle:
        csv_rows = {r["doc_id"]: r for r in csv_module.DictReader(handle)}
    failed = csv_rows["failed_doc"]
    assert failed["judgment_status"] == "failed" and failed["overall_plausibility"] == ""
    for axis_key in axes:
        for column in postprocess.axis_columns(axis_key):
            assert failed[column] == ""
    assert failed["tone"] == "warm"  # variables are joined even for failed documents
    good = csv_rows["good_doc"]
    assert good["overall_plausibility"] == "2" and good["tone"] == "dry"  # string-form variables parsed
    assert good[f"{axes[0]}__problem_count"] == "2" and good[f"{axes[0]}__worst_severity"] == "breaks_believability"
    assert good[f"{axes[2]}__worst_severity"] == "" and good[f"{axes[2]}__problem_count"] == "0"
    assert good[f"{axes[5]}__applicable"] == "False"

    summary = {r["axis"]: r for r in rows["axis_summary"]}
    assert summary[axes[5]]["documents_applicable"] == 0 and summary[axes[5]]["mean_score_among_applicable"] is None
    assert summary[axes[1]]["mean_score_among_applicable"] == 2.0  # (1 + 3) / 2
    assert summary[axes[0]]["problems_with_severity_minor"] == 1
    assert summary[axes[0]]["problems_with_severity_breaks_believability"] == 1
    assert summary[axes[1]]["applicable_documents_with_score_1"] == 1

    variable_means = {(r["variable"], r["value"]): r for r in rows["variable_means"]}
    assert ("tone", "warm") not in variable_means  # failed document contributes nothing
    dry = variable_means[("tone", "dry")]
    assert dry["document_count"] == 2 and dry["overall_plausibility_mean"] == 3.0
    assert dry[f"{axes[1]}__mean_score_among_applicable"] == 2.0
    assert dry[f"{axes[5]}__mean_score_among_applicable"] is None and dry[f"{axes[5]}__applicable_document_count"] == 0

    assert len(rows["problems"]) == 2 and rows["problems"][0]["axis"] == axes[0]
    assert rows["problems"][0]["tone"] == "dry" and rows["problems"][0]["quote_found_in_document"] is True
    assert len(rows["other"]) == 1 and rows["other"][0]["quote_found_in_document"] is True
