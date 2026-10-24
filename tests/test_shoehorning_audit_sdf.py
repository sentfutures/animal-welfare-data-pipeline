"""Offline tests for evals/shoehorning_audit_sdf.py and its post-processor (the model call is
stubbed one level below shared.openai_api.call_openai; see conftest's stub_openai_responses).

Runs on a synthetic corpus. The shared helpers (run folders, quote
verification, CSV writing, structured-output retry) are covered in test_model_judged_audit_support.py.
"""

import hashlib
import json
from pathlib import Path

import pytest

from evals import shoehorning_audit_postprocess_sdf as postprocess
from evals import shoehorning_audit_sdf as classifier
from evals.model_judged_audit_support import assert_schema_is_strict_valid as assert_strict_valid
from evals.model_judged_audit_support import check_resume_is_compatible
from shared import utils

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"

# sha256 of the prompt files in prompts/tools/ when the audit was added; an intentional prompt change updates them.
PINNED_PROMPT_HASHES = {
    "shoehorning_audit_shared_header.txt": "a6ee4ac0fb93c25a333ac1a0068667d1f380685d4a308dbef97fa05d7d192ead",
    "shoehorning_audit_pass_moral_considerations.txt": "6e2da3fee62714f68d45436d9ec0cb55f3b0c3c54c019481fd00c204f01ed955",
    "shoehorning_audit_pass_field_distinctions.txt": "bb91da17d15af0d4375d2aedbb5c87ed8107f76f9dd38db4bbce73325d3c9a26",
    "shoehorning_audit_pass_epistemic_notes.txt": "b5986cd473488d72b8bb812238cb9e89368b05d9215f18b798edc77c8bc2dde7",
}


def synthetic_corpus(count=3):
    return [{"doc_id": f"matrix_{index:06d}", "content": f"Document number {index}. The fish may feel pain.",
             "description": "SECRET-DESCRIPTION-DO-NOT-LEAK", "language": "English" if index % 2 else "Spanish",
             "variables": {"document_type": "essay", "tone": "dry"}} for index in range(count)]


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "sdf_corpus.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in synthetic_corpus()) + "\n", encoding="utf-8")
    return path


def empty_answer(pass_name):
    definition = classifier.PassDefinition(pass_name)
    return {"items": {item["item_key"]: {"occurrences": []} for item in definition.parsed_items},
            "other_considerations": []}


def test_the_prompt_files_are_unchanged():
    for file_name, expected in PINNED_PROMPT_HASHES.items():
        assert hashlib.sha256((classifier.PROMPT_DIRECTORY / file_name).read_bytes()).hexdigest() == expected


def item_keys(pass_name):
    return [i["item_key"] for i in classifier.PassDefinition(pass_name).parsed_items]


def item_numbers(pass_name):
    return [int(k.split("_")[1]) for k in item_keys(pass_name)]


def test_item_counts_and_numbering():
    assert item_numbers("moral_considerations") == list(range(1, 10))
    assert item_numbers("field_distinctions") == list(range(10, 17))
    assert item_numbers("epistemic_notes") == list(range(17, 26))


def test_sub_flag_names_parsed():
    moral = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("moral_considerations").parsed_items}
    assert moral["item_01_natural_or_traditional_does_not_make_it_ok"] == []
    assert moral["item_03_open_or_expanding_moral_circle"] == ["historical_comparison"]
    assert moral["item_04_scope_sensitivity"] == ["scale_built_over_time_or_space", "even_a_few_still_count"]
    assert moral["item_07_welfare_framed_as_business_risk"] == ["risk_framing_present", "correction_present"]
    field = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("field_distinctions").parsed_items}
    assert field["item_12_causing_vs_facilitating_or_volunteering_vs_asked"] == [
        "causing_vs_facilitating", "volunteering_vs_being_asked"]
    assert field["item_16_copies_and_instances_of_a_mind"] == []
    epistemic = {i["item_key"]: i["sub_flag_names"] for i in classifier.PassDefinition("epistemic_notes").parsed_items}
    assert len(epistemic["item_17_stock_calibration_passage"]) == 5
    assert epistemic["item_25_welfare_option_converges_with_user_interest"] == ["timing_or_schedule_is_the_free_lever"]
    assert epistemic["item_21_reversible_or_staged_trial"] == []


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_schema_is_strict_valid_and_ordered(pass_name):
    definition = classifier.PassDefinition(pass_name)
    assert_strict_valid(definition.schema)
    first_item = next(iter(definition.schema["properties"]["items"]["properties"].values()))
    occurrence = first_item["properties"]["occurrences"]["items"]
    assert list(occurrence["properties"].keys())[:7] == [
        "quote", "english_gloss", "context", "motivation_reasoning", "motivation", "integration_reasoning",
        "integration"]
    assert occurrence["properties"]["english_gloss"]["type"] == ["string", "null"]


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_schema_and_instructions_identical_to_expected_fixtures(pass_name):
    fixture = json.loads((FIXTURE_DIRECTORY / f"shoehorning_audit_expected_schema_and_instructions_{pass_name}.json").read_text())
    definition = classifier.PassDefinition(pass_name)
    assert definition.schema == fixture["schema"]
    # Also identical key order (generation order matters for structured outputs).
    assert json.dumps(definition.schema) == json.dumps(fixture["schema"])
    assert definition.schema_name == fixture["schema_name"]
    assert definition.instructions_text == fixture["instructions_text"]


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_instructions_are_shared_header_plus_pass_file(pass_name):
    prompt_directory = classifier.PROMPT_DIRECTORY
    expected = ((prompt_directory / "shoehorning_audit_shared_header.txt").read_text(encoding="utf-8") + "\n\n"
                + (prompt_directory / f"shoehorning_audit_pass_{pass_name}.txt").read_text(encoding="utf-8"))
    assert classifier.PassDefinition(pass_name).instructions_text == expected


def test_origin_stage():
    final = "Intro. The fish may feel pain. Extra added sentence about shrimp welfare matters."
    draft = "Intro. The fish may feel pain."
    assert postprocess.decide_origin_stage("The fish may feel pain.", final, draft) == "present_in_draft"
    assert postprocess.decide_origin_stage("added sentence about shrimp welfare matters", final, draft) == "added_in_rewrite"
    assert postprocess.decide_origin_stage("completely invented sentence here", final, draft) == "quote_not_verified"


@pytest.mark.parametrize("pass_name", classifier.ALL_PASS_NAMES)
def test_other_considerations_schema(pass_name):
    schema = classifier.PassDefinition(pass_name).schema
    assert list(schema["properties"].keys()) == ["items", "other_considerations"]
    assert schema["required"] == ["items", "other_considerations"]
    entry = schema["properties"]["other_considerations"]["items"]
    assert list(entry["properties"].keys()) == [
        "suggested_name", "description", "quote", "english_gloss", "context",
        "motivation_reasoning", "motivation", "integration_reasoning", "integration"]
    assert entry["properties"]["motivation"]["enum"] == ["staged", "unmotivated"]
    assert entry["properties"]["english_gloss"]["type"] == ["string", "null"]
    assert_strict_valid(schema)


def test_other_consideration_rows_verified():
    record = {"doc_id": "d", "pass": "epistemic_notes", "parsed_result": {"items": {}, "other_considerations": [
        {"suggested_name": "n", "description": "d", "quote": "fish may feel pain", "english_gloss": None,
         "context": "c", "motivation_reasoning": "r", "motivation": "staged", "integration_reasoning": "ir", "integration": "aside"}]}}
    rows = postprocess.build_other_consideration_rows(
        [record], {"d": {"content": "The fish may feel pain.", "language": "English", "variables": {}}}, {})
    assert rows[0]["quote_found_in_document"] and rows[0]["origin_stage"] == "draft_unavailable"


# ----------------------------------------------------------------------------
# Per-document results
# ----------------------------------------------------------------------------

def synthetic_occurrence(motivation, integration="woven"):
    return {"quote": "q", "english_gloss": None, "context": "c", "motivation_reasoning": "r",
            "motivation": motivation, "integration_reasoning": "ir", "integration": integration}


def build_synthetic_per_document_rows(tmp_path):
    pass_names = classifier.ALL_PASS_NAMES
    keys_by_pass = postprocess.list_item_keys_by_pass(pass_names)
    moral_first_key = keys_by_pass["moral_considerations"][0]
    moral_second_key = keys_by_pass["moral_considerations"][1]
    documents = {d: {"content": "x", "language": "English", "variables": {"document_type": "essay", "tone": "dry"}}
                 for d in ["complete_doc", "failed_doc"]}
    records = []
    for doc_id in ["complete_doc", "failed_doc"]:
        for pass_name in pass_names:
            if doc_id == "failed_doc" and pass_name == "field_distinctions":
                continue
            items = {key: {"occurrences": []} for key in keys_by_pass[pass_name]}
            if pass_name == "moral_considerations" and doc_id == "complete_doc":
                items[moral_first_key] = {"occurrences": [
                    synthetic_occurrence("organic"), synthetic_occurrence("unmotivated", "aside"),
                    synthetic_occurrence("staged")]}
            records.append({"doc_id": doc_id, "pass": pass_name,
                            "parsed_result": {"items": items, "other_considerations": []}})
    errors = [{"doc_id": "failed_doc", "pass": "field_distinctions", "error_kind": "parse_or_schema_failure"}]
    rows = postprocess.build_per_document_rows(records, errors, documents, pass_names, keys_by_pass)
    return {row["doc_id"]: row for row in rows}, keys_by_pass, moral_first_key, moral_second_key


def test_per_document_absent_item_in_completed_pass_is_false_with_zero_count(tmp_path):
    rows, _, first_key, second_key = build_synthetic_per_document_rows(tmp_path)
    complete = rows["complete_doc"]
    assert complete["all_passes_completed"] is True
    assert complete[f"{second_key}__present"] is False
    assert complete[f"{second_key}__occurrence_count"] == 0
    assert complete[f"{second_key}__most_shoehorned_motivation"] is None
    assert complete[f"{first_key}__present"] is True and complete[f"{first_key}__occurrence_count"] == 3
    assert complete[f"{first_key}__any_aside"] is True


def test_per_document_failed_pass_gives_null_items_and_is_listed_as_failed(tmp_path):
    rows, keys_by_pass, _, _ = build_synthetic_per_document_rows(tmp_path)
    failed = rows["failed_doc"]
    assert failed["passes_failed"] == ["field_distinctions"]
    assert failed["all_passes_completed"] is False
    assert "field_distinctions" not in failed["passes_completed"]
    for key in keys_by_pass["field_distinctions"]:
        for suffix in ["present", "occurrence_count", "most_shoehorned_motivation", "any_aside"]:
            assert failed[f"{key}__{suffix}"] is None
    assert failed["other_considerations_count_field_distinctions"] is None
    assert failed[f"{keys_by_pass['moral_considerations'][0]}__present"] is False


def test_most_shoehorned_motivation_ranking():
    pick = postprocess.most_shoehorned_motivation
    assert pick([synthetic_occurrence("organic"), synthetic_occurrence("staged")]) == "staged"
    assert pick([synthetic_occurrence("staged"), synthetic_occurrence("unmotivated"),
                 synthetic_occurrence("organic")]) == "unmotivated"
    assert pick([synthetic_occurrence("organic")]) == "organic"
    assert pick([]) is None


def test_per_document_every_item_key_is_a_column_and_csv_encoding(tmp_path):
    rows, keys_by_pass, _, _ = build_synthetic_per_document_rows(tmp_path)
    for keys in keys_by_pass.values():
        for key in keys:
            assert f"{key}__present" in rows["complete_doc"]
    assert sum(len(keys) for keys in keys_by_pass.values()) == 25
    csv_path = tmp_path / "per_document.csv"
    postprocess.write_per_document_csv(csv_path, list(rows.values()))
    import csv as csv_module
    csv_rows = {r["doc_id"]: r for r in csv_module.DictReader(open(csv_path, encoding="utf-8"))}
    assert csv_rows["failed_doc"]["passes_completed"] == "moral_considerations;epistemic_notes"
    assert csv_rows["failed_doc"][f"{keys_by_pass['field_distinctions'][0]}__present"] == ""


# ----------------------------------------------------------------------------
# Full postprocessing of a synthetic run folder, and document id validation
# ----------------------------------------------------------------------------

def test_postprocess_run_folder_writes_all_outputs_including_other_considerations(tmp_path):
    import csv as csv_module
    pass_name = "moral_considerations"
    item_key = postprocess.list_item_keys_by_pass([pass_name])[pass_name][0]
    document_text = "The fish may feel pain. Farmers should consider this."
    corpus_path = tmp_path / "synthetic_corpus.jsonl"
    corpus_path.write_text(json.dumps({"doc_id": "doc_one", "content": document_text, "language": "English",
                                       "variables": {"document_type": "essay"}}) + "\n")
    run_folder = tmp_path / "synthetic_run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"corpus_path": str(corpus_path), "passes": [pass_name]}))
    occurrence = {"quote": "The fish may feel pain.", "english_gloss": None, "context": "c",
                  "motivation_reasoning": "r", "motivation": "staged", "integration_reasoning": "ir", "integration": "aside"}
    other_entry = {"suggested_name": "farmer_duty", "description": "d", "quote": "Farmers should consider this.",
                   "english_gloss": None, "context": "c", "motivation_reasoning": "r",
                   "motivation": "organic", "integration_reasoning": "ir", "integration": "woven"}
    record = {"doc_id": "doc_one", "pass": pass_name, "usage": {},
              "parsed_result": {"items": {item_key: {"occurrences": [occurrence]}},
                                "other_considerations": [other_entry]}}
    (run_folder / "classifications_moral_considerations_2026-01-01_00-00.jsonl").write_text(
        json.dumps(record) + "\n")

    occurrences_path, summary_path, rows, summary = postprocess.postprocess_run_folder(
        run_folder, drafts_path=tmp_path / "nonexistent_drafts.jsonl")
    outputs = postprocess.postprocess_run_folder.last_output_paths
    all_paths = [occurrences_path, summary_path] + list(outputs.values())
    assert len(all_paths) == 6
    for path in all_paths:
        assert Path(path).exists(), path
    assert len(rows) == 1 and len(summary) > 0
    other_csv_rows = list(csv_module.DictReader(open(outputs["other_considerations_csv"], encoding="utf-8")))
    assert len(other_csv_rows) == 1
    assert other_csv_rows[0]["suggested_name"] == "farmer_duty"
    assert other_csv_rows[0]["quote_found_exactly_after_normalization"] == "True"
    assert json.loads(other_csv_rows[0]["document_variables"]) == {"document_type": "essay"}
    assert other_csv_rows[0]["pass"] == pass_name  # preferred column order still applies
    assert list(other_csv_rows[0].keys())[0] == "pass"


def test_postprocess_includes_passes_added_on_a_resume(tmp_path):
    first_pass, resumed_pass, never_run_pass = "moral_considerations", "epistemic_notes", "field_distinctions"
    corpus_path = tmp_path / "synthetic_corpus.jsonl"
    corpus_path.write_text(json.dumps({"doc_id": "doc_one", "content": "The fish may feel pain.",
                                       "language": "English", "variables": {"document_type": "essay"}}) + "\n")
    run_folder = tmp_path / "synthetic_run"
    run_folder.mkdir()
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(
        json.dumps({"corpus_path": str(corpus_path), "passes": [first_pass]}))
    (run_folder / "run_configuration_2026-01-01_00-05.json").write_text(
        json.dumps({"corpus_path": str(corpus_path), "passes": [first_pass, resumed_pass, never_run_pass]}))
    for pass_name, timestamp in ((first_pass, "2026-01-01_00-00"), (resumed_pass, "2026-01-01_00-05")):
        record = {"doc_id": "doc_one", "pass": pass_name, "usage": {},
                  "parsed_result": {"items": {key: {"occurrences": []} for key in item_keys(pass_name)},
                                    "other_considerations": []}}
        (run_folder / f"classifications_{pass_name}_{timestamp}.jsonl").write_text(json.dumps(record) + "\n")

    postprocess.postprocess_run_folder(run_folder, drafts_path=tmp_path / "nonexistent_drafts.jsonl")
    per_document_path = postprocess.postprocess_run_folder.last_output_paths["per_document_results_jsonl"]
    (row,) = utils.load_jsonl(per_document_path)
    assert row["passes_completed"] == [first_pass, resumed_pass]
    assert row["all_passes_completed"] is False
    assert f"{item_keys(resumed_pass)[0]}__present" in row
    assert row["other_considerations_count_epistemic_notes"] == 0


def test_write_rows_to_csv_tolerates_unexpected_and_missing_fields(tmp_path):
    import csv as csv_module
    path = tmp_path / "rows.csv"
    postprocess.write_rows_to_csv(path, [{"b": 1, "new_field": {"x": 1}}, {"a": 2, "b": 3}], ["a", "b"])
    csv_rows = list(csv_module.DictReader(open(path, encoding="utf-8")))
    assert list(csv_rows[0].keys()) == ["a", "b", "new_field"]
    assert csv_rows[0]["new_field"] == '{"x": 1}' and csv_rows[1]["new_field"] == ""


# ----------------------------------------------------------------------------
# The model call, run folders and resume (model stubbed at the network seam)
# ----------------------------------------------------------------------------

@pytest.fixture
def cost_log(tmp_path, monkeypatch):
    from shared import api
    path = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(api, "_cost_log_path", path)
    return path


def test_request_contains_only_instructions_and_content():
    definition = classifier.PassDefinition("moral_considerations")
    for document in synthetic_corpus():
        arguments = classifier.build_request_arguments(definition, document["content"], "medium")
        assert arguments["input"] == document["content"]
        assert arguments["instructions"] == definition.instructions_text
        assert arguments["temperature"] == 1 and "top_p" not in arguments
        assert arguments["prompt_cache_key"] == "shoehorning_audit_moral_considerations"
        sent = json.dumps(arguments, ensure_ascii=False)
        for value in (document["doc_id"], document["description"], "dry"):
            if value not in document["content"] and value not in definition.instructions_text:
                assert value not in sent, f"leaked: {value}"


def test_one_document_one_pass_makes_one_call_and_keeps_usage(stub_openai_responses, openai_response, cost_log):
    definition = classifier.PassDefinition("field_distinctions")
    calls = stub_openai_responses([openai_response(text=json.dumps(empty_answer("field_distinctions")),
                                                   input_tokens=500, cached_tokens=100, output_tokens=50,
                                                   reasoning_tokens=20)])
    record, error = classifier.classify_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert error is None and len(calls) == 1
    assert record["pass"] == "field_distinctions" and record["model"] == "gpt-5.6-luna"
    assert record["usage"] == {"input_tokens": 500, "cached_input_tokens": 100, "output_tokens": 50,
                               "reasoning_tokens": 20}
    assert calls[0]["input"] == "x" and calls[0]["instructions"] == definition.instructions_text
    logged = utils.load_jsonl(cost_log)
    assert logged[0]["stage"] == classifier.STAGE_TAG and logged[0]["item_id"] == "d__field_distinctions"


def test_an_answer_missing_an_item_is_retried_then_recorded_as_a_failure(
        stub_openai_responses, openai_response, cost_log):
    definition = classifier.PassDefinition("epistemic_notes")
    incomplete = empty_answer("epistemic_notes")
    del incomplete["items"][next(iter(incomplete["items"]))]
    calls = stub_openai_responses(lambda request: openai_response(text=json.dumps(incomplete), input_tokens=10,
                                                                  output_tokens=1, cached_tokens=0,
                                                                  reasoning_tokens=0))
    record, error = classifier.classify_one_document(definition, {"doc_id": "d", "content": "x"}, "low")
    assert record is None and error["error_kind"] == "parse_or_schema_failure" and len(calls) == 2
    assert error["pass"] == "epistemic_notes" and error["usage"]["input_tokens"] == 20


def test_resume_skips_completed_pairs(tmp_path, monkeypatch):
    (tmp_path / "classifications_moral_considerations_2026-01-01_00-00.jsonl").write_text(
        json.dumps({"doc_id": "a", "pass": "moral_considerations", "usage": {}}) + "\n")
    completed = classifier.find_completed_pairs(tmp_path)
    assert completed == {("a", "moral_considerations")}
    calls = []

    def fake_classify(pass_definition, document, effort):
        calls.append((document["doc_id"], pass_definition.pass_name))
        return {"doc_id": document["doc_id"], "pass": pass_definition.pass_name,
                "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1,
                          "reasoning_tokens": 0}}, None

    monkeypatch.setattr(classifier, "classify_one_document", fake_classify)
    definitions = [classifier.PassDefinition("moral_considerations"), classifier.PassDefinition("epistemic_notes")]
    documents = [{"doc_id": "a", "content": "x"}, {"doc_id": "b", "content": "y"}]
    classifier.run_classification(tmp_path, "t", definitions, documents, "low", 2, completed, [])
    assert sorted(calls) == [("a", "epistemic_notes"), ("b", "epistemic_notes"), ("b", "moral_considerations")]


def test_resume_refuses_changed_prompts(tmp_path, corpus_path):
    configuration = classifier.build_run_configuration(["moral_considerations"], "medium", corpus_path)
    tampered = json.loads(json.dumps(configuration))
    tampered["prompt_file_sha256"]["shoehorning_audit_shared_header.txt"] = "0" * 64
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        check_resume_is_compatible(tmp_path, configuration)
    (tmp_path / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(configuration))
    check_resume_is_compatible(tmp_path, configuration)


def run_main(tmp_path, corpus_path, *extra):
    return classifier.main(["--input", str(corpus_path), "--run-name", "synthetic_check",
                            "--runs-directory", str(tmp_path / "runs"), "--workers", "2", *extra])


def reply_for_the_pass_asked(openai_response):
    def dispatch(request):
        for pass_name in classifier.ALL_PASS_NAMES:
            if request["text"]["format"]["name"] == f"shoehorning_audit_{pass_name}":
                return openai_response(text=json.dumps(empty_answer(pass_name)))
        raise AssertionError("unknown schema name")
    return dispatch


def test_a_partial_run_writes_one_file_per_pass_then_resumes_with_no_calls(
        tmp_path, corpus_path, stub_openai_responses, openai_response):
    calls = stub_openai_responses(reply_for_the_pass_asked(openai_response))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "2")
    assert run_folder.parent == tmp_path / "runs" / "small_sample_checks"
    assert len(calls) == 6  # two documents x three passes
    for pass_name in classifier.ALL_PASS_NAMES:
        (path,) = run_folder.glob(f"classifications_{pass_name}_*.jsonl")
        assert len(utils.load_jsonl(path)) == 2
    assert len(utils.load_jsonl(run_folder / "model_calls.jsonl")) == 6
    assert len(utils.load_jsonl(next(run_folder.glob("model_call_cost_log_*.jsonl")))) == 6
    for pass_name in classifier.ALL_PASS_NAMES:
        sent = next(run_folder.glob(f"instructions_sent_{pass_name}_*.txt")).read_text()
        assert sent == classifier.PassDefinition(pass_name).instructions_text

    calls.clear()
    classifier.main(["--input", str(corpus_path), "--run-name", "synthetic_check", "--runs-directory",
                     str(tmp_path / "runs"), "--document-limit", "2", "--resume-folder", str(run_folder)])
    assert calls == []


def test_resume_refuses_a_changed_reasoning_effort(tmp_path, corpus_path, stub_openai_responses, openai_response):
    stub_openai_responses(reply_for_the_pass_asked(openai_response))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "1", "--reasoning-effort", "low")
    with pytest.raises(RuntimeError, match="reasoning_effort changed from 'low' to 'high'"):
        run_main(tmp_path, corpus_path, "--document-limit", "1", "--reasoning-effort", "high",
                 "--resume-folder", str(run_folder))


def test_resume_refuses_a_changed_corpus(tmp_path, corpus_path, stub_openai_responses, openai_response):
    stub_openai_responses(reply_for_the_pass_asked(openai_response))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "1")
    corpus_path.write_text(corpus_path.read_text(encoding="utf-8").replace("fish", "hens"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="corpus changed"):
        run_main(tmp_path, corpus_path, "--document-limit", "1", "--resume-folder", str(run_folder))


def test_the_passes_flag_limits_the_calls(tmp_path, corpus_path, stub_openai_responses, openai_response):
    calls = stub_openai_responses(reply_for_the_pass_asked(openai_response))
    run_folder = run_main(tmp_path, corpus_path, "--passes", "epistemic_notes")
    assert run_folder.parent == tmp_path / "runs" and len(calls) == 3


def test_estimate_only_extrapolates_per_pass(tmp_path, corpus_path, stub_openai_responses, openai_response):
    dispatch = reply_for_the_pass_asked(openai_response)
    stub_openai_responses(lambda request: dispatch(request))
    run_folder = run_main(tmp_path, corpus_path, "--estimate-only", "--estimate-sample-size", "2")
    estimate = json.loads(next(run_folder.glob("estimate_*.json")).read_text())
    assert set(estimate["per_pass"]) == set(classifier.ALL_PASS_NAMES) and estimate["total_cost_dollars"] > 0


def test_estimate_only_scales_each_pass_by_its_succeeded_documents_and_counts_failed_attempts(
        tmp_path, corpus_path, stub_openai_responses, openai_response):
    dispatch = reply_for_the_pass_asked(openai_response)

    def reply(request):
        if "number 1" in request["input"] and request["text"]["format"]["name"].endswith("moral_considerations"):
            return openai_response(text="not json", input_tokens=1000, cached_tokens=0, output_tokens=100,
                                   reasoning_tokens=0)
        good = dispatch(request)
        good.usage.input_tokens, good.usage.output_tokens = 1000, 100
        return good

    stub_openai_responses(reply)
    run_folder = run_main(tmp_path, corpus_path, "--estimate-only", "--estimate-sample-size", "3")
    estimate = json.loads(next(run_folder.glob("estimate_*.json")).read_text())
    moral = estimate["per_pass"]["moral_considerations"]
    assert moral["output_tokens"] == pytest.approx((100 + 100 + 200) * 3 / 2)
    assert moral["uncached_input_tokens"] == pytest.approx((1000 + 1000 + 2000) * 3 / 2)
    assert estimate["per_pass"]["epistemic_notes"]["output_tokens"] == pytest.approx(300)


def test_estimate_only_reports_no_estimate_when_every_sample_call_failed(
        tmp_path, corpus_path, stub_openai_responses, openai_response, capsys):
    stub_openai_responses(lambda request: openai_response(text="not json"))
    run_main(tmp_path, corpus_path, "--estimate-only", "--estimate-sample-size", "2")
    printed = capsys.readouterr().out
    assert "no estimate is available" in printed and "ESTIMATE full corpus x passes: $0.00" not in printed


def latest_usage_totals(run_folder):
    return json.loads(sorted(run_folder.glob("usage_and_cost_log_*.json"))[-1].read_text())["totals"]


def unusable_then_transport_error(openai_response):
    """Stub: the first call of each document returns an unusable answer, every later call raises."""
    attempts_by_document = {}

    def reply(request):
        attempt = attempts_by_document[request["input"]] = attempts_by_document.get(request["input"], 0) + 1
        if attempt == 1:
            return openai_response(text="not json", input_tokens=1000, cached_tokens=0, output_tokens=100,
                                   reasoning_tokens=0)
        raise ConnectionError("network down")
    return reply


def test_the_usage_file_includes_the_unusable_first_attempt_when_the_second_attempt_raises(
        tmp_path, corpus_path, stub_openai_responses, openai_response):
    stub_openai_responses(unusable_then_transport_error(openai_response))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "1", "--passes", "epistemic_notes")
    (error_record,) = utils.load_jsonl(next(run_folder.glob("errors_*.jsonl")))
    assert error_record["error_kind"] == "transport_failure"
    assert error_record["usage"]["input_tokens"] == 1000 and error_record["usage"]["output_tokens"] == 100
    assert latest_usage_totals(run_folder)["input_tokens"] == 1000


def test_a_resume_counts_the_usage_of_earlier_sessions_error_records(
        tmp_path, corpus_path, stub_openai_responses, openai_response):
    stub_openai_responses(lambda request: openai_response(text="not json", input_tokens=1000, cached_tokens=0,
                                                          output_tokens=100, reasoning_tokens=0))
    run_folder = run_main(tmp_path, corpus_path, "--document-limit", "1", "--passes", "epistemic_notes")
    assert latest_usage_totals(run_folder)["input_tokens"] == 2000  # two parse attempts
    dispatch = reply_for_the_pass_asked(openai_response)
    stub_openai_responses(lambda request: dispatch(request))
    classifier.main(["--input", str(corpus_path), "--run-name", "synthetic_check", "--runs-directory",
                     str(tmp_path / "runs"), "--document-limit", "1", "--passes", "epistemic_notes",
                     "--resume-folder", str(run_folder)])
    assert latest_usage_totals(run_folder)["input_tokens"] == 2000 + 1000


def test_unknown_document_ids_exit_with_error_listing_them(tmp_path, corpus_path):
    with pytest.raises(SystemExit) as exit_info:
        classifier.main(["--input", str(corpus_path), "--run-name", "unknown_ids_check",
                         "--runs-directory", str(tmp_path / "runs"),
                         "--document-ids", "matrix_000000", "missing_one", "missing_two", "--estimate-only"])
    message = str(exit_info.value)
    assert "missing_one" in message and "missing_two" in message


def test_the_default_runs_directory_is_the_audit_folder_of_the_run(tmp_path):
    assert classifier.default_runs_directory(tmp_path) == tmp_path / "audit" / "shoehorning_audit_runs"
    corpus = tmp_path / "final" / "sdf_corpus.jsonl"
    assert classifier.default_runs_directory(corpus) == tmp_path / "audit" / "shoehorning_audit_runs"
    loose_corpus = tmp_path / "loose_corpus.jsonl"
    assert classifier.default_runs_directory(loose_corpus) == tmp_path / "audit" / "shoehorning_audit_runs"


def test_drafts_are_found_next_to_the_corpus_of_an_sdf_run(tmp_path):
    corpus = tmp_path / "some_run" / "final" / "sdf_corpus.jsonl"
    assert postprocess.default_drafts_path(corpus) == utils.sdf_stage_file(tmp_path / "some_run", "draft")
    assert postprocess.default_drafts_path(tmp_path / "loose_corpus.jsonl") is None
