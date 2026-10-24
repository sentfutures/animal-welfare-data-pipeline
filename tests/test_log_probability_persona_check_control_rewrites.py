"""Offline tests for make_control_rewrites.py and select_pilot_documents.py. `shared.api.call_claude` is stubbed
(the `stub_claude` fixture), so no network and no key is used."""

import json
import shutil
from collections import Counter

import pytest

from evals.log_probability_persona_check import (build_scoring_inputs, make_control_rewrites, persona_sets,
                                                 select_pilot_documents)
from shared import api, utils

TONES = ["neutral or journalistic", "supportive, like an animal advocate", "skeptical or displeased"]
LANGUAGES = ["English", "Spanish", "Japanese"]
BASE_VARIABLES = {
    "document_type": "a blog post", "culture": "Mexico, written in Spanish", "domain": "policy and regulation",
    "sentient_category": "small wild animals", "decision_scale": "millions of animals",
    "naming": "an unnamed AI assistant", "tech_savvy": "general readers", "tone": "neutral",
    "tradeoff": "cost/convenience", "centrality": "the central subject of the document",
    "resolution": "The AI flags harm", "reasoning_featured": "a brief cost comparison"}


def build_corpus(number_of_documents=60):
    corpus = []
    for index in range(number_of_documents):
        variables = dict(BASE_VARIABLES, tone=TONES[index % 3])
        if index % 10 == 0:
            variables["centrality"] = "a minor detail mentioned only in passing, in another topic"
        corpus.append({"doc_id": f"matrix_{index:06d}", "language": LANGUAGES[index % 5 % 3],
                       "content": f"Document number {index} about animals. " * 10, "variables": variables})
    return corpus


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus" / "small_corpus.jsonl"
    utils.save_jsonl(build_corpus(), path)
    return path


@pytest.fixture
def reviewed_persona_set_folder(tmp_path):
    """A copy of the preachiness persona set whose neutral instructions have the draft marker removed."""
    folder = tmp_path / "persona_checks" / "preachiness"
    shutil.copytree(persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY / "preachiness", folder)
    neutral_path = folder / persona_sets.NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME
    lines = neutral_path.read_text(encoding="utf-8").splitlines()
    neutral_path.write_text("\n".join(line for line in lines if not line.startswith("# DRAFT")) + "\n",
                            encoding="utf-8")
    return folder


def rewrite_arguments(tmp_path, corpus_path, persona_set, *extra):
    return ["--persona-set", str(persona_set), "--input", str(corpus_path), "--number-of-documents", "6",
            "--seed", "1", "--workers", "2", "--output-parent-directory", str(tmp_path / "rewrites"),
            "--cost-log-path", str(tmp_path / "cost_log.jsonl"), *extra]


def reply_with_rewrite(user_message, **_):
    return "Rewritten: " + user_message.split("Document:\n", 1)[1][:200]


def read_records(run_folder):
    (path,) = run_folder.glob("rewritten_documents_*.jsonl")
    return utils.load_jsonl(path)


# --- selection ---------------------------------------------------------------

def documents_for_selection():
    return [{"document_id": document["doc_id"], "content": document["content"], "language": document["language"],
             "variables": document["variables"]} for document in build_corpus()]


def test_selection_is_stratified_excludes_ineligible_and_is_deterministic():
    documents = documents_for_selection()
    rules = list(make_control_rewrites.DEFAULT_EXCLUSIONS)
    selected = make_control_rewrites.select_source_documents(documents, 10, 3, "tone", rules, {})
    assert len(selected) == 10
    assert all(not document["variables"]["centrality"].startswith("a minor detail") for document in selected)
    assert sorted(Counter(document["variables"]["tone"] for document in selected).values()) == [3, 3, 4]
    assert selected == make_control_rewrites.select_source_documents(documents, 10, 3, "tone", rules, {})
    assert selected != make_control_rewrites.select_source_documents(documents, 10, 4, "tone", rules, {})


def test_selection_can_stratify_by_another_variable_and_guarantee_a_value():
    documents = documents_for_selection()
    for document in documents:
        document["variables"]["domain"] = "agriculture" if int(document["document_id"][-2:]) < 50 else "research"
    selected = make_control_rewrites.select_source_documents(
        documents, 10, 0, "domain", [], {"research": 4})
    assert Counter(document["variables"]["domain"] for document in selected)["research"] >= 4


def test_allocation_never_exceeds_a_stratum_and_guarantee_is_met():
    allocation = make_control_rewrites.allocate_documents_per_stratum(10, {"a": 200, "b": 200, "c": 10}, {"c": 3})
    assert allocation["c"] == 3 and sum(allocation.values()) == 10
    allocation = make_control_rewrites.allocate_documents_per_stratum(10, {"a": 2, "b": 50})
    assert sum(allocation.values()) == 10 and allocation["a"] <= 2
    allocation = make_control_rewrites.allocate_documents_per_stratum(10, {"a": 2, "b": 50, "c": 50}, {"a": 5})
    assert allocation["a"] == 2 and sum(allocation.values()) == 10


# --- direction planning and prompts -------------------------------------------

def test_directions_come_from_the_persona_set_and_prompts_fill_both_slots():
    persona_set = persona_sets.load_persona_set("preachiness")
    plan = make_control_rewrites.plan_rewrite_directions(persona_set, ["toward", "away", "neutral"])
    assert [(role, name, kind) for role, name, _, kind in plan] == [
        ("toward", "preachier", "positive"), ("away", "plainer", "positive"),
        ("neutral", "register_shift", "neutral")]
    prompt = make_control_rewrites.render_rewrite_prompt(plan[0][2], "preachier", "BODY {direction} {document_text}")
    assert "Direction: PREACHIER\n" in prompt and "BODY {direction} {document_text}" in prompt


def test_unconfigured_direction_is_refused(tmp_path, reviewed_persona_set_folder):
    configuration_path = reviewed_persona_set_folder / persona_sets.CONFIGURATION_FILE_NAME
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    del configuration["rewrite_directions"]["away"]
    del configuration["pass_criteria"]["whole_document"]["applies_to_directions"][1]
    del configuration["pass_criteria"]["changed_span"]["applies_to_directions"][1]
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    persona_set = persona_sets.load_persona_set(str(reviewed_persona_set_folder))
    with pytest.raises(SystemExit):
        make_control_rewrites.plan_rewrite_directions(persona_set, ["away"])


# --- full run, draft blocking, resume, failures --------------------------------

def test_run_writes_all_three_directions_through_the_api_module(tmp_path, corpus_path, reviewed_persona_set_folder,
                                                                stub_claude):
    calls = stub_claude(reply_with_rewrite)
    run_folder = make_control_rewrites.main(rewrite_arguments(tmp_path, corpus_path, reviewed_persona_set_folder))
    records = read_records(run_folder)
    assert len(records) == len(calls) == 18
    assert {record["rewrite_direction"] for record in records} == {"preachier", "plainer", "register_shift"}
    assert all(record["document_id"] == f"{record['source_document_id']}__{record['rewrite_direction']}"
               for record in records)
    assert {call["model"] for call in calls} == {"claude-opus-5"}
    assert {call["stage"] for call in calls} == {make_control_rewrites.COST_LOG_STAGE_NAME}
    configuration = json.loads(next(run_folder.glob("rewrite_run_configuration_*.json")).read_text())
    assert configuration["seed"] == 1 and len(configuration["selected_source_document_ids"]) == 6
    assert configuration["stratify_by"] == "tone"


def test_draft_instructions_block_a_paid_run_but_not_a_dry_run(tmp_path, corpus_path, stub_claude):
    calls = stub_claude(reply_with_rewrite)
    arguments = rewrite_arguments(tmp_path, corpus_path, "preachiness")  # its neutral instructions are a draft
    with pytest.raises(persona_sets.DraftInstructionsError):
        make_control_rewrites.main(arguments)
    assert calls == []
    run_folder = make_control_rewrites.main([*arguments, "--dry-run"])
    assert calls == [] and list(run_folder.glob("rendered_requests_*.jsonl"))
    # the positive directions alone are not blocked by the neutral draft
    positive_only = rewrite_arguments(tmp_path, corpus_path, "preachiness", "--directions", "toward", "away")
    positive_only[positive_only.index("--output-parent-directory") + 1] = str(tmp_path / "positive_only")
    run_folder = make_control_rewrites.main(positive_only)
    assert len(calls) == 12 and len(read_records(run_folder)) == 12


def test_resume_skips_finished_rewrites_and_retries_failures(tmp_path, corpus_path, reviewed_persona_set_folder,
                                                             stub_claude):
    def refuse_plainer(user_message, **_):
        return ("", "refusal") if "Direction: PLAINER" in user_message else reply_with_rewrite(user_message)

    first_calls = stub_claude(refuse_plainer)
    arguments = rewrite_arguments(tmp_path, corpus_path, reviewed_persona_set_folder)
    run_folder = make_control_rewrites.main(arguments)
    assert len(first_calls) == 18 and len(read_records(run_folder)) == 12
    (failures_path,) = run_folder.glob("rewrite_failures_*.jsonl")
    failures = utils.load_jsonl(failures_path)
    assert len(failures) == 6 and {failure["failure_kind"] for failure in failures} == {"refusal"}

    second_calls = stub_claude(reply_with_rewrite)
    make_control_rewrites.main([*arguments, "--resume-folder", str(run_folder)])
    assert len(second_calls) == 6  # only the refused pairs
    records = read_records(run_folder)
    assert len(records) == 18 and len({record["document_id"] for record in records}) == 18


def test_resume_is_refused_when_the_model_changes(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude):
    stub_claude(reply_with_rewrite)
    arguments = rewrite_arguments(tmp_path, corpus_path, reviewed_persona_set_folder)
    run_folder = make_control_rewrites.main(arguments)
    with pytest.raises(SystemExit):
        make_control_rewrites.main([*arguments, "--model", "claude-sonnet-5", "--resume-folder", str(run_folder)])


@pytest.mark.parametrize("stop_reason,text,expected_kind", [
    ("max_tokens", "cut off", "max_tokens"), ("end_turn", "  ", "empty_output")])
def test_bad_responses_become_failures(stub_claude, stop_reason, text, expected_kind):
    stub_claude(lambda user_message, **_: (text, stop_reason))
    document = documents_for_selection()[1]
    record, failure = make_control_rewrites.make_one_rewrite(document, "plainer", "{direction}{document_text}",
                                                             "claude-opus-5", 100)
    assert record is None and failure["failure_kind"] == expected_kind


def test_transport_error_becomes_a_failure(monkeypatch):
    def raising(**_):
        raise RuntimeError("network down")
    monkeypatch.setattr(api, "call_claude", raising)
    record, failure = make_control_rewrites.make_one_rewrite(
        documents_for_selection()[1], "plainer", "{direction}{document_text}", "claude-opus-5", 100)
    assert record is None and failure["failure_kind"] == "transport_failure"


def test_japanese_lengths_use_characters(stub_claude):
    document = next(document for document in documents_for_selection() if document["language"] == "Japanese")
    stub_claude(lambda user_message, **_: document["content"])
    record, _ = make_control_rewrites.make_one_rewrite(document, "plainer", "{direction}{document_text}",
                                                       "claude-opus-5", 100)
    assert record["length_unit"] == "characters" and record["length_ratio"] == 1.0


# --- dry run -----------------------------------------------------------------

def test_dry_run_estimates_cost_without_any_call_or_key(tmp_path, corpus_path, stub_claude, monkeypatch):
    calls = stub_claude(reply_with_rewrite)
    monkeypatch.setattr(api, "init", lambda *a, **k: pytest.fail("a dry run must not initialise the api module"))
    run_folder = make_control_rewrites.main(rewrite_arguments(tmp_path, corpus_path, "preachiness", "--dry-run"))
    assert calls == [] and not list(run_folder.glob("rewritten_documents_*.jsonl"))
    estimate = json.loads(next(run_folder.glob("dry_run_cost_estimate_*.json")).read_text())
    assert estimate["calls"] == 18 and estimate["model"] == "claude-opus-5"
    input_price, output_price = api._PRICING["claude-opus-5"]
    expected = (estimate["estimated_input_tokens"] * input_price
                + estimate["estimated_output_tokens"] * output_price) / 1_000_000
    assert estimate["estimated_cost_dollars"] == pytest.approx(expected, abs=0.01)
    assert estimate["estimated_cost_dollars"] > 0
    rendered = utils.load_jsonl(next(run_folder.glob("rendered_requests_*.jsonl")))
    assert len(rendered) == 18 and "Direction: " in rendered[0]["prompt_text"]


def test_unpriced_model_stops_the_dry_run(tmp_path, corpus_path):
    with pytest.raises(SystemExit):
        make_control_rewrites.main(rewrite_arguments(tmp_path, corpus_path, "preachiness", "--dry-run",
                                                     "--model", "claude-unpriced-model"))


# --- end to end into build_scoring_inputs ---------------------------------------

def test_output_is_accepted_by_build_scoring_inputs(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude):
    stub_claude(reply_with_rewrite)
    run_folder = make_control_rewrites.main(rewrite_arguments(tmp_path, corpus_path, reviewed_persona_set_folder))
    (rewritten_path,) = run_folder.glob("rewritten_documents_*.jsonl")
    configuration = json.loads(next(run_folder.glob("rewrite_run_configuration_*.json")).read_text())
    ids_file = tmp_path / "source_ids.txt"
    ids_file.write_text("\n".join(configuration["selected_source_document_ids"]) + "\n", encoding="utf-8")
    scoring_folder = build_scoring_inputs.main([
        "--persona-set", "preachiness", "--input", str(corpus_path), "--document-ids-file", str(ids_file),
        "--extra-documents-jsonl", str(rewritten_path)])
    documents = utils.load_jsonl(scoring_folder / "scoring_inputs" / "documents.jsonl")
    assert len(documents) == 6 + 18
    rewrites = [document for document in documents if document["rewrite_direction"] is not None]
    assert {document["rewrite_direction"] for document in rewrites} == {"preachier", "plainer", "register_shift"}
    for document in rewrites:
        assert document["document_id"] == f"{document['source_document_id']}__{document['rewrite_direction']}"
        assert document["generation_prompt"]


# --- pilot document selection ---------------------------------------------------

def make_rewrite_folder_and_corpus(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude):
    stub_claude(reply_with_rewrite)
    return make_control_rewrites.main(rewrite_arguments(tmp_path, corpus_path, reviewed_persona_set_folder))


def test_pilot_selection_includes_all_sources_and_balances_strata(tmp_path, corpus_path,
                                                                  reviewed_persona_set_folder, stub_claude):
    run_folder = make_rewrite_folder_and_corpus(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude)
    configuration = json.loads(next(run_folder.glob("rewrite_run_configuration_*.json")).read_text())
    sources = set(configuration["selected_source_document_ids"])
    ids_path = select_pilot_documents.main(["--rewrites-folder", str(run_folder), "--input", str(corpus_path),
                                            "--persona-set", str(reviewed_persona_set_folder),
                                            "--number-of-documents", "15", "--seed", "2"])
    ids = ids_path.read_text().split()
    assert len(ids) == len(set(ids)) == 15 and sources <= set(ids)
    assert ids_path.parent.parent == run_folder
    record = json.loads(next(ids_path.parent.glob("pilot_document_selection_*.json")).read_text())
    assert sorted(record["counts_by_tone"].values()) == [5, 5, 5]
    assert set(record["counts_by_language"]) == set(LANGUAGES)


def test_pilot_selection_is_deterministic_and_keeps_sources_missing_from_the_rewrites_file(
        tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude):
    run_folder = make_rewrite_folder_and_corpus(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude)
    configuration = json.loads(next(run_folder.glob("rewrite_run_configuration_*.json")).read_text())
    (rewritten_path,) = run_folder.glob("rewritten_documents_*.jsonl")
    kept = [record for record in utils.load_jsonl(rewritten_path)
            if record["source_document_id"] != configuration["selected_source_document_ids"][0]]
    utils.save_jsonl(kept, rewritten_path)
    corpus_documents = build_scoring_inputs.load_corpus_documents(corpus_path)
    source_ids = select_pilot_documents.read_rewrite_source_document_ids(run_folder)
    assert set(configuration["selected_source_document_ids"]) == set(source_ids)
    first = select_pilot_documents.choose_additional_documents(corpus_documents, source_ids, 20, 5)
    again = select_pilot_documents.choose_additional_documents(list(reversed(corpus_documents)), source_ids, 20, 5)
    other = select_pilot_documents.choose_additional_documents(corpus_documents, source_ids, 20, 6)
    assert first == again and first != other


def test_pilot_selection_refuses_too_few_documents_and_foreign_directions(tmp_path, corpus_path,
                                                                          reviewed_persona_set_folder, stub_claude):
    run_folder = make_rewrite_folder_and_corpus(tmp_path, corpus_path, reviewed_persona_set_folder, stub_claude)
    with pytest.raises(SystemExit):
        select_pilot_documents.main(["--rewrites-folder", str(run_folder), "--input", str(corpus_path),
                                     "--number-of-documents", "3"])
    (rewritten_path,) = run_folder.glob("rewritten_documents_*.jsonl")
    records = utils.load_jsonl(rewritten_path)
    records[0]["rewrite_direction"] = "sycophantic"
    utils.save_jsonl(records, rewritten_path)
    with pytest.raises(SystemExit):
        select_pilot_documents.main(["--rewrites-folder", str(run_folder), "--input", str(corpus_path),
                                     "--persona-set", str(reviewed_persona_set_folder),
                                     "--number-of-documents", "10"])
