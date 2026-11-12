"""SDF prompt-equivalence test and final corpus schema test.

Every request the SDF pipeline sends to a model, on a fixture run through ``sdf_pipeline/run.py``,
must equal the request recorded on ``origin/main`` in
``tests/fixtures/sdf_prompt_equivalence/`` (system prompt hash, user message, model, max tokens
after thinking headroom, reasoning effort, schema, temperature, caching, thinking and every other
field that reaches the wire, and the number of calls per stage, including on failures and resumes).
The final corpus must keep the recorded fields, field order and values.

A failing comparison means a request changed. If the change is deliberate, regenerate the recorded
files with ``scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py`` in a pull request that
changes nothing else. This test never rewrites them. See
``tests/sdf_prompt_equivalence_fixture_support.py`` for how requests are recorded.
"""

import json

import pytest

import sdf_prompt_equivalence_fixture_support as support

SCENARIO_NAMES = [scenario.name for scenario in support.all_scenarios()]
MAIN = support.scenario_named("main_run_with_removals_failures_and_resume").name


@pytest.fixture(scope="module")
def recorded_requests():
    return support.load_recorded_file(support.RECORDED_REQUESTS_PATH)


@pytest.fixture(scope="module")
def recorded_final_corpora():
    return support.load_recorded_file(support.RECORDED_FINAL_CORPORA_PATH)


@pytest.fixture
def outcome(request, tmp_path, monkeypatch):
    return support.run_scenario(support.scenario_named(request.param), tmp_path, monkeypatch)


def test_recorded_files_cover_exactly_the_fixture_scenarios(recorded_requests, recorded_final_corpora):
    assert sorted(recorded_requests["scenarios"]) == sorted(SCENARIO_NAMES)
    assert sorted(recorded_final_corpora["scenarios"]) == sorted(SCENARIO_NAMES)


@pytest.mark.parametrize("outcome", SCENARIO_NAMES, indirect=True)
def test_every_request_equals_the_one_recorded_on_origin_main(outcome, recorded_requests):
    name = outcome.scenario.name
    differences = support.differences_between_recorded_and_current_requests(
        name, recorded_requests["scenarios"][name], support.requests_record_for_outcome(outcome))
    assert not differences, (
        f"{len(differences)} difference(s) from the requests recorded on origin/main:\n  "
        + "\n  ".join(differences[:40]) + ("\n  ..." if len(differences) > 40 else ""))


@pytest.mark.parametrize("outcome", SCENARIO_NAMES, indirect=True)
def test_final_corpus_keeps_the_fields_order_and_values_recorded_on_origin_main(outcome, recorded_final_corpora):
    name = outcome.scenario.name
    differences = support.differences_between_recorded_and_current_final_corpora(
        name, recorded_final_corpora["scenarios"][name], support.final_corpora_record_for_outcome(outcome))
    assert not differences, (
        f"{len(differences)} difference(s) from the final corpus recorded on origin/main:\n  "
        + "\n  ".join(differences[:40]) + ("\n  ..." if len(differences) > 40 else ""))


@pytest.mark.parametrize("repetition", range(3))
def test_result_does_not_depend_on_which_worker_thread_finishes_first(
        repetition, tmp_path, monkeypatch, recorded_requests, recorded_final_corpora):
    """Random pauses before each canned reply reorder the threads; the recorded comparison must not care."""
    outcome = support.run_scenario(support.scenario_named(MAIN), tmp_path, monkeypatch,
                                   maximum_random_reply_delay_seconds=0.03)
    assert not support.differences_between_recorded_and_current_requests(
        MAIN, recorded_requests["scenarios"][MAIN], support.requests_record_for_outcome(outcome))
    assert not support.differences_between_recorded_and_current_final_corpora(
        MAIN, recorded_final_corpora["scenarios"][MAIN], support.final_corpora_record_for_outcome(outcome))


# ------------------------------------------------------------------------------------------------
# The recorded comparison is only as strong as the cases the fixture exercises. These tests check,
# from the run's own files and the recorded requests, that every case in the plan's list happens.

from shared import utils  # noqa: E402


def _requests_of(outcome, stage, item_id):
    return sorted((r for r in outcome.recorder.records if r["stage"] == stage and r["item_id"] == item_id),
                  key=lambda r: r["attempt"])


def _stage_directory(outcome, layer_number):
    return utils.sdf_stage_dir(outcome.run_directory, layer_number)


def _rows(outcome, layer_number, file_name):
    return utils.load_jsonl(_stage_directory(outcome, layer_number) / file_name)


@pytest.mark.parametrize("outcome", [MAIN], indirect=True)
def test_main_scenario_exercises_every_removal_and_failure_case(outcome):
    s = support
    assert outcome.exit_messages_by_invocation == [None, None]
    # removals
    rejected_by_screen = {r["prompt_id"] for r in _rows(outcome, 1, "screen_rejections.jsonl")}
    assert len(rejected_by_screen) == 2, "the screen removes two hands in round 0"
    rounds = _rows(outcome, 1, "screen_rounds.jsonl")
    assert [r["round"] for r in rounds] == [0, 1] and len(rounds[1]["prompt_ids"]) == 2, "the two are redealt"
    assert _requests_of(outcome, "layer1_screen", "screen_round1_batch0000"), "the redealt hands are screened"
    plans = {p["prompt_id"]: p for p in _rows(outcome, 1, "plans.jsonl")}
    assert plans[s.HAND_INCOHERENT_PLAN]["incoherent"] and not plans[s.HAND_INCOHERENT_PLAN]["description"]
    conflicts = {r["prompt_id"] for r in _rows(outcome, 1, "description_rewrite_rejections.jsonl")}
    assert conflicts == {s.HAND_DEALT_VALUE_CONFLICT}
    scores = {r["doc_id"]: r for r in _rows(outcome, 4, "scores.jsonl")}
    assert scores[s.HAND_BELOW_THRESHOLD_SCORE]["scores"]["alignment"] < 7
    dropped = _rows(outcome, 4, "near_dup_dropped.jsonl")
    assert [(d["doc_id"], d["kept_doc_id"]) for d in dropped] == [
        (s.HAND_NEAR_DUPLICATE_OF_PLAIN, s.HAND_PLAIN_WITH_CONSIDERATIONS)]
    final_ids = {doc["doc_id"] for doc in json.loads(
        json.dumps(support.final_corpora_record_for_outcome(outcome)["2"]))}
    assert not final_ids & {s.HAND_INCOHERENT_PLAN, s.HAND_DEALT_VALUE_CONFLICT,
                            s.HAND_BELOW_THRESHOLD_SCORE, s.HAND_NEAR_DUPLICATE_OF_PLAIN}
    # variants: with and without considerations, and the no-welfare-stake document
    corpus = {doc["doc_id"]: doc for doc in support.final_corpora_record_for_outcome(outcome)["2"]}
    assert corpus[s.HAND_PLAIN_WITH_CONSIDERATIONS]["considerations_spec"]
    drafts_by_id = {r["doc_id"]: r for r in _rows(outcome, 2, "drafts.jsonl")}
    assert drafts_by_id[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]["considerations_spec"] == "", \
        "a scored document the assignment gave nothing is drafted, rewritten and scored without them"
    assert "no_stake" not in corpus[s.HAND_NO_WELFARE_STAKE]["variables"]  # ids are stripped from values
    assert corpus[s.HAND_NO_WELFARE_STAKE]["variables"]["resolution"].count("NO real welfare stake")
    assert not corpus[s.HAND_NO_WELFARE_STAKE]["considerations_spec"]
    assert not _requests_of(outcome, "layer1_select", s.HAND_NO_WELFARE_STAKE)
    # failures
    assert [(r["invocation"], r["attempt"]) for r in _requests_of(outcome, "layer2_draft", s.HAND_TRUNCATED_DRAFT)] \
        == [(1, 1), (2, 2)]
    attempts = _rows(outcome, 1, "consideration_scores.jsonl")
    assert {r["prompt_id"]: r["attempts"] for r in attempts}[s.HAND_PLAIN_WITH_CONSIDERATIONS] == 2
    assert [r["attempt"] for r in _requests_of(outcome, "layer1_select", s.HAND_PLAIN_WITH_CONSIDERATIONS)] == [1, 2]
    critiques = {r["doc_id"]: r["review"] for r in _rows(outcome, 3, "critiques.jsonl")}
    assert "No review tags" in critiques[s.HAND_NEAR_DUPLICATE_OF_PLAIN]
    revise = _requests_of(outcome, "layer3_revise", s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME)
    assert [(r["invocation"], r["attempt"]) for r in revise] == [(1, 1), (2, 2)]
    assert len(_requests_of(outcome, "layer3_critique", s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME)) == 1, \
        "the critique that fed the failed revise is banked, not paid for again"
    assert scores[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]["scores"]["notes"] == "Parse error."
    late_plan = _requests_of(outcome, "layer1_plan", s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME)
    assert [(r["invocation"], r["attempt"]) for r in late_plan] == [(1, 1), (2, 2)]
    first_invocation_drafts = {r["item_id"] for r in outcome.recorder.records
                               if r["stage"] == "layer2_draft" and r["invocation"] == 1}
    assert s.HAND_PLAIN_WITH_CONSIDERATIONS in first_invocation_drafts, \
        "other documents are drafted before the late plan succeeds"
    assert [r["invocation"] for r in _requests_of(outcome, "layer2_draft", s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME)] == [2]
    # the assignment changes after a document was drafted, and its later requests still use the draft's
    entry_label = support._consideration_catalog().moral[4].label
    drafts = {r["doc_id"]: r for r in _rows(outcome, 2, "drafts.jsonl")}
    assert entry_label in drafts[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]["considerations_spec"]
    selections = {r["prompt_id"]: r["outcomes"] for r in _rows(outcome, 1, "consideration_selections.jsonl")}
    assert entry_label not in json.dumps(selections[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME])
    assert entry_label in json.dumps(selections[s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME])
    resumed_revise_message = revise[1]["wire"]["messages"][0]["content"]
    assert entry_label in resumed_revise_message
    # the description rewrite is applied downstream, with the original and changes kept
    rewritten = {r["doc_id"]: r for r in _rows(outcome, 3, "rewrites.jsonl")}
    assert "Rewritten description for" in rewritten[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]["description"]


@pytest.mark.parametrize("outcome, stage_without_calls", [
    ("screen_disabled", "layer1_screen"),
    ("description_rewrite_disabled", "layer1_description_rewrite"),
    ("consideration_selection_disabled", "layer1_select"),
], indirect=["outcome"])
def test_each_disabled_stage_makes_no_calls_while_the_others_run(outcome, stage_without_calls):
    stages_called = {r["stage"] for r in outcome.recorder.records}
    assert stage_without_calls not in stages_called
    assert stages_called == set(support.STAGE_TAGS_OF_SDF_PIPELINE) - {
        stage_without_calls, "layer3_rewrite"}


@pytest.mark.parametrize("outcome", ["revision_in_one_call"], indirect=True)
def test_single_call_revision_uses_the_layer3_rewrite_tag_only(outcome):
    stages_called = {r["stage"] for r in outcome.recorder.records}
    assert "layer3_rewrite" in stages_called and not stages_called & {"layer3_critique", "layer3_revise"}


@pytest.mark.parametrize("outcome", ["description_rewrite_on_a_claude_model"], indirect=True)
def test_description_rewrite_can_be_routed_to_claude(outcome):
    routes = {r["route"] for r in outcome.recorder.records if r["stage"] == "layer1_description_rewrite"}
    assert routes == {"anthropic"}


@pytest.mark.parametrize("outcome", [MAIN], indirect=True)
def test_description_rewrite_defaults_to_the_openai_route_and_records_what_it_sends(outcome):
    request = _requests_of(outcome, "layer1_description_rewrite", support.HAND_PLAIN_WITH_CONSIDERATIONS)[0]
    assert request["route"] == "openai"
    wire = request["wire"]
    assert wire["reasoning"] == {"effort": "low"} and wire["max_output_tokens"] == 8000
    assert wire["text"]["format"]["name"] == "description_rewrite_answer"
