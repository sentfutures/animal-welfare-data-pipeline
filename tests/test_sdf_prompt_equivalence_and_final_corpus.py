"""SDF prompt-equivalence test, stage-file test and final corpus schema test.

Every request the SDF pipeline sends to a model, on the fixture scenarios run through
``sdf_pipeline/run.py``, must equal the request recorded on ``origin/main`` in
``tests/fixtures/sdf_prompt_equivalence/`` (system prompt hash, user message, model, max tokens
after thinking headroom, reasoning effort, schema and its key order, temperature, caching, thinking
and every other field that reaches the wire, and the number of calls per stage, including on
failures, stops, interrupted runs and resumes). The stage files, checkpoints and final corpus must
keep their recorded fields, field order and values.

A failing comparison means a request or a record changed. If that is deliberate, regenerate the
recorded files with ``scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py`` in the same
pull request as the change, and commit them with it. A pull request that only refactors never
rewrites them. This test never rewrites them. See
``tests/sdf_prompt_equivalence_fixture_support.py`` for how requests are recorded and
``tests/sdf_prompt_equivalence_scenarios.py`` for the scenarios.
"""

import json
import tempfile
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios

SCENARIO_NAMES = [scenario.name for scenario in scenarios.all_scenarios()]
MAIN = scenarios.MAIN_SCENARIO_NAME


@pytest.fixture(scope="module")
def outcome_of():
    """Runs each scenario once per module, on first use, and keeps the outcome."""
    cache = {}
    with tempfile.TemporaryDirectory() as base_directory:
        def get(name):
            if name not in cache:
                cache[name] = support.run_scenario_in_isolation(
                    scenarios.scenario_named(name), Path(base_directory) / name)
            return cache[name]
        yield get


def _failure_message(kind, differences):
    return (f"{len(differences)} difference(s) from the {kind} recorded on origin/main:\n  "
            + "\n  ".join(differences[:40]) + ("\n  ..." if len(differences) > 40 else "")
            + f"\n{support.REGENERATION_RULE}")


def test_recorded_files_cover_exactly_the_fixture_scenarios():
    recorded = sorted(path.name for path in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"))
    assert recorded == sorted(f"recorded_{name}_on_origin_main.json" for name in SCENARIO_NAMES)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_request_equals_the_one_recorded_on_origin_main(name, outcome_of):
    differences = support.differences_between_recorded_and_current_requests(
        name, support.load_recorded_scenario(name), support.record_for_outcome(outcome_of(name)))
    assert not differences, _failure_message("requests", differences)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_stage_files_and_checkpoints_keep_the_fields_order_and_values_recorded_on_origin_main(name, outcome_of):
    differences = support.differences_between_recorded_and_current_stage_files(
        name, support.load_recorded_scenario(name), support.record_for_outcome(outcome_of(name)))
    assert not differences, _failure_message("stage files", differences)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_final_corpus_keeps_the_fields_order_and_values_recorded_on_origin_main(name, outcome_of):
    differences = support.differences_between_recorded_and_current_final_corpora(
        name, support.load_recorded_scenario(name), support.record_for_outcome(outcome_of(name)))
    assert not differences, _failure_message("final corpus", differences)


@pytest.mark.parametrize("repetition", range(2))
def test_result_does_not_depend_on_which_worker_thread_finishes_first(repetition, tmp_path):
    """Random pauses before each canned reply reorder the threads; the comparison must not care."""
    outcome = support.run_scenario_in_isolation(
        scenarios.scenario_named(MAIN), tmp_path, maximum_random_reply_delay_seconds=0.03)
    recorded, current = support.load_recorded_scenario(MAIN), support.record_for_outcome(outcome)
    assert not support.differences_between_recorded_and_current_requests(MAIN, recorded, current)
    assert not support.differences_between_recorded_and_current_stage_files(MAIN, recorded, current)
    assert not support.differences_between_recorded_and_current_final_corpora(MAIN, recorded, current)


# ------------------------------------------------------------------------------------------------
# The recorded comparison is only as strong as the cases the fixture exercises. These tests check,
# from the run's own files and the recorded requests, that every case really happens.

def _requests_of(outcome, stage, item_id):
    return sorted((r for r in outcome.recorder.records if r["stage"] == stage and r["item_id"] == item_id),
                  key=lambda r: r["attempt"])


def _calls_in_invocation(outcome, invocation, stage):
    return sum(1 for r in outcome.recorder.records if r["invocation"] == invocation and r["stage"] == stage)


def test_main_scenario_exercises_every_removal_and_failure_case(outcome_of):
    outcome, s = outcome_of(MAIN), scenarios
    assert outcome.exit_messages_by_invocation == [None, None]
    rejected_by_screen = {r["prompt_id"] for r in support.stage_rows(outcome.run_directory, "screen_rejections.jsonl")}
    assert len(rejected_by_screen) == 2, "the screen removes two hands in round 0"
    rounds = support.stage_rows(outcome.run_directory, "screen_rounds.jsonl")
    assert [r["round"] for r in rounds] == [0, 1] and len(rounds[1]["prompt_ids"]) == 2, "the two are redealt"
    assert _requests_of(outcome, "layer1_screen", "screen_round1_batch0000"), "the redealt hands are screened"
    plans = {p["prompt_id"]: p for p in support.stage_rows(outcome.run_directory, "plans.jsonl")}
    assert plans[s.HAND_INCOHERENT_PLAN]["incoherent"] and not plans[s.HAND_INCOHERENT_PLAN]["description"]
    assert plans[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]["description"].startswith("Planned description (old heading form)")
    conflicts = {r["prompt_id"] for r in support.stage_rows(outcome.run_directory, "description_rewrite_rejections.jsonl")}
    assert conflicts == {s.HAND_DEALT_VALUE_CONFLICT}
    scores = {r["doc_id"]: r for r in support.stage_rows(outcome.run_directory, "scores.jsonl")}
    assert scores[s.HAND_BELOW_THRESHOLD_SCORE]["scores"]["alignment"] < 7
    dropped = support.stage_rows(outcome.run_directory, "near_dup_dropped.jsonl")
    assert [(d["doc_id"], d["kept_doc_id"]) for d in dropped] == [
        (s.HAND_NEAR_DUPLICATE_OF_PLAIN, s.HAND_PLAIN_WITH_CONSIDERATIONS)]
    corpus = {doc["doc_id"]: doc for doc in json.loads(json.dumps(
        [json.loads(line) for line in outcome.final_corpus_lines_after_each_invocation[1]]))}
    assert not set(corpus) & {s.HAND_INCOHERENT_PLAN, s.HAND_DEALT_VALUE_CONFLICT,
                              s.HAND_BELOW_THRESHOLD_SCORE, s.HAND_NEAR_DUPLICATE_OF_PLAIN}
    drafts_by_id = {r["doc_id"]: r for r in support.stage_rows(outcome.run_directory, "drafts.jsonl")}
    assert corpus[s.HAND_PLAIN_WITH_CONSIDERATIONS]["considerations_spec"].count("\n") == 3, \
        "four considerations: two moral and two epistemic"
    assert drafts_by_id[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]["considerations_spec"] == "", \
        "a scored document the assignment gave nothing is drafted, rewritten and scored without them"
    assert corpus[s.HAND_NO_WELFARE_STAKE]["variables"]["resolution"].count("NO real welfare stake")
    assert not corpus[s.HAND_NO_WELFARE_STAKE]["considerations_spec"]
    assert not _requests_of(outcome, "layer1_select", s.HAND_NO_WELFARE_STAKE)
    assert [(r["invocation"], r["attempt"]) for r in _requests_of(outcome, "layer2_draft", s.HAND_TRUNCATED_DRAFT)] \
        == [(1, 1), (2, 2)]
    attempts = {r["prompt_id"]: r["attempts"] for r in support.stage_rows(outcome.run_directory, "consideration_scores.jsonl")}
    assert attempts[s.HAND_PLAIN_WITH_CONSIDERATIONS] == 2
    critiques = {r["doc_id"]: r["review"] for r in support.stage_rows(outcome.run_directory, "critiques.jsonl")}
    assert "No review tags" in critiques[s.HAND_NEAR_DUPLICATE_OF_PLAIN]
    revise = _requests_of(outcome, "layer3_revise", s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME)
    assert [(r["invocation"], r["attempt"]) for r in revise] == [(1, 1), (2, 2)]
    assert len(_requests_of(outcome, "layer3_critique", s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME)) == 1
    assert scores[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]["scores"]["notes"] == "Parse error."
    late_plan = _requests_of(outcome, "layer1_plan", s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME)
    assert [(r["invocation"], r["attempt"]) for r in late_plan] == [(1, 1), (2, 2)]
    assert any(r["stage"] == "layer2_draft" and r["invocation"] == 1 and r["item_id"] == s.HAND_PLAIN_WITH_CONSIDERATIONS
               for r in outcome.recorder.records), "other documents are drafted before the late plan succeeds"
    # the assignment changes after a document was drafted, and its later requests still use the draft's
    catalog = support._consideration_catalog()
    entry_label = catalog.moral[4].label
    assert entry_label in drafts_by_id[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]["considerations_spec"]
    selections = {r["prompt_id"]: r["outcomes"] for r in support.stage_rows(outcome.run_directory, "consideration_selections.jsonl")}
    assert entry_label not in json.dumps(selections[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME])
    assert entry_label in json.dumps(selections[s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME])
    assert entry_label in revise[1]["wire"]["messages"][0]["content"]
    # two documents have a rewritten description, one of them never redealt and never failing
    rewritten = {r["doc_id"]: r for r in support.stage_rows(outcome.run_directory, "rewrites.jsonl")}
    for hand in (s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME, s.HAND_PLAIN_WITH_CONSIDERATIONS):
        assert rewritten[hand]["description"].startswith("Rewritten description for")
    # ties between documents for one entry exist (the tie-break is by hash of the seed)
    tied_scores = [r for r in support.stage_rows(outcome.run_directory, "consideration_scores.jsonl")]
    assert sum(1 for r in tied_scores if r["scores"]["moral"][catalog.moral[6].label]["score"] == 5) == 2


def test_the_real_config_scenario_leaves_every_stage_model_unset_as_config_yaml_does(outcome_of):
    outcome = outcome_of("real_config_unchanged_except_the_prompt_count")
    real = __import__("yaml").safe_load((support.REPOSITORY_ROOT / "config.yaml").read_text())
    used = outcome.configs_used[0]
    changed = {key for key in used["sdf"] if used["sdf"][key] != real["sdf"].get(key)}
    assert changed == {"n_prompts"} and {k: v for k, v in used.items() if k != "sdf"} == {
        k: v for k, v in real.items() if k != "sdf"}
    models = {r["stage"]: r["wire"]["model"] for r in outcome.recorder.records if r["route"] == "anthropic"}
    assert set(models.values()) == {real["model"]}


def test_each_failure_scenario_reaches_the_stops_and_retries_it_is_named_for(outcome_of):
    """Spot checks that the injected failures happened where the scenario says (the recorded files
    hold the full call counts; the stop messages are asserted when each scenario runs)."""
    screen = outcome_of("failures_in_the_screen_across_resumes")
    assert [_calls_in_invocation(screen, n, "layer1_screen") for n in (1, 2, 3)] == [3, 2, 1]
    assert _calls_in_invocation(screen, 1, "layer1_plan") == 0, "a failed batch stops the run before planning"
    rewrite = outcome_of("failures_in_the_description_rewrite_across_resumes")
    assert _calls_in_invocation(rewrite, 1, "layer1_select") == 0, "held-out descriptions stop the run after the step"
    select = outcome_of("failures_in_consideration_scoring_across_resumes")
    assert [r["attempt"] for r in _requests_of(select, "layer1_select", scenarios._hand(1))] == [1, 2, 3, 4]
    assert _calls_in_invocation(select, 1, "layer2_draft") == 0, "the run stops before the assignment"
    critique = outcome_of("failures_in_the_critique_and_the_revise_across_resumes")
    assert _calls_in_invocation(critique, 1, "layer3_revise") == 0 and _calls_in_invocation(critique, 1, "layer4_score") == 0
    draft = outcome_of("unusable_draft_replies_are_not_a_systemic_stop")
    assert draft.exit_messages_by_invocation == [None, None]
    for name, stage, later_stage in (("every_plan_call_fails", "layer1_plan", "layer1_description_rewrite"),
                                     ("every_draft_call_fails", "layer2_draft", "layer3_critique"),
                                     ("every_critique_call_fails", "layer3_critique", "layer4_score"),
                                     ("every_score_call_fails", "layer4_score", None)):
        stopped = outcome_of(name)
        assert _calls_in_invocation(stopped, 1, stage) == 2
        assert later_stage is None or _calls_in_invocation(stopped, 1, later_stage) == 0


def test_interrupted_runs_reuse_saved_state(outcome_of):
    screen = outcome_of("interrupted_screen_rounds_resume_with_a_different_batch_size")
    batches = support.stage_rows(screen.run_directory, "screen_batches.jsonl")
    assert [len(b["prompt_ids"]) for b in batches if b["round"] == 0] == [3, 3], "saved batches reused, not regrouped"
    assert len(screen.recorder.records) and _calls_in_invocation(screen, 4, "layer1_screen") == 0, \
        "the accepted hands win over a lost screen checkpoint"
    assert [r["round"] for r in support.stage_rows(screen.run_directory, "screen_rounds.jsonl")] == [0, 1]
    records = outcome_of("interrupted_run_records_and_checkpoint_marks_disagree")
    assert not [r for r in records.recorder.records if r["invocation"] == 2], \
        "a record without a mark and a mark without a record both count as done"


def test_layer_restarts_reload_the_earlier_stages(outcome_of):
    restart = outcome_of("layer_restarts_reload_earlier_stages_from_files")
    assert [_calls_in_invocation(restart, n, "layer2_draft") for n in (1, 2, 3, 4, 5)] == [4, 4, 0, 0, 3]
    assert [_calls_in_invocation(restart, n, "layer3_critique") for n in (1, 2, 3, 4, 5)] == [4, 4, 4, 0, 3]
    assert [_calls_in_invocation(restart, n, "layer4_score") for n in (1, 2, 3, 4, 5)] == [4, 4, 4, 4, 3]
    assert all(_calls_in_invocation(restart, n, "layer1_plan") == 0 for n in (2, 3, 4, 5))
    rewrites = {r["doc_id"]: r for r in restart.stage_files_after_each_invocation[2]["layer3/rewrites.jsonl"]}
    assert "revision" in rewrites[scenarios._hand(0)], "the explicit revision record is carried through"


def test_gate_and_cull_boundaries_are_on_both_sides_of_each_threshold(outcome_of):
    outcome = outcome_of("gate_and_near_duplicate_boundaries")
    final = [json.loads(line)["doc_id"] for line in outcome.final_corpus_lines_after_each_invocation[0]]
    assert final == [scenarios._hand(n) for n in (0, 1, 2, 5, 7)]
    dropped = support.stage_rows(outcome.run_directory, "near_dup_dropped.jsonl")
    assert [d["doc_id"] for d in dropped] == [scenarios._hand(6)] and 0.9 <= dropped[0]["similarity"] < 0.99


def test_a_screen_removal_that_cannot_be_redealt_drops_the_cards(outcome_of):
    outcome = outcome_of("screen_removal_that_cannot_be_redealt")
    rounds = support.stage_rows(outcome.run_directory, "screen_rounds.jsonl")
    assert rounds[-1]["prompt_ids"] == [] and "could not be redealt" in rounds[-1]["note"]


def test_code_defaults_are_used_when_the_config_names_nothing(outcome_of):
    only_plan_model = outcome_of("minimal_config_with_only_a_plan_model")
    models = {r["stage"]: r["wire"]["model"] for r in only_plan_model.recorder.records}
    assert models["layer1_screen"] == models["layer1_select"] == models["layer1_description_rewrite"] == "claude-haiku-4-5"
    assert models["layer2_draft"] == models["layer3_critique"] == models["layer4_score"] == "claude-sonnet-5"
    assert [r["round"] for r in support.stage_rows(only_plan_model.run_directory, "screen_rounds.jsonl")] == [0, 1, 2, 3], \
        "the default is three redeal rounds"
    assert [b["prompt_ids"] and len(b["prompt_ids"]) for b in support.stage_rows(only_plan_model.run_directory, "screen_batches.jsonl")
            if b["round"] == 0] == [10, 2], "the default screen batch size is 10"
    assert {r["prompt_id"]: r["attempts"] for r in support.stage_rows(only_plan_model.run_directory, "consideration_scores.jsonl")
            }[scenarios._hand(0)] == 3, "the default is three scoring attempts"
    openai_and_adaptive = outcome_of("minimal_config_with_openai_rewrite_and_adaptive_thinking_models")
    rewrite = next(r for r in openai_and_adaptive.recorder.records if r["stage"] == "layer1_description_rewrite")
    assert rewrite["route"] == "openai" and rewrite["wire"]["reasoning"] == {"effort": "medium"}
    draft = next(r for r in openai_and_adaptive.recorder.records if r["stage"] == "layer2_draft")
    assert draft["wire"]["max_tokens"] == 6000 + 8000, "the default thinking headroom"
    assert not [r for r in openai_and_adaptive.recorder.records if r["stage"] == "layer1_select" and
                "NO real welfare stake" in r["wire"]["messages"][0]["content"]]


@pytest.mark.parametrize("name", ["temperature_kept_on_every_stage_with_a_two_call_revision",
                                  "temperature_kept_on_every_stage_with_a_one_call_revision"])
def test_every_claude_request_carries_the_global_temperature(name, outcome_of):
    claude_requests = [r for r in outcome_of(name).recorder.records if r["route"] == "anthropic"]
    assert claude_requests and {r["wire"].get("temperature") for r in claude_requests} == {0.5}
    assert {r["stage"] for r in claude_requests} >= {"layer1_description_rewrite", "layer1_select", "layer2_draft"}


@pytest.mark.parametrize("name, stage_without_calls", [
    ("screen_disabled", "layer1_screen"),
    ("description_rewrite_disabled", "layer1_description_rewrite"),
    ("consideration_selection_disabled", "layer1_select"),
])
def test_each_disabled_stage_makes_no_calls_while_the_others_run(name, stage_without_calls, outcome_of):
    stages_called = {r["stage"] for r in outcome_of(name).recorder.records}
    assert stage_without_calls not in stages_called
    assert stages_called == set(support.STAGE_TAGS_OF_SDF_PIPELINE) - {stage_without_calls, "layer3_rewrite"}


def test_single_call_revision_uses_the_layer3_rewrite_tag_only(outcome_of):
    stages_called = {r["stage"] for r in outcome_of("revision_in_one_call").recorder.records}
    assert "layer3_rewrite" in stages_called and not stages_called & {"layer3_critique", "layer3_revise"}


def test_description_rewrite_routes(outcome_of):
    claude = {r["route"] for r in outcome_of("description_rewrite_on_a_claude_model").recorder.records
              if r["stage"] == "layer1_description_rewrite"}
    assert claude == {"anthropic"}
    request = _requests_of(outcome_of(MAIN), "layer1_description_rewrite", scenarios.HAND_PLAIN_WITH_CONSIDERATIONS)[0]
    assert request["route"] == "openai"
    assert request["wire"]["reasoning"] == {"effort": "low"} and request["wire"]["max_output_tokens"] == 8000
    assert request["wire"]["text"]["format"]["name"] == "description_rewrite_answer"


def test_deal_repair_scenario_repairs_five_no_stake_hands(outcome_of):
    outcome = outcome_of("deal_repair_of_five_no_stake_hands_with_every_plan_call_failing")
    hands = support.stage_rows(outcome.run_directory, "screen_hands.jsonl")
    no_stake = [h for h in hands if "NO real welfare stake" in h["variables"]["resolution"]]
    assert len(hands) == 25 and len(no_stake) == 5
    assert not [h for h in no_stake if h["variables"]["centrality"].startswith(("the central subject", "not the central"))], \
        "the deal rule moved every no-stake hand off the welfare centralities"
    assert _calls_in_invocation(outcome, 1, "layer1_plan") == 25 and _calls_in_invocation(outcome, 1, "layer2_draft") == 0


def test_resume_over_padded_plan_descriptions_sends_them_stripped_to_the_rewrite_only(outcome_of):
    outcome = outcome_of("resume_over_plans_with_padded_descriptions")
    rewrite = next(r for r in outcome.recorder.records if r["stage"] == "layer1_description_rewrite" and r["invocation"] == 2)
    assert "   Planned description" not in rewrite["wire"]["input"] and "Planned description" in rewrite["wire"]["input"]
    draft = next(r for r in outcome.recorder.records if r["stage"] == "layer2_draft" and r["invocation"] == 2)
    assert "   Planned description" in draft["wire"]["messages"][0]["content"], "the draft prompt carries the stored text"
