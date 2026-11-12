"""The document state loader on every run the SDF prompt-equivalence fixture makes.

The scenarios of ``tests/sdf_prompt_equivalence_scenarios.py`` run the current-layout pipeline end to
end with canned replies and cover every removal and failure kind, interrupted runs, resumes, restarts
from a later layer and the disabled stages. After each invocation of each scenario the run folder is
loaded and the states are compared with what the stage files hold, read here directly (not through the
loader), and the final corpus derived from the states is compared with ``final/sdf_corpus.jsonl``.

The loader must also never write into the run folder: every file's bytes are compared before and after.
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline.document_state import (
    DescriptionRewriteVerdict, DescriptionVersionKind, DocumentOutcome, PipelineStep, RemovalReasonCode,
    RunLayout, TextVersionKind)
from sdf_pipeline.document_state_final_corpus import derive_final_corpus_records
from sdf_pipeline.document_state_loader import load_document_states_with_run_notes
from shared import utils

SCENARIO_NAMES = [scenario.name for scenario in scenarios.all_scenarios()]
MAIN = scenarios.MAIN_SCENARIO_NAME

# Scenarios that edit files the way a crash leaves them: their states are expected to carry loader notes.
SCENARIOS_WITH_EXPECTED_LOADER_NOTES = {
    "interrupted_run_records_and_checkpoint_marks_disagree",
    # a described plan whose description rewrite has no stored result is held out of drafting
    "layer_restarts_reload_earlier_stages_from_files",
    "failures_in_the_description_rewrite_across_resumes",
}


def _bytes_of_every_file(run_directory: Path) -> dict:
    return {str(path.relative_to(run_directory)): path.read_bytes()
            for path in sorted(run_directory.rglob("*")) if path.is_file()}


def _stage_rows(run_directory: Path, layer_number: int, file_name: str) -> list:
    return utils.load_jsonl(utils.sdf_stage_dir(run_directory, layer_number) / file_name)


class InvocationSnapshot:
    """What the loader returned after one invocation, with the raw stage files read independently."""

    def __init__(self, invocation_number: int, run_directory: Path):
        self.invocation_number = invocation_number
        before = _bytes_of_every_file(run_directory)
        self.loaded = load_document_states_with_run_notes(run_directory)
        assert _bytes_of_every_file(run_directory) == before, "the loader changed a file in the run folder"
        self.states = self.loaded.states
        self.by_id = {state.document_id: state for state in self.states}
        self.raw = {name: _stage_rows(run_directory, layer, name) for layer, name in (
            (1, "screen_hands.jsonl"), (1, "screen_rounds.jsonl"), (1, "screen_rejections.jsonl"),
            (1, "screen_judgments.jsonl"), (1, "prompts.jsonl"), (1, "plans.jsonl"),
            (1, "description_rewrites.jsonl"), (1, "description_rewrite_rejections.jsonl"),
            (1, "consideration_scores.jsonl"), (1, "consideration_selections.jsonl"),
            (1, "consideration_scoring_failures.jsonl"),
            (2, "drafts.jsonl"), (3, "critiques.jsonl"), (3, "rewrites.jsonl"), (4, "scores.jsonl"),
            (4, "near_dup_dropped.jsonl"))}
        final_path = run_directory / "final" / "sdf_corpus.jsonl"
        self.final_exists = final_path.is_file()
        self.final_records = utils.load_jsonl(final_path)
        self.derived_records = derive_final_corpus_records(self.states)


@pytest.fixture(scope="module")
def snapshots_of():
    """{scenario name: [InvocationSnapshot per invocation]}, each scenario run once, on first use."""
    cache = {}
    with tempfile.TemporaryDirectory() as base_directory:
        def get(name):
            if name not in cache:
                collected = []
                support.run_scenario_in_isolation(
                    scenarios.scenario_named(name), Path(base_directory) / name,
                    after_each_invocation=lambda number, run_directory: collected.append(
                        InvocationSnapshot(number, run_directory)))
                cache[name] = collected
            return cache[name]
        yield get


def _states_with_reason(snapshot, reason_code):
    return {s.document_id for s in snapshot.states if s.removal and s.removal.reason_code is reason_code}


def _last_per_id(rows, key):
    return {row[key]: row for row in rows}


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_states_match_the_stage_files_after_every_invocation(name, snapshots_of):
    for snapshot in snapshots_of(name):
        where = f"{name}, after invocation {snapshot.invocation_number}"
        raw = snapshot.raw
        assert snapshot.loaded.layout is RunLayout.CURRENT
        assert snapshot.loaded.run_notes == [], where
        ids = [s.document_id for s in snapshot.states]
        assert len(ids) == len(set(ids)), f"{where}: a document appears twice"

        # Removals recorded in files
        assert _states_with_reason(snapshot, RemovalReasonCode.EXTREMELY_IMPLAUSIBLE_AT_SCREEN) \
            == {r["prompt_id"] for r in raw["screen_rejections.jsonl"]}, where
        assert _states_with_reason(snapshot, RemovalReasonCode.DECLARED_INCOHERENT_AT_PLAN) \
            == {p["prompt_id"] for p in raw["plans.jsonl"] if p["incoherent"]}, where
        assert _states_with_reason(snapshot, RemovalReasonCode.DEALT_VALUE_CONFLICT_AT_DESCRIPTION_REWRITE) \
            == {r["prompt_id"] for r in raw["description_rewrite_rejections.jsonl"]}, where
        assert not _states_with_reason(snapshot, RemovalReasonCode.ABSENT_FROM_FINAL_CORPUS_WITHOUT_RECORDED_REASON), where
        if snapshot.final_exists:
            culled = {r["doc_id"] for r in raw["near_dup_dropped.jsonl"]}
            assert _states_with_reason(snapshot, RemovalReasonCode.NEAR_DUPLICATE_AT_CULL) <= culled, where

        # Descriptions
        plans = _last_per_id(raw["plans.jsonl"], "prompt_id")
        results = _last_per_id(raw["description_rewrites.jsonl"], "prompt_id")
        for state in snapshot.states:
            plan, result = plans.get(state.document_id), results.get(state.document_id)
            if plan is None or not plan["description"]:
                assert state.description_versions == (), where
            else:
                assert state.description_versions[0].kind is DescriptionVersionKind.PLANNED
                assert state.description_versions[0].text == plan["description"]
            if result is None:
                assert state.description_rewrite_decision is None
            elif result["dealt_value_conflict"]:
                assert state.description_rewrite_decision.verdict is DescriptionRewriteVerdict.DEALT_VALUE_CONFLICT
                assert state.description_rewrite_decision.conflict_reason == result["conflict_reason"]
            elif result["rewrite_needed"] and plan is not None:
                assert state.description_rewrite_decision.verdict is DescriptionRewriteVerdict.REWRITTEN
                rewritten = state.description_versions[1]
                assert rewritten.kind is DescriptionVersionKind.REWRITTEN
                assert rewritten.text == result["rewritten_description"]
                assert list(rewritten.changes) == result["changes"]
                assert state.current_description == result["rewritten_description"]
            else:
                assert state.description_rewrite_decision.verdict is DescriptionRewriteVerdict.KEPT_ORIGINAL
                assert len(state.description_versions) == (1 if plan is not None else 0)

        # Text versions and what each was written from
        drafts = _last_per_id(raw["drafts.jsonl"], "doc_id")
        rewrites = _last_per_id(raw["rewrites.jsonl"], "doc_id")
        critiques = _last_per_id(raw["critiques.jsonl"], "doc_id")
        for state in snapshot.states:
            draft, rewrite = drafts.get(state.document_id), rewrites.get(state.document_id)
            kinds = [v.kind for v in state.text_versions]
            assert kinds == ([TextVersionKind.DRAFT] if draft else []) + ([TextVersionKind.REVISION] if rewrite else []), where
            for version, record in zip(state.text_versions, [r for r in (draft, rewrite) if r]):
                assert version.text == record["content"]
                assert version.written_from.description == record["description"]
                assert version.written_from.variables == record["variables"]
                assert version.written_from.considerations_spec == record["considerations_spec"]
            if rewrite:
                assert state.text_versions[-1].critique == rewrite["review"]
                assert state.banked_critique is None
            elif state.document_id in critiques:
                assert state.banked_critique == critiques[state.document_id]["review"]

        # Deal order
        hand_positions = [(s.deal_round_number, s.hand_index) for s in snapshot.states if s.hand_index is not None]
        assert hand_positions == sorted(hand_positions), where
        hand_ids_in_files = [h["prompt_id"] for h in raw["screen_hands.jsonl"]] or [p["prompt_id"] for p in raw["prompts.jsonl"]]
        assert set(ids) >= set(hand_ids_in_files) - set(_unrecorded_round_ids(raw)), where

        # Loader notes
        if name not in SCENARIOS_WITH_EXPECTED_LOADER_NOTES:
            assert [(s.document_id, s.loader_notes) for s in snapshot.states if s.loader_notes] == [], where


def _unrecorded_round_ids(raw) -> set:
    recorded = {pid for r in raw["screen_rounds.jsonl"] for pid in r["prompt_ids"]}
    return {h["prompt_id"] for h in raw["screen_hands.jsonl"]} - recorded


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_corpus_derived_from_the_states_equals_the_final_corpus_after_every_invocation(name, snapshots_of):
    for snapshot in snapshots_of(name):
        where = f"{name}, after invocation {snapshot.invocation_number}"
        if not snapshot.final_exists:
            assert snapshot.derived_records == [], where
            continue
        derived_by_id = {r["doc_id"]: r for r in snapshot.derived_records}
        final_by_id = {r["doc_id"]: r for r in snapshot.final_records}
        assert list(derived_by_id) == [r["doc_id"] for r in snapshot.derived_records]
        assert set(derived_by_id) == set(final_by_id), where
        for doc_id, record in final_by_id.items():
            assert derived_by_id[doc_id] == record, f"{where}: {doc_id}"
            assert list(derived_by_id[doc_id]) == list(record), f"{where}: field order of {doc_id}"
        # Today's order is deal order for a first invocation; a resume appends what it finished late.
        if snapshot.invocation_number == 1:
            assert list(derived_by_id) == [r["doc_id"] for r in snapshot.final_records], where


def test_outcome_and_reason_code_counts_of_the_main_scenario(snapshots_of):
    first, second = snapshots_of(MAIN)
    for snapshot in (first, second):
        assert len(snapshot.states) == 12
    counts = Counter((s.outcome, s.removal.reason_code if s.removal else None) for s in second.states)
    assert counts == Counter({
        (DocumentOutcome.REMOVED, RemovalReasonCode.EXTREMELY_IMPLAUSIBLE_AT_SCREEN): 2,
        (DocumentOutcome.REMOVED, RemovalReasonCode.DECLARED_INCOHERENT_AT_PLAN): 1,
        (DocumentOutcome.REMOVED, RemovalReasonCode.DEALT_VALUE_CONFLICT_AT_DESCRIPTION_REWRITE): 1,
        (DocumentOutcome.REMOVED, RemovalReasonCode.BELOW_THRESHOLD_AT_GATE): 2,
        (DocumentOutcome.REMOVED, RemovalReasonCode.NEAR_DUPLICATE_AT_CULL): 1,
        (DocumentOutcome.FINAL, None): 5,
    })
    s = scenarios
    assert second.by_id[s.HAND_INCOHERENT_PLAN].furthest_step_reached is PipelineStep.PLAN
    assert second.by_id[s.HAND_DEALT_VALUE_CONFLICT].furthest_step_reached is PipelineStep.DESCRIPTION_REWRITE
    near = second.by_id[s.HAND_NEAR_DUPLICATE_OF_PLAIN].removal
    assert near.evidence["kept_doc_id"] == s.HAND_PLAIN_WITH_CONSIDERATIONS
    assert second.by_id[s.HAND_NO_WELFARE_STAKE].is_final
    assert second.by_id[s.HAND_NO_WELFARE_STAKE].consideration_scoring is None
    for removed_by_screen in _states_with_reason(second, RemovalReasonCode.EXTREMELY_IMPLAUSIBLE_AT_SCREEN):
        removal = second.by_id[removed_by_screen].removal
        assert removal.evidence["round"] == 0 and removal.evidence["cards_were_redealt_into_round"] == 1
        assert second.by_id[removed_by_screen].deal_round_number == 0
    assert {s_.document_id for s_ in second.states if s_.deal_round_number == 1} \
        == {s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME, s.HAND_SCORE_ANSWER_DOES_NOT_PARSE}


def test_main_scenario_documents_whose_state_changed_between_invocations(snapshots_of):
    first, second = snapshots_of(MAIN)
    s = scenarios

    # the plan that failed in invocation 1 is a document with a deal and nothing else until it succeeds
    late_plan_before = first.by_id[s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME]
    assert late_plan_before.plan_reply is None and late_plan_before.outcome is DocumentOutcome.IN_PROGRESS
    assert late_plan_before.furthest_step_reached is PipelineStep.COMPOSE_PLANNING_PROMPT
    assert second.by_id[s.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME].plan_reply

    # the banked critique whose revise failed, then the revision that pairs with it
    banked = first.by_id[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]
    assert banked.banked_critique and len(banked.text_versions) == 1
    assert banked.furthest_step_reached is PipelineStep.CRITIQUE
    revised = second.by_id[s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]
    assert revised.banked_critique is None and len(revised.text_versions) == 2
    assert revised.text_versions[1].critique == banked.banked_critique

    # the document whose assignment changed after it was drafted
    assert banked.considerations_assignment is not None
    assert [c.label for c in banked.considerations_assignment] != [] and not banked.latest_text_was_written_from_a_different_assignment
    assert revised.considerations_assignment == ()
    assert revised.latest_text_was_written_from_a_different_assignment
    assert revised.text_versions[1].written_from.considerations_spec != ""
    assert revised.text_versions[1].written_from.variables["moral_considerations"] != []
    others = [x for x in second.states if x.latest_text_was_written_from_a_different_assignment]
    assert [x.document_id for x in others] == [s.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME]
    assert not any(x.latest_text_was_written_from_a_different_description for x in second.states)

    # the score parse error recorded as 5/5/5
    parse_error = second.by_id[s.HAND_SCORE_ANSWER_DOES_NOT_PARSE]
    assert parse_error.scores.was_parse_error and parse_error.scores.alignment == 5
    assert parse_error.removal.reason_code is RemovalReasonCode.BELOW_THRESHOLD_AT_GATE
    assert parse_error.removal.evidence["score_was_parse_error"] is True
    assert not second.by_id[s.HAND_BELOW_THRESHOLD_SCORE].scores.was_parse_error

    # an unusable first scoring answer is not a recorded failure; the attempts are in the scoring record
    assert second.by_id[s.HAND_PLAIN_WITH_CONSIDERATIONS].consideration_scoring.attempts == 2
    assert second.by_id[s.HAND_PLAIN_WITH_CONSIDERATIONS].last_failure is None

    # a critique with no review tags is kept whole, and a revise reply with no review notes is the first
    # reply of its critique
    assert second.by_id[s.HAND_NEAR_DUPLICATE_OF_PLAIN].text_versions[1].critique.startswith("Problems: the opening")


def test_held_out_description_rewrite_is_in_progress_with_a_note(snapshots_of):
    last = snapshots_of("layer_restarts_reload_earlier_stages_from_files")[-1]
    held_out = last.by_id[scenarios._hand(2)]
    assert held_out.description_rewrite_decision is None and held_out.current_description
    first_failures = snapshots_of("failures_in_the_description_rewrite_across_resumes")[0]
    assert all(s.description_rewrite_decision is None and s.outcome is DocumentOutcome.IN_PROGRESS
               for s in first_failures.states if s.current_description and s.document_id in ("matrix_000000", "matrix_000001"))
    assert held_out.outcome is DocumentOutcome.IN_PROGRESS
    # its earlier considerations rows are still in the files, so that is the furthest step with a record
    assert held_out.furthest_step_reached is PipelineStep.CONSIDERATION_ASSIGNMENT
    assert held_out.text_versions == ()
    assert any("holds it out of drafting" in note for note in held_out.loader_notes)
    # the documents the rewrite did check are final
    assert last.by_id[scenarios._hand(0)].is_final


def test_explicit_revision_record_of_a_seeded_run_is_kept(snapshots_of):
    after_layer_3 = snapshots_of("layer_restarts_reload_earlier_stages_from_files")[2]
    assert after_layer_3.by_id[scenarios._hand(0)].revision_override is not None
    assert after_layer_3.by_id[scenarios._hand(1)].revision_override is None


def test_scoring_failures_are_recorded_and_resolved_after_the_resume(snapshots_of):
    first, second = snapshots_of("failures_in_consideration_scoring_across_resumes")
    failed_before = [s for s in first.states if s.last_failure]
    assert {s.document_id for s in failed_before} == {scenarios._hand(0), scenarios._hand(1)}
    assert all(not s.last_failure.resolved and s.last_failure.step is PipelineStep.CONSIDERATION_SCORING
               for s in failed_before)
    assert all(s.last_failure.resolved for s in second.states if s.last_failure)
    assert first.by_id[scenarios._hand(0)].consideration_scoring is None


def test_interrupted_screen_leaves_hands_in_progress_at_each_stage(snapshots_of):
    first = snapshots_of("interrupted_screen_rounds_resume_with_a_different_batch_size")[0]
    judged = [s for s in first.states if s.screen_judgment is not None]
    unjudged = [s for s in first.states if s.screen_judgment is None]
    assert judged and unjudged
    assert all(s.outcome is DocumentOutcome.IN_PROGRESS for s in unjudged)
    assert all(s.furthest_step_reached is PipelineStep.DEAL for s in unjudged)
    assert all(s.furthest_step_reached is PipelineStep.SCREEN for s in judged if not s.removal)


def test_checkpoint_marks_without_records_are_noted_for_the_interrupted_run(snapshots_of):
    second = snapshots_of("interrupted_run_records_and_checkpoint_marks_disagree")[1]
    notes = [n for s in second.states for n in s.loader_notes]
    assert any("plans.jsonl: marked done" in n for n in notes)
    assert any("drafts.jsonl: marked done" in n for n in notes)
    assert any("rewrites.jsonl: marked done" in n for n in notes)
    assert any("scores.jsonl: marked done" in n for n in notes)
    hand_with_plan_record_lost = second.by_id[scenarios._hand(0)]
    assert any("a full run ignores" in n for n in hand_with_plan_record_lost.loader_notes)
    assert not hand_with_plan_record_lost.is_final


def test_disabled_stages_leave_their_fields_empty_and_still_reach_the_final_corpus(snapshots_of):
    only = snapshots_of("screen_disabled")[0]
    assert all(s.screen_judgment is None and s.deal_round_number == 0 for s in only.states)
    assert [s.document_id for s in only.states] == ["matrix_000000", "matrix_000001", "matrix_000002"]
    only = snapshots_of("description_rewrite_disabled")[0]
    assert all(s.description_rewrite_decision is None and len(s.description_versions) == 1 for s in only.states)
    assert only.derived_records
    only = snapshots_of("consideration_selection_disabled")[0]
    assert all(s.consideration_scoring is None and s.considerations_assignment is None for s in only.states)
    assert not any(s.latest_text_was_written_from_a_different_assignment for s in only.states)
    only = snapshots_of("revision_in_one_call")[0]
    assert all(s.banked_critique is None and s.text_versions[1].critique for s in only.states if s.is_final)


def test_screen_removal_that_cannot_be_redealt_is_dropped_after_the_last_round(snapshots_of):
    only = snapshots_of("screen_removal_that_cannot_be_redealt")[0]
    removed = [s for s in only.states if s.removal]
    assert len(removed) == 1
    assert removed[0].removal.evidence["cards_were_redealt_into_round"] is None
    assert removed[0].removal.evidence["dropped_after_the_last_round"] is True


def test_gate_boundaries_are_inclusive(snapshots_of):
    only = snapshots_of("gate_and_near_duplicate_boundaries")[0]
    reasons = {s.document_id: s.removal.reason_code for s in only.states if s.removal}
    assert reasons[scenarios._hand(3)] is RemovalReasonCode.BELOW_THRESHOLD_AT_GATE
    assert reasons[scenarios._hand(4)] is RemovalReasonCode.BELOW_THRESHOLD_AT_GATE
    assert reasons[scenarios._hand(6)] is RemovalReasonCode.NEAR_DUPLICATE_AT_CULL
    assert only.by_id[scenarios._hand(0)].is_final and only.by_id[scenarios._hand(7)].is_final
