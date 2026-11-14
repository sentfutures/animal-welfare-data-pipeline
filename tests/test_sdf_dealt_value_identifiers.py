"""Dealt value ids: written as ``variable_value_ids`` beside ``variables`` in every stage file record that
copies ``variables`` and in the final corpus, for runs created with this change; never in a request.

A run records them when its manifest says so (decided once, when the run is created). A run created before
that, however often it is resumed, never gains the field. The loader reads the ids from the stage files and
falls back to a lookup in the run's frozen ``inputs/prompts/variables.txt``; it ignores a ``document_events.jsonl`` left in an older run folder.
Nothing that builds a prompt reads the ids.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.layer2_plan_and_description_rewrite import plan as plan_step
from sdf_pipeline import start_a_run_from_an_earlier_runs_step as seeding
from sdf_pipeline.layer1_deal_and_screen import run_layer as screen
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import (
    RECORDS_VALUE_IDS_MANIFEST_KEY, load_frozen_value_ids, value_ids_of_dealt_hand)
from sdf_pipeline.shared_step_machinery.document_state_loader import load_document_states
from shared import utils
from sdf_plan_draft_rewrite_and_score_test_builders import plan_reply
from sdf_screen_test_builders import remove_first_hand_of_round_zero_batches, screen_config  # noqa: F401

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPOSITORY_ROOT / "outputs" / "sdf" / "runs"
VALUE_IDS_BY_AXIS = cp.load_variables(REPOSITORY_ROOT / "prompts" / "sdf" / "variables.txt")[2]
SCREEN_ON_SCENARIO = "temperature_kept_on_every_stage_with_a_two_call_revision"
SCREEN_OFF_SCENARIO = "minimal_config_with_openai_rewrite_and_adaptive_thinking_models"


def ids_that_name_the_dealt_text(hand_variables, ids_of_hand, value_ids_by_axis):
    return all(value_ids_by_axis[axis][ids_of_hand[axis]] == text for axis, text in hand_variables.items())


def test_the_lookup_gives_ids_and_a_note_for_text_nobody_has_or_two_values_share():
    ids, notes = value_ids_of_dealt_hand(
        {"tone": "warm", "mood": "missing", "pace": "same"},
        {"tone": {"t1": "warm"}, "mood": {"m1": "calm"}, "pace": {"p1": "same", "p2": "same"}})
    assert ids == {"tone": "t1"}
    assert len(notes) == 2 and "{mood}" in notes[0] and "{pace}" in notes[1]


def test_the_consideration_lists_of_a_plan_are_not_dealt_values_and_get_neither_an_id_nor_a_note():
    ids, notes = value_ids_of_dealt_hand(
        {"tone": "warm", "moral_considerations": ["a label"], "epistemic_considerations": []},
        {"tone": {"t1": "warm"}})
    assert ids == {"tone": "t1"} and notes == []


# ------------------------------------------------------------------ running whole scenarios

@pytest.fixture(scope="module")
def outcome_of():
    cache = {}
    with tempfile.TemporaryDirectory() as base_directory:
        def get(name, run_started_before_the_field_existed=False):
            key = (name, run_started_before_the_field_existed)
            if key not in cache:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    if run_started_before_the_field_existed:
                        real_create_run_dir = utils.create_run_dir
                        monkeypatch.setattr(
                            utils, "create_run_dir",
                            lambda *a, **k: real_create_run_dir(*a, **{**k, "extra_manifest_fields": None}))
                    cache[key] = support.run_scenario_in_isolation(
                        scenarios.scenario_named(name), Path(base_directory) / f"{name}_{int(run_started_before_the_field_existed)}")
            return cache[key]
        yield get


def final_corpus_records(outcome):
    return [json.loads(line) for line in outcome.final_corpus_lines_after_each_invocation[-1]]


def records_carrying_variables(outcome):
    """[(file, record)] for every stage file record and final corpus record with a ``variables`` field."""
    found = [(file_name, record) for file_name, content in outcome.stage_files_after_each_invocation[-1].items()
             if isinstance(content, list) for record in content if "variables" in record]
    return found + [("final/sdf_corpus.jsonl", record) for record in final_corpus_records(outcome)]


@pytest.mark.parametrize("name", [SCREEN_ON_SCENARIO, SCREEN_OFF_SCENARIO, scenarios.MAIN_SCENARIO_NAME])
def test_a_new_run_writes_ids_beside_variables_in_every_stage_file_and_the_final_corpus(name, outcome_of):
    outcome = outcome_of(name)
    manifest = json.loads((outcome.run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest[RECORDS_VALUE_IDS_MANIFEST_KEY] is True
    records = records_carrying_variables(outcome)
    files = {file_name for file_name, _ in records}
    expected = {"layer2_plan_and_description_rewrite/plans.jsonl", "layer4_draft_critique_and_revise/drafts.jsonl", "layer4_draft_critique_and_revise/rewrites.jsonl", "layer5_score_and_final_selection/scores.jsonl",
                "final/sdf_corpus.jsonl", "layer1_deal_and_screen/screen_hands.jsonl" if name != SCREEN_OFF_SCENARIO else "layer2_plan_and_description_rewrite/prompts.jsonl"}
    assert expected <= files, expected - files
    for file_name, record in records:
        keys = list(record)
        assert keys[keys.index("variables") + 1] == "variable_value_ids", (file_name, keys)
        dealt = {axis: text for axis, text in record["variables"].items()
                 if axis not in ("moral_considerations", "epistemic_considerations")}
        assert set(record["variable_value_ids"]) == set(dealt), file_name
        assert ids_that_name_the_dealt_text(dealt, record["variable_value_ids"], VALUE_IDS_BY_AXIS), file_name


def test_the_screen_on_and_off_paths_are_the_same_deal_and_work_out_the_same_ids(
        screen_config, prompts_sdf, stage_dir, stub_claude, tmp_path):  # noqa: F811
    screen_config["sdf"]["n_prompts"] = 12
    _, _, value_ids = cp.load_variables(prompts_sdf / cp.DEFAULT_VARIABLES.name)
    template = utils.sdf_template_path(prompts_sdf, 1).read_text(encoding="utf-8")
    values, weights, _ = cp.load_variables(prompts_sdf / cp.DEFAULT_VARIABLES.name)
    rules = cp.load_deal_rules(prompts_sdf / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
    screen_hands = [outcome.new_state for outcome in screen.DealStep(
        template, values, weights, 12, screen_config["sdf"].get("seed", 0), rules, value_ids, True).apply([]).values()]
    (stage_dir.parent / "run_manifest.json").write_text(json.dumps({RECORDS_VALUE_IDS_MANIFEST_KEY: True}))
    stub_claude(lambda user_message, **kw: plan_reply())
    plan_step.run(screen_config, prompts_sdf, stage_dir)   # no hands given: the screen is off, the plan layer deals
    composed = utils.load_jsonl(stage_dir / "prompts.jsonl")
    assert [(h["prompt_id"], h["variables"], h["variable_value_ids"]) for h in screen_hands] == [
        (r["prompt_id"], r["variables"], r["variable_value_ids"]) for r in composed]
    assert all(list(r)[:3] == ["prompt_id", "variables", "variable_value_ids"] for r in composed)
    for hand in screen_hands:
        assert ids_that_name_the_dealt_text(hand["variables"], hand["variable_value_ids"], value_ids)


def test_a_run_without_the_manifest_key_gets_the_same_hands_and_no_ids_on_either_path(
        screen_config, prompts_sdf, stage_dir, stub_claude):  # noqa: F811
    screen_config["sdf"]["n_prompts"] = 12
    stub_claude(lambda user_message, **kw: plan_reply())
    plan_step.run(screen_config, prompts_sdf, stage_dir)
    without = utils.load_jsonl(stage_dir / "prompts.jsonl")
    assert all("variable_value_ids" not in r for r in without)
    shutil.rmtree(stage_dir); stage_dir.mkdir()
    (stage_dir.parent / "run_manifest.json").write_text(json.dumps({RECORDS_VALUE_IDS_MANIFEST_KEY: True}))
    plan_step.run(screen_config, prompts_sdf, stage_dir)
    with_ids = utils.load_jsonl(stage_dir / "prompts.jsonl")
    assert [(r["prompt_id"], r["variables"], r["system"], r["prompt"]) for r in without] == [
        (r["prompt_id"], r["variables"], r["system"], r["prompt"]) for r in with_ids]


@pytest.mark.parametrize("name", [SCREEN_ON_SCENARIO, SCREEN_OFF_SCENARIO, scenarios.MAIN_SCENARIO_NAME])
def test_a_run_started_before_the_field_existed_never_gains_it_even_when_resumed(name, outcome_of):
    outcome = outcome_of(name, run_started_before_the_field_existed=True)
    manifest = json.loads((outcome.run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert RECORDS_VALUE_IDS_MANIFEST_KEY not in manifest
    records = records_carrying_variables(outcome)
    assert records and all("variable_value_ids" not in record for _, record in records)
    # and apart from the field the files are those of a new run
    new_run = outcome_of(name)
    stripped = [{k: v for k, v in r.items() if k != "variable_value_ids"} for _, r in records_carrying_variables(new_run)]
    assert stripped == [r for _, r in records]


def test_the_main_scenario_resumes_within_one_run_and_every_record_still_has_ids(outcome_of):
    outcome = outcome_of(scenarios.MAIN_SCENARIO_NAME)
    assert len(outcome.exit_messages_by_invocation) == 2
    assert all("variable_value_ids" in record for _, record in records_carrying_variables(outcome))


# ------------------------------------------------------------------ the loader

def test_the_loader_reads_ids_from_the_stage_files_and_ignores_an_event_log_left_in_an_old_run_folder(outcome_of):
    outcome = outcome_of(SCREEN_ON_SCENARIO)
    run = outcome.run_directory
    event_log = run / "document_events.jsonl"
    assert not event_log.exists()   # the pipeline no longer writes one
    hands = {h["prompt_id"]: h for h in utils.load_jsonl(support.utils.sdf_stage_dir(run, 1) / "screen_hands.jsonl")}
    events = [{"event": "round_dealt", "round": 0, "prompt_ids": list(hands), "note": "original deal",
               "dealt_value_ids": {hand_id: {"tone": "not_a_real_id"} for hand_id in hands}}]
    event_log.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")   # an old run folder's log
    states = load_document_states(run)
    assert states and all(s.dealt_value_ids == hands[s.document_id]["variable_value_ids"] for s in states)
    assert all(s.dealt_value_id_notes == () for s in states)
    event_log.unlink()   # and its absence changes nothing
    assert [s.dealt_value_ids for s in load_document_states(run)] == [s.dealt_value_ids for s in states]


def test_the_loader_falls_back_to_the_frozen_variables_for_stage_files_without_ids(outcome_of, tmp_path):
    outcome = outcome_of(SCREEN_ON_SCENARIO, run_started_before_the_field_existed=True)
    states = load_document_states(outcome.run_directory)
    hands = {h["prompt_id"]: h for h in utils.load_jsonl(utils.sdf_stage_dir(outcome.run_directory, 1) / "screen_hands.jsonl")}
    assert (outcome.run_directory / "inputs" / "prompts" / "variables.txt").is_file()
    for state in states:
        assert ids_that_name_the_dealt_text(hands[state.document_id]["variables"], state.dealt_value_ids, VALUE_IDS_BY_AXIS)
        assert state.dealt_value_id_notes == ()
    # without a frozen variables.txt there is nothing to look in: no ids, one note saying why
    copy = tmp_path / "run_without_snapshot"
    shutil.copytree(outcome.run_directory, copy)
    (copy / "inputs" / "prompts" / "variables.txt").unlink()
    assert all(s.dealt_value_ids == {} and len(s.dealt_value_id_notes) == 1 for s in load_document_states(copy))


def test_a_document_with_no_deal_record_gets_no_notes_about_its_consideration_lists(outcome_of, tmp_path):
    outcome = outcome_of(SCREEN_ON_SCENARIO, run_started_before_the_field_existed=True)
    copy = tmp_path / "run_without_deal_records"
    shutil.copytree(outcome.run_directory, copy)
    layer1_directory = utils.sdf_stage_dir(copy, 1)
    plan_directory = utils.sdf_stage_dir(copy, 2)
    for name in ("screen_hands.jsonl", "screened_hands.jsonl", "screen_rounds.jsonl", "prompts.jsonl"):
        (layer1_directory / name).unlink(missing_ok=True)
        (plan_directory / name).unlink(missing_ok=True)
    plans = utils.load_jsonl(plan_directory / "plans.jsonl")
    plans[0]["variables"] = {**plans[0]["variables"], "moral_considerations": ["a label"], "epistemic_considerations": []}
    utils.save_jsonl(plans, plan_directory / "plans.jsonl")
    states = [s for s in load_document_states(copy) if s.document_id == plans[0]["prompt_id"]]
    assert states and states[0].dealt_variables is not None and "moral_considerations" in states[0].dealt_variables
    assert states[0].dealt_value_id_notes == ()
    assert set(states[0].dealt_value_ids) == set(states[0].dealt_variables) - {"moral_considerations", "epistemic_considerations"}


# ------------------------------------------------------------------ seeding

@pytest.mark.parametrize("from_step", ["plan", "draft", "critique", "score"])
def test_seeding_copies_the_ids_with_the_records_it_copies(outcome_of, tmp_path, from_step):
    outcome = outcome_of(SCREEN_ON_SCENARIO)
    seeded = seeding.seed_run_from_chosen_step(
        config=outcome.configs_used[0], source_run=outcome.run_directory, from_step=from_step, document_ids=None,
        runs_root=tmp_path / "runs", label="seeded-ids", repository_root=REPOSITORY_ROOT)
    copied = 0
    for stage_file in sorted(seeded.run_directory.glob("layer*/*.jsonl")) + sorted(seeded.run_directory.glob("layer*/*/*.jsonl")):
        for record in utils.load_jsonl(stage_file):
            if "variables" in record:
                assert "variable_value_ids" in record, stage_file.name
                copied += 1
    assert copied > 0 and sum(seeded.copied_record_counts.values()) > 0
    source_ids = {r["prompt_id"]: r["variable_value_ids"]
                  for r in utils.load_jsonl(utils.sdf_stage_dir(outcome.run_directory, 1) / "plans.jsonl")}
    seeded_plans = utils.sdf_stage_dir(seeded.run_directory, 1) / "plans.jsonl"
    if seeded_plans.is_file():
        for record in utils.load_jsonl(seeded_plans):
            assert record["variable_value_ids"] == source_ids[record["prompt_id"]]


def test_a_seeded_copy_of_an_old_run_has_no_ids(outcome_of, tmp_path):
    outcome = outcome_of(SCREEN_ON_SCENARIO, run_started_before_the_field_existed=True)
    seeded = seeding.seed_run_from_chosen_step(
        config=outcome.configs_used[0], source_run=outcome.run_directory, from_step="draft", document_ids=None,
        runs_root=tmp_path / "runs", label="seeded-old", repository_root=REPOSITORY_ROOT)
    for stage_file in seeded.run_directory.glob("layer*/*.jsonl"):
        assert all("variable_value_ids" not in record for record in utils.load_jsonl(stage_file))


COMMITTED_RUNS = sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir())
COMMITTED_RUNS_WITH_A_SNAPSHOT = [run for run in COMMITTED_RUNS if (run / "inputs" / "prompts" / "variables.txt").is_file()]


def test_the_committed_snapshots_are_checked_here_because_none_of_them_has_ids_yet():
    assert COMMITTED_RUNS_WITH_A_SNAPSHOT, "no committed run has a frozen variables.txt any more"


@pytest.mark.parametrize("run", COMMITTED_RUNS, ids=lambda run: run.name)
def test_committed_runs_have_ids_only_where_their_snapshot_carries_them_and_a_note_where_it_cannot(run):
    snapshot_ids, why_not = load_frozen_value_ids(run)
    for state in load_document_states(run):
        if not state.dealt_variables:
            assert state.dealt_value_ids == {} and state.dealt_value_id_notes == ()
        elif snapshot_ids is None:
            # every committed snapshot predates ids (or there is none): no id, one note saying why
            assert state.dealt_value_ids == {}
            assert len(state.dealt_value_id_notes) == 1 and why_not.split(" (")[0][:20] in state.dealt_value_id_notes[0]
        else:
            for axis, value_id in state.dealt_value_ids.items():
                assert snapshot_ids[axis][value_id] == state.dealt_variables[axis]
            assert len(state.dealt_value_id_notes) == len(set(state.dealt_variables) - set(state.dealt_value_ids))
