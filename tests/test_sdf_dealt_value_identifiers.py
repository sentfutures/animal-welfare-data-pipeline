"""Dealt value ids: kept in the document state and the event log only, never in a stage file or request.

New runs record the ids of each hand in the ``round_dealt`` event; older runs get them from a lookup in
the run's frozen ``inputs/prompts/variables.txt``. Nothing that builds a prompt reads ``dealt_value_ids``.
"""

import json
import shutil
from pathlib import Path

import pytest

from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_screen as screen
from sdf_pipeline.dealt_value_identifiers import load_frozen_value_ids, value_ids_of_dealt_hand
from sdf_pipeline.document_event_log import fold_dealt_value_ids, read_document_events
from sdf_pipeline.document_state_loader import load_document_states
from shared import utils
from test_layer1_screen import remove_first_hand_of_round_zero_batches, screen_config  # noqa: F401

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "sdf" / "runs"


def ids_that_name_the_dealt_text(hand_variables, ids_of_hand, value_ids_by_axis):
    return all(value_ids_by_axis[axis][ids_of_hand[axis]] == text for axis, text in hand_variables.items())


def test_the_lookup_gives_ids_and_a_note_for_text_nobody_has_or_two_values_share():
    ids, notes = value_ids_of_dealt_hand(
        {"tone": "warm", "mood": "missing", "pace": "same"},
        {"tone": {"t1": "warm"}, "mood": {"m1": "calm"}, "pace": {"p1": "same", "p2": "same"}})
    assert ids == {"tone": "t1"}
    assert len(notes) == 2 and "{mood}" in notes[0] and "{pace}" in notes[1]


def test_new_run_ids_equal_the_ids_of_the_dealt_values_and_stay_out_of_every_stage_file(
        screen_config, prompts_sdf, stage_dir, stub_claude):  # noqa: F811
    stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    _, _, value_ids = cp.load_variables(prompts_sdf / cp.DEFAULT_VARIABLES.name)
    events = read_document_events(stage_dir.parent / "document_events.jsonl")
    recorded = fold_dealt_value_ids(events)
    hands = utils.load_jsonl(stage_dir / "screen_hands.jsonl")
    assert {h["prompt_id"] for h in hands} == set(recorded)           # round 0 and the redeal
    assert any(h["round"] > 0 for h in hands)
    for hand in hands:
        assert ids_that_name_the_dealt_text(hand["variables"], recorded[hand["prompt_id"]], value_ids)
        assert set(recorded[hand["prompt_id"]]) == set(hand["variables"])
    for path in stage_dir.rglob("*"):
        if path.is_file():
            assert "dealt_value_ids" not in path.read_text(encoding="utf-8"), path.name
    # the loader reads the event ids and puts nothing in ``variables``
    states = {s.document_id: s for s in load_document_states(stage_dir.parent)}
    for hand in hands:
        state = states[hand["prompt_id"]]
        assert state.dealt_value_ids == recorded[hand["prompt_id"]]
        assert state.dealt_value_id_notes == ()
        assert state.variables == hand["variables"]


def test_an_old_run_without_ids_in_its_events_is_filled_from_its_frozen_variables(
        screen_config, prompts_sdf, stage_dir, stub_claude):  # noqa: F811
    stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    run = stage_dir.parent
    events_path = run / "document_events.jsonl"
    events = read_document_events(events_path)
    for event in events:
        event.pop("dealt_value_ids", None)
        event.pop("dealt_value_id_notes", None)
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    hands = utils.load_jsonl(stage_dir / "screen_hands.jsonl")
    _, _, value_ids = cp.load_variables(prompts_sdf / cp.DEFAULT_VARIABLES.name)

    without_snapshot = {s.document_id: s for s in load_document_states(run)}
    assert all(s.dealt_value_ids == {} and len(s.dealt_value_id_notes) == 1
               and "no frozen" in s.dealt_value_id_notes[0] for s in without_snapshot.values())

    (run / "inputs" / "prompts").mkdir(parents=True)
    shutil.copy(prompts_sdf / cp.DEFAULT_VARIABLES.name, run / "inputs" / "prompts" / "variables.txt")
    with_snapshot = {s.document_id: s for s in load_document_states(run)}
    for hand in hands:
        state = with_snapshot[hand["prompt_id"]]
        assert ids_that_name_the_dealt_text(hand["variables"], state.dealt_value_ids, value_ids)
        assert state.dealt_value_id_notes == ()

    # a value reworded after the run: its axis gets no id and a note, the others keep theirs
    frozen = run / "inputs" / "prompts" / "variables.txt"
    first_text = next(iter(hands[0]["variables"].values()))
    frozen.write_text(frozen.read_text(encoding="utf-8").replace(first_text, first_text + " reworded"),
                      encoding="utf-8")
    reworded = load_document_states(run)[0]
    assert len(reworded.dealt_value_id_notes) >= 1 and len(reworded.dealt_value_ids) < len(hands[0]["variables"])


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
