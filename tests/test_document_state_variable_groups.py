"""The three read-only groups of a document state's ``variables`` dict (``dealt_cards``, ``revision_cards``,
``assigned_considerations``), on the committed SDF runs and on the prompt-equivalence fixture scenarios.

The groups filter the stored dict by key and store nothing, so the tests check that every key lands in exactly
one group, that the groups together give back ``variables``, that ``variables`` is untouched by reading them,
and that every dealt card names an axis of the variables file the run was dealt from.
"""

import copy
import tempfile
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline import compose_prompts
from sdf_pipeline.document_state import ASSIGNED_CONSIDERATION_KEYS
from sdf_pipeline.document_state_loader import load_document_states
from shared import utils

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "sdf" / "runs"
COMMITTED_RUN_NAMES = sorted(path.name for path in RUNS_ROOT.iterdir() if path.is_dir()) if RUNS_ROOT.is_dir() else []


def _axis_names_of_the_variables_file_dealt_from(run_directory: Path) -> set:
    """The axes of the variables file the run froze at its start, or of today's file for a run that froze none."""
    frozen = run_directory / "inputs" / "prompts" / "variables.txt"
    path = frozen if frozen.is_file() else compose_prompts.DEFAULT_VARIABLES
    return set(compose_prompts.parse_variables(path))


def _check_groups_of_every_state(states, axis_names) -> int:
    keys_seen = 0
    for state in states:
        variables_before = copy.deepcopy(state.variables)
        groups = (state.dealt_cards, state.revision_cards, state.assigned_considerations)
        assert state.variables == variables_before, "reading a group changed variables"
        key_sets = [set(group) for group in groups]
        assert not (key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]), \
            f"{state.document_id}: a key is in two groups"
        assert {key: value for group in groups for key, value in group.items()} == state.variables
        assert set(state.revision_cards) <= set(compose_prompts.REVISION_AXES)
        assert set(state.assigned_considerations) <= set(ASSIGNED_CONSIDERATION_KEYS)
        # no key is hidden in the dealt group: each is an axis of the variables file the run was dealt from
        assert set(state.dealt_cards) <= axis_names, \
            f"{state.document_id}: keys of no known axis {set(state.dealt_cards) - axis_names}"
        assert state.variables == (state.latest_text_version.written_from.variables
                                   if state.latest_text_version and state.latest_text_version.written_from.variables is not None
                                   else (state.dealt_variables or {}))
        keys_seen += len(state.variables)
    return keys_seen


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_variable_groups_partition_variables_on_the_committed_runs(run_name):
    run_directory = RUNS_ROOT / run_name
    try:
        states = load_document_states(run_directory)
    except Exception as error:  # a folder under runs/ that is not a loadable run
        pytest.skip(f"not a loadable run: {error}")
    _check_groups_of_every_state(states, _axis_names_of_the_variables_file_dealt_from(run_directory))


def test_committed_runs_predate_revision_cards_and_assigned_considerations():
    """Every committed run was made before the revision axes and the consideration assignment, so its
    variables are all dealt cards (the fixture scenarios below cover the other two groups)."""
    for run_name in COMMITTED_RUN_NAMES:
        for state in load_document_states(RUNS_ROOT / run_name):
            assert state.revision_cards == {} and state.assigned_considerations == {}
            assert state.dealt_cards == state.variables


@pytest.mark.parametrize("name", [scenario.name for scenario in scenarios.all_scenarios()])
def test_variable_groups_partition_variables_on_the_fixture_scenarios(name):
    with tempfile.TemporaryDirectory() as base_directory:
        captured = []
        support.run_scenario_in_isolation(
            scenarios.scenario_named(name), Path(base_directory) / name,
            after_each_invocation=lambda number, run_directory: captured.append(
                (load_document_states(run_directory), _axis_names_of_the_variables_file_dealt_from(run_directory))))
        for states, axis_names in captured:
            _check_groups_of_every_state(states, axis_names)
        if name == scenarios.MAIN_SCENARIO_NAME:
            final_states = captured[-1][0]
            assert any(state.assigned_considerations for state in final_states)
            assert any(state.revision_cards for state in final_states)
