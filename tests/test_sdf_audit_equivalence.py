"""Every judge request of every model-judged SDF audit, on a run of each on-disk layout, equals its recording in ``tests/fixtures/sdf_audit_equivalence/``.
A deliberate change regenerates the recordings with ``scripts/regenerate_sdf_audit_equivalence_recorded_requests.py``; a refactor never does.
The recording engine is in ``tests/sdf_audit_equivalence_fixture_support.py``."""

import tempfile
from pathlib import Path

import pytest

import sdf_audit_equivalence_fixture_support as support

CASES = [(audit.name, layout.name) for audit in support.AUDITS for layout in support.RUN_LAYOUTS]


@pytest.fixture(scope="module")
def outcome_of():
    """Prepares the three runs once, then runs each audit on each run once, on first use."""
    cache = {}
    with tempfile.TemporaryDirectory() as base_directory_name:
        base_directory = Path(base_directory_name)
        prepared_runs = support.prepare_all_runs(base_directory)

        def get(audit_name, layout_name):
            key = (audit_name, layout_name)
            if key not in cache:
                cache[key] = support.run_audit_in_isolation(
                    support.audit_named(audit_name), support.layout_named(layout_name),
                    prepared_runs[layout_name], base_directory / "audits" / support.case_name(*key))
            return cache[key]
        yield get


def test_recorded_files_cover_exactly_the_audits_and_layouts():
    recorded = sorted(path.name for path in support.FIXTURE_DIRECTORY.glob("recorded_*_on_origin_main.json"))
    assert recorded == sorted(f"recorded_{support.case_name(*case)}_on_origin_main.json" for case in CASES), \
        support.REGENERATION_RULE


@pytest.mark.parametrize("audit_name,layout_name", CASES)
def test_every_judge_request_equals_the_one_recorded_on_origin_main(audit_name, layout_name, outcome_of):
    recorded = support.load_recorded_case(audit_name, layout_name)
    current = support.record_for_outcome(outcome_of(audit_name, layout_name))
    differences = support.differences_between_recorded_and_current_requests(recorded, current)
    assert not differences, (
        f"{len(differences)} difference(s) from the judge requests recorded on origin/main:\n  "
        + "\n  ".join(differences[:20]) + ("\n  ..." if len(differences) > 20 else "")
        + f"\n{support.REGENERATION_RULE}")


def test_the_recorded_requests_exercise_every_model_judged_audit_on_every_readable_layout():
    """Each model-judged audit must have sent requests on the matrix and current layouts."""
    for audit in support.AUDITS:
        for layout_name in ("pre_renumber_matrix", "current"):
            recorded = support.load_recorded_case(audit.name, layout_name)
            if audit.makes_model_calls:
                assert recorded["requests"], f"{audit.name} on {layout_name} recorded no judge request"
            if audit.embeds_texts:
                assert recorded["embedding_requests"], f"{audit.name} on {layout_name} recorded no embedding request"
            if not audit.makes_model_calls:
                assert recorded["ended_with"] == "returned" and recorded["built_input_files"]
