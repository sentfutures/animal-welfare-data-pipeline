"""DAD's model requests are unchanged by the shared request-settings resolver.

Drives the DAD pipeline through ``dad_pipeline/run.py``'s real ``main`` on a two-scenario run with canned
replies, for the api, claude_code and auto backends and three configurations, and compares every request that
reaches a provider client (model, system prompt and cache flag, messages, max_tokens, temperature, thinking,
effort, output schema, route, and the Claude Code SDK options) with the files under
tests/fixtures/dad_request_settings_equivalence/, which were recorded on origin/main code by
scripts/regenerate_dad_request_settings_equivalence_recorded_requests.py. See
tests/dad_request_settings_equivalence_fixture_support.py for what is recorded and how.

A failure names the configuration, backend, stage, item, attempt and field. A pull request that changes a DAD
request on purpose regenerates the recorded files in the same pull request; a refactor never does.
"""

import json

import pytest

import dad_request_settings_equivalence_fixture_support as support


def _id_of(case_key):
    return support.case_name(*case_key)


@pytest.mark.parametrize("case_key", support.all_case_keys(), ids=_id_of)
def test_dad_requests_match_the_recording_made_on_origin_main(case_key, tmp_path, monkeypatch):
    configuration_name, backend = case_key
    recorded = support.load_recorded_case(configuration_name, backend)
    outcome = support.run_case(configuration_name, backend, tmp_path, monkeypatch)
    differences = support.differences_between_recorded_and_current_case(recorded, outcome)
    assert not differences, (
        f"{len(differences)} difference(s) between the DAD requests recorded on origin/main and the ones sent "
        f"now:\n  " + "\n  ".join(differences) + f"\n{support.REGENERATION_RULE}")


@pytest.mark.parametrize("configuration_name", list(support.CONFIGURATIONS))
def test_recordings_came_from_unmodified_origin_main_code(configuration_name):
    path = support.recorded_path_for_configuration(configuration_name)
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded["python_files_differing_from_origin_main_when_recorded"] == [], (
        f"{path.name} was recorded with these Python files differing from origin/main: "
        f"{recorded['python_files_differing_from_origin_main_when_recorded']}. {support.REGENERATION_RULE}")


def test_every_dad_stage_and_every_backend_is_recorded():
    stages, routes = set(), set()
    for configuration_name, backend in support.all_case_keys():
        for request in support.load_recorded_case(configuration_name, backend)["requests"]:
            stages.add(request["stage"])
            routes.add((backend, request["route"]))
    assert stages == set(support.DAD_STAGE_TAGS)
    assert {("api", "anthropic"), ("claude_code", "claude_code"), ("auto", "claude_code"),
            ("auto", "anthropic"), ("api", "openai")} <= routes
