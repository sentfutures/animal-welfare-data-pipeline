"""Behavior tests for the layer-1 batch incoherence screen, driven through
layer1_screen.run() and screen_descriptions() with a stubbed
shared.api.call_claude (no test here may reach the API; see conftest). The real
screen template stays in play.

Money paths: parse happy path, malformed/truncated/out-of-range answers are not
checkpointed, resume after a completed screen makes zero calls, batches are
stable across resume, a disabled screen makes no calls.
"""

import json
import re
from pathlib import Path

import pytest

from evals import audit_sdf
from sdf_pipeline import layer1_screen as screen
from shared import utils

REPO_ROOT = Path(__file__).resolve().parent.parent
BAD_MARKER = "BAD-SPEC"


def make_plans(count=25, bad_indices=(3, 17)):
    return [
        {
            "prompt_id": f"matrix_{i:06d}",
            "variables": {
                "tone": "skeptical or displeased" if i % 2 else "neutral or journalistic",
                "culture": "France, written in French, with French idioms and references",
            },
            "plan": "…",
            "description": f"Spec {i}. {BAD_MARKER if i in bad_indices else 'Fine.'}",
            "incoherent": False,
        }
        for i in range(count)
    ]


def make_plans_with_a_skeptical_minority(count=40):
    """A quarter of the specifications are skeptical, so a skeptical-heavy removal can exceed 2x."""
    plans = make_plans(count, bad_indices=())
    for i, plan in enumerate(plans):
        plan["variables"]["tone"] = "skeptical or displeased" if i % 4 == 1 else "neutral or journalistic"
    return plans


def numbers_in(user_message):
    """(number, text) for every <specification> block the screen prompt shows."""
    return [(int(n), body) for n, body in re.findall(
        r'<specification number="(\d+)">\n(.*?)</specification>', user_message, re.DOTALL)]


def screen_answer(count, remove=(), reasons=None, **overrides):
    """A schema-shaped answer for a batch of ``count`` specifications."""
    entries = [
        {"number": n, "reasoning": f"Considered specification {n}.", "remove": n in remove,
         "reason": (reasons or {}).get(n, "This specification contradicts itself.") if n in remove else ""}
        for n in range(1, count + 1)
    ]
    return json.dumps({"specifications": entries, **overrides})


def no_removals(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)))


def flag_bad_specs(user_message, **kw):
    """Dispatcher: removes exactly the specifications containing BAD_MARKER."""
    shown = numbers_in(user_message)
    return screen_answer(len(shown), remove={n for n, body in shown if BAD_MARKER in body})


@pytest.fixture
def stage_dir(tmp_path):
    directory = tmp_path / "layer1"
    directory.mkdir()
    return directory


@pytest.fixture
def screen_config(tiny_config):
    tiny_config["sdf"]["screen_batch_size"] = 10
    return tiny_config


def rejected_ids(stage_dir):
    return {r["prompt_id"] for r in utils.load_jsonl(stage_dir / "screen_rejections.jsonl")}


# --- parsing and the public function ---

def test_parse_valid_answer_keeps_every_judgment_in_number_order():
    entries = [
        {"number": 2, "reasoning": "Venue does not exist.", "remove": True, "reason": "Impossible venue."},
        {"number": 1, "reasoning": "Fine.", "remove": False, "reason": ""},
    ]
    judgments = screen.parse_screen_answer(json.dumps({"specifications": entries}), 2)
    assert [(j.number, j.index, j.remove, j.reason) for j in judgments] == [
        (1, 0, False, ""), (2, 1, True, "Impossible venue.")]
    assert judgments[0].reasoning == "Fine."


def valid_entry(number, **overrides):
    return {"number": number, "reasoning": "r", "remove": False, "reason": "", **overrides}


@pytest.mark.parametrize("raw", [
    "no json at all",
    "{\"specifications\": [",                                          # malformed JSON
    json.dumps([valid_entry(1)]),                                       # wrong top-level shape
    json.dumps({"specifications": [valid_entry(1)]}),                   # wrong count (too few)
    json.dumps({"specifications": [valid_entry(1), valid_entry(2), valid_entry(3)]}),  # too many
    json.dumps({"specifications": [valid_entry(1), valid_entry(1)]}),   # duplicate number
    json.dumps({"specifications": [valid_entry(1), valid_entry(3)]}),   # out of range
    json.dumps({"specifications": [valid_entry(0), valid_entry(2)]}),   # out of range
    json.dumps({"specifications": [valid_entry(1), valid_entry(2, remove=True, reason="  ")]}),  # no reason
    json.dumps({"specifications": [valid_entry(1), {"number": 2, "reasoning": "r", "remove": False}]}),  # missing field
    json.dumps({"specifications": [valid_entry(1), valid_entry(2, remove="yes")]}),  # wrong type
    json.dumps({"specifications": [valid_entry(1), valid_entry("2")]}),  # wrong type
])
def test_parse_rejects_unusable_answers(raw):
    with pytest.raises(screen.ScreenAnswerError):
        screen.parse_screen_answer(raw, 2)


def test_answer_schema_is_filled_per_batch_and_keeps_property_order(prompts_sdf):
    prompts = screen.load_screen_prompts(prompts_sdf)
    schema = screen.build_answer_schema(prompts, 4)
    item = schema["properties"]["specifications"]["items"]
    assert item["properties"]["number"]["enum"] == [1, 2, 3, 4]
    assert list(item["properties"]) == ["number", "reasoning", "remove", "reason"]  # reasoning before decision
    assert item["required"] == ["number", "reasoning", "remove", "reason"]
    assert item["additionalProperties"] is False and schema["additionalProperties"] is False
    assert prompts.answer_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] == []


def test_screen_descriptions_returns_every_judgment_through_the_public_function(stub_claude, prompts_sdf):
    calls = stub_claude(flag_bad_specs)
    prompts = screen.load_screen_prompts(prompts_sdf)
    judgments = screen.screen_descriptions(
        ["fine one", f"{BAD_MARKER} two", "fine three"], prompts=prompts, model="m", item_id="b0")
    assert [(j.number, j.remove) for j in judgments] == [(1, False), (2, True), (3, False)]
    assert judgments[1].reason == "This specification contradicts itself."
    assert all(j.reasoning for j in judgments)
    assert calls[0]["stage"] == "layer1_screen" and calls[0]["model"] == "m"
    # the call is schema-constrained, with the enum matching this batch's size
    sent_schema = calls[0]["json_schema"]
    assert sent_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] == [1, 2, 3]
    assert 'number="3"' in calls[0]["user_message"] and 'number="4"' not in calls[0]["user_message"]
    assert "Dealt variables" not in calls[0]["user_message"]
    assert calls[0]["system_prompt"]


def test_screen_descriptions_shows_variables_only_when_given(stub_claude, prompts_sdf):
    calls = stub_claude(no_removals)
    prompts = screen.load_screen_prompts(prompts_sdf)
    screen.screen_descriptions(["a", "b"], [{"tone": "TONE-X"}, {"tone": "TONE-Y"}], prompts=prompts)
    assert "TONE-X" in calls[0]["user_message"] and "TONE-Y" in calls[0]["user_message"]
    assert "Dealt variables" in calls[0]["user_message"]
    with pytest.raises(ValueError, match="parallel"):
        screen.screen_descriptions(["a", "b"], [{"tone": "x"}], prompts=prompts)


def test_no_model_facing_prose_lives_in_the_python_module():
    source = (REPO_ROOT / "sdf_pipeline" / "layer1_screen.py").read_text(encoding="utf-8")
    for phrase in ("Dealt variables", "Use them only as extra context", "Mark a specification"):
        assert phrase not in source


def test_screen_prompt_never_suggests_a_target_count(prompts_sdf):
    system, user = screen.build_screen_prompt(screen.load_screen_prompts(prompts_sdf), ["a", "b", "c"])
    assert "none" in user.lower() and "several" in user.lower()
    assert "<removals>" not in user and "think through" not in user.lower()
    assert "reasoning first" in user and "change or drop a core element" in user
    assert not re.search(r"\b(at least|at most|about|roughly|around) (one|two|three|\d+)\b",
                         user, re.IGNORECASE)
    assert "{" not in user


def test_truncated_answer_is_rejected(stub_claude, prompts_sdf):
    stub_claude(lambda user_message, **kw: (screen_answer(1), "max_tokens"))
    with pytest.raises(screen.ScreenAnswerError, match="truncated"):
        screen.screen_descriptions(["a"], prompts=screen.load_screen_prompts(prompts_sdf))


# --- the stage ---

def test_valid_answers_drop_the_right_specifications(screen_config, prompts_sdf, stage_dir, stub_claude):
    plans = make_plans()
    calls = stub_claude(flag_bad_specs)
    kept = screen.run(screen_config, prompts_sdf, stage_dir, plans)

    assert {p["prompt_id"] for p in kept} == {p["prompt_id"] for p in plans} - {"matrix_000003", "matrix_000017"}
    rejections = utils.load_jsonl(stage_dir / "screen_rejections.jsonl")
    assert {r["prompt_id"] for r in rejections} == {"matrix_000003", "matrix_000017"}
    for rejection in rejections:
        assert rejection["reason"] == "This specification contradicts itself."
        assert BAD_MARKER in rejection["description"]
    # 25 specifications in batches of 10 -> 10, 10, 5; every call tagged for the cost log
    assert len(calls) == 3 and all(c["stage"] == "layer1_screen" for c in calls)
    batches = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    assert sorted(len(b["prompt_ids"]) for b in batches) == [5, 10, 10]
    assert sorted(pid for b in batches for pid in b["prompt_ids"]) == sorted(p["prompt_id"] for p in plans)
    # the model saw local numbers only, never prompt ids
    assert not any("matrix_0" in c["user_message"] for c in calls)
    assert all(c["temperature"] is None for c in calls)  # global temperature applies


def test_incoherent_plans_are_not_screened_and_stay_in_the_returned_plans(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    plans = make_plans(5, bad_indices=())
    plans[2].update(description=None, incoherent=True)
    calls = stub_claude(no_removals)
    kept = screen.run(screen_config, prompts_sdf, stage_dir, plans)
    assert len(calls) == 1 and len(numbers_in(calls[0]["user_message"])) == 4
    assert [p["prompt_id"] for p in kept] == [p["prompt_id"] for p in plans]


def test_batches_are_written_before_any_call(screen_config, prompts_sdf, stage_dir, stub_claude):
    seen_at_first_call = []

    def dispatcher(user_message, **kw):
        batches_path = stage_dir / "screen_batches.jsonl"
        seen_at_first_call.append(len(utils.load_jsonl(batches_path)))
        return no_removals(user_message)

    stub_claude(dispatcher)
    screen.run(screen_config, prompts_sdf, stage_dir, make_plans())
    assert seen_at_first_call == [3, 3, 3]


def test_model_falls_back_from_screen_to_plan_to_global(screen_config, prompts_sdf, stage_dir, stub_claude, tmp_path):
    calls = stub_claude(no_removals)
    plans = make_plans(3, bad_indices=())
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    assert calls[-1]["model"] is None                        # global model
    screen_config["sdf"]["plan_model"] = "plan-model-x"
    second = tmp_path / "second"
    second.mkdir()
    screen.run(screen_config, prompts_sdf, second, plans)
    assert calls[-1]["model"] == "plan-model-x"
    screen_config["sdf"]["screen_model"] = "screen-model-x"
    third = tmp_path / "third"
    third.mkdir()
    screen.run(screen_config, prompts_sdf, third, plans)
    assert calls[-1]["model"] == "screen-model-x"


def test_batch_size_and_variables_knobs(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_batch_size"] = 4
    screen_config["sdf"]["screen_show_variables"] = True
    calls = stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir, make_plans(10, bad_indices=()))
    assert len(calls) == 3  # 4 + 4 + 2
    assert all("Dealt variables" in c["user_message"] for c in calls)


def test_malformed_answer_is_not_checkpointed_and_resume_retries_exactly_that_batch(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    plans = make_plans()
    failing_batch = {}

    def dispatcher(user_message, **kw):
        failing_batch.setdefault("id", kw["item_id"])  # the first batch to be called fails once
        if kw["item_id"] == failing_batch["id"] and not failing_batch.get("retried"):
            failing_batch["retried"] = True
            return "I think number 2 might be a problem."  # not JSON
        return flag_bad_specs(user_message, **kw)

    calls = stub_claude(dispatcher)
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    checkpoint = utils.Checkpoint(stage_dir / "_screen_checkpoint.json")
    assert checkpoint.done_count == 2 and not checkpoint.is_done(failing_batch["id"])

    resumed_calls = stub_claude(flag_bad_specs)
    kept = screen.run(screen_config, prompts_sdf, stage_dir, plans)
    assert [c["item_id"] for c in resumed_calls] == [failing_batch["id"]]
    assert {p["prompt_id"] for p in kept} == {p["prompt_id"] for p in plans} - {"matrix_000003", "matrix_000017"}
    assert len(calls) == 3


@pytest.mark.parametrize("make_bad_answer", [
    lambda count: screen_answer(count - 1),                              # wrong count
    lambda count: screen_answer(count + 1),                              # too many entries
    lambda count: json.dumps({"specifications": [
        {"number": 11, "reasoning": "r", "remove": False, "reason": ""}] * count}),  # out of range, duplicated
    lambda count: screen_answer(count, remove={1}, reasons={1: ""}),    # remove without a reason
    lambda count: "{not json",                                           # malformed JSON
    lambda count: (screen_answer(count), "max_tokens"),                  # truncated
])
def test_unusable_answers_are_not_checkpointed(
    screen_config, prompts_sdf, stage_dir, stub_claude, make_bad_answer
):
    plans = make_plans(5, bad_indices=())
    stub_claude(lambda user_message, **kw: make_bad_answer(len(numbers_in(user_message))))
    with pytest.raises(SystemExit, match="every pending screen call failed"):
        screen.run(screen_config, prompts_sdf, stage_dir, plans)
    assert utils.Checkpoint(stage_dir / "_screen_checkpoint.json").done_count == 0
    assert not (stage_dir / "screen_rejections.jsonl").exists()
    assert not (stage_dir / "screen_judgments.jsonl").exists()


def test_every_judgment_including_keeps_is_stored_with_its_reasoning(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    plans = make_plans()
    stub_claude(flag_bad_specs)
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    judgments = utils.load_jsonl(stage_dir / "screen_judgments.jsonl")
    assert len(judgments) == len(plans)
    assert {j["prompt_id"] for j in judgments} == {p["prompt_id"] for p in plans}
    assert all(j["reasoning"] and j["batch_id"].startswith("screen_batch_") for j in judgments)
    assert {j["prompt_id"] for j in judgments if j["remove"]} == {"matrix_000003", "matrix_000017"}
    assert all(j["reason"] == "" for j in judgments if not j["remove"])


def test_resume_after_a_completed_screen_makes_zero_calls(screen_config, prompts_sdf, stage_dir, stub_claude):
    plans = make_plans()
    stub_claude(flag_bad_specs)
    first = screen.run(screen_config, prompts_sdf, stage_dir, plans)
    calls = stub_claude([])  # any call would exhaust the empty queue and fail
    again = screen.run(screen_config, prompts_sdf, stage_dir, plans)
    assert calls == []
    assert [p["prompt_id"] for p in again] == [p["prompt_id"] for p in first]
    assert len(utils.load_jsonl(stage_dir / "screen_rejections.jsonl")) == 2


def test_batches_are_stable_across_resume_even_if_plan_order_changes(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    plans = make_plans()
    stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    before = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    calls = stub_claude([])
    screen.run(screen_config, prompts_sdf, stage_dir, list(reversed(plans)))
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl") == before and calls == []


def test_batches_are_seeded_and_plan_order_does_not_matter(screen_config, prompts_sdf, tmp_path, stub_claude):
    stub_claude(no_removals)
    plans = make_plans()
    directories = []
    for name, order, seed in (("a", plans, 0), ("b", list(reversed(plans)), 0), ("c", plans, 1)):
        directory = tmp_path / name
        directory.mkdir()
        screen_config["sdf"]["seed"] = seed
        screen.run(screen_config, prompts_sdf, directory, order)
        directories.append(utils.load_jsonl(directory / "screen_batches.jsonl"))
    assert directories[0] == directories[1]
    assert directories[0] != directories[2]


def test_plans_added_on_resume_are_batched_without_touching_existing_batches(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    plans = make_plans(20, bad_indices=())
    stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir, plans[:10])
    before = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    calls = stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    after = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    assert after[:len(before)] == before and len(after) == 2
    assert len(calls) == 1  # only the new batch is screened


def test_disabled_screen_makes_no_calls_and_writes_nothing(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_enabled"] = False
    calls = stub_claude([])
    plans = make_plans()
    assert screen.run(screen_config, prompts_sdf, stage_dir, plans) == plans
    assert calls == [] and list(stage_dir.iterdir()) == []


def test_snapshot_without_the_screen_template_skips_the_screen(screen_config, tmp_path, stage_dir, stub_claude):
    calls = stub_claude([])
    old_snapshot = tmp_path / "old_prompts"
    old_snapshot.mkdir()
    plans = make_plans()
    assert screen.run(screen_config, old_snapshot, stage_dir, plans) == plans
    assert calls == []


def test_drop_rejected_applies_saved_rejections_without_calls(screen_config, prompts_sdf, stage_dir, stub_claude):
    plans = make_plans()
    stub_claude(flag_bad_specs)
    screen.run(screen_config, prompts_sdf, stage_dir, plans)
    calls = stub_claude([])
    kept = screen.drop_rejected(screen_config, stage_dir, plans)
    assert len(kept) == len(plans) - 2 and calls == []
    screen_config["sdf"]["screen_enabled"] = False
    assert screen.drop_rejected(screen_config, stage_dir, plans) == plans


# --- reporting ---

def write_run_with_screen(run_dir, plans, removed_ids):
    stage_dir = run_dir / "layer1"
    stage_dir.mkdir(parents=True)
    utils.save_jsonl(plans, stage_dir / "plans.jsonl")
    batches = [{"batch_id": "screen_batch_0000", "prompt_ids": [p["prompt_id"] for p in plans]}]
    utils.save_jsonl(batches, stage_dir / "screen_batches.jsonl")
    utils.save_jsonl([{"prompt_id": pid, "batch_id": "screen_batch_0000", "number": 1, "reason": "r",
                       "description": "d"} for pid in removed_ids], stage_dir / "screen_rejections.jsonl")
    utils.Checkpoint(stage_dir / "_screen_checkpoint.json").mark_done("screen_batch_0000")
    return stage_dir


def test_removal_rate_rows_flag_values_removed_at_over_twice_the_corpus_rate(tmp_path):
    plans = make_plans_with_a_skeptical_minority()  # 10 skeptical, 30 neutral
    skeptical_removed = [p["prompt_id"] for p in plans if p["variables"]["tone"].startswith("skeptical")][:4]
    neutral_removed = [p["prompt_id"] for p in plans if p["variables"]["tone"].startswith("neutral")][:1]
    write_run_with_screen(tmp_path, plans, skeptical_removed + neutral_removed)
    outputs = screen.load_screen_outputs(tmp_path)
    rows = screen.removal_rate_rows(plans, outputs)
    by_value = {(r["variable"], r["value"]): r for r in rows}
    skeptical = by_value[("tone", "skeptical or displeased")]
    assert (skeptical["screened"], skeptical["removed"], skeptical["rate"]) == (10, 4, 0.4)
    assert skeptical["corpus_rate"] == pytest.approx(5 / 40)
    assert skeptical["flagged"]
    assert not by_value[("tone", "neutral or journalistic")]["flagged"]
    assert not by_value[("culture", plans[0]["variables"]["culture"])]["flagged"]
    assert rows[0]["flagged"]  # flagged rows sort first


def test_load_screen_outputs_is_none_for_a_run_without_the_screen(tmp_path):
    (tmp_path / "layer1").mkdir()
    assert screen.load_screen_outputs(tmp_path) is None
    assert screen.load_screen_outputs(tmp_path / "does_not_exist") is None


def test_only_checkpointed_batches_count_as_screened(tmp_path):
    plans = make_plans(6, bad_indices=())
    stage_dir = write_run_with_screen(tmp_path, plans, [])
    utils.save_jsonl([{"batch_id": "screen_batch_0000", "prompt_ids": ["matrix_000000", "matrix_000001"]},
                      {"batch_id": "screen_batch_0001", "prompt_ids": ["matrix_000002"]}],
                     stage_dir / "screen_batches.jsonl")
    outputs = screen.load_screen_outputs(tmp_path)
    assert outputs["screened_ids"] == {"matrix_000000", "matrix_000001"}
    assert (outputs["batches_done"], outputs["batches_total"]) == (1, 2)


def test_audit_reports_screen_removals_and_flags(tmp_path, capsys):
    plans = make_plans_with_a_skeptical_minority()
    skeptical = [p["prompt_id"] for p in plans if p["variables"]["tone"].startswith("skeptical")][:4]
    write_run_with_screen(tmp_path, plans, skeptical)
    report = {}
    audit_sdf.audit_screen(tmp_path, report)
    output = capsys.readouterr().out
    assert "4 of 40 screened" in output and "skeptical or displeased" in output and "BAD" in output
    assert report["screen"]["removed"] == 4
    assert report["screen"]["flagged_values"] == ["tone = skeptical or displeased"]


def test_audit_still_works_on_a_run_without_screen_files(tmp_path, capsys):
    (tmp_path / "layer1").mkdir()
    report = {}
    audit_sdf.audit_screen(tmp_path, report)
    assert "no screen files" in capsys.readouterr().out
    assert "screen" not in report
