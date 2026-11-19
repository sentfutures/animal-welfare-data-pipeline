"""The implausibility screen (the deal and the redeal are pinned by the prompt equivalence recordings).

Step unit tests call ``group``, ``read_reply`` and ``records`` with no stubbed API and no files. The
runner and layer tests pin the requests, the stage files, the stop messages, resume (identical rounds,
zero calls for finished batches).
``run_layer.run`` is the layer's entry point (scripts and tests import it).
"""

import dataclasses
import json
import random

import pytest

from sdf_pipeline.layer1_deal_variables_and_screen import run_layer as screen_layer
from sdf_pipeline.layer1_deal_variables_and_screen import screen
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind, ModelReply, Removed, StepProgress
from sdf_step_and_command_line_test_helpers import run_of
from shared import utils
from shared.sdf_run_layouts import DEAL_VARIABLES_AND_SCREEN
from sdf_screen_test_builders import (
    no_removals, remove_the_first_three_hands_of_every_batch,
    screen_answer, screen_config)  # noqa: F401  (screen_config is a fixture)


def make_hand(index, round_number=0, **variables):
    return {"prompt_id": f"matrix_{index:06d}", "round": round_number,
            "variables": variables or {"tone": f"tone {index}"}}


def render_lines(hand):
    return f"tone: {hand['variables'].get('tone', '')}"


def screen_run_of(config, prompts_sdf, stage_dir, seed=0):
    """A run of the screen with batches of 3, a simple line renderer and the given shuffle seed."""
    sdf = {**config["sdf"], "screen_model": "screen-model", "screen_batch_size": 3, "seed": seed}
    run = run_of({**config, "sdf": sdf}, prompts_sdf, DEAL_VARIABLES_AND_SCREEN, stage_dir)
    run.loaded[screen._render_variable_lines.__wrapped__] = render_lines
    return run


@pytest.fixture
def screen_run(tiny_config, prompts_sdf, stage_dir):
    return screen_run_of(tiny_config, prompts_sdf, stage_dir)


def batch_of(hand_indexes, batch_id="screen_round0_batch0000"):
    return [{**make_hand(i), "batch_id": batch_id} for i in hand_indexes]


def ids_of(batch):
    return [hand["prompt_id"] for hand in batch]


# ---------------------------------------------------------------- grouping

def test_group_is_a_seeded_shuffle_of_the_sorted_ids_cut_into_batches(tiny_config, prompts_sdf, stage_dir, tmp_path):
    hands = [make_hand(i, round_number=2) for i in (4, 0, 3, 1, 2, 5, 6)]
    run = screen_run_of(tiny_config, prompts_sdf, stage_dir, seed=7)
    batches_by_id = screen.STEP.group(hands, run)
    batches = list(batches_by_id.values())
    expected_order = sorted(h["prompt_id"] for h in hands)
    random.Random("7:layer1_screen:round2").shuffle(expected_order)
    assert [ids_of(b) for b in batches] == [expected_order[0:3], expected_order[3:6], expected_order[6:]]
    assert list(batches_by_id) == [b[0]["batch_id"] for b in batches] == [f"screen_round2_batch000{i}" for i in range(3)]
    assert all(hand["round"] == 2 for batch in batches for hand in batch)
    # input order is irrelevant; the seed and the round are not

    def batch_ids_regrouped(changed_hands, seed, name):
        folder = tmp_path / name
        folder.mkdir()
        return [ids_of(b) for b in screen.STEP.group(
            changed_hands, screen_run_of(tiny_config, prompts_sdf, folder, seed=seed)).values()]

    assert batch_ids_regrouped(list(reversed(hands)), 7, "reversed") == [ids_of(b) for b in batches]
    assert batch_ids_regrouped(hands, 8, "other_seed") != [ids_of(b) for b in batches]
    assert batch_ids_regrouped([{**h, "round": 3} for h in hands], 7, "other_round") != [ids_of(b) for b in batches]


def test_group_matches_the_saved_plan_records(tiny_config, prompts_sdf, stage_dir):
    hands = [make_hand(i) for i in range(8)]
    planned = screen.plan_round_batches(stage_dir, 0, [h["prompt_id"] for h in hands], 3, 5)
    batches = screen.STEP.group(hands, screen_run_of(tiny_config, prompts_sdf, stage_dir, seed=5)).values()
    assert [{"batch_id": b[0]["batch_id"], "round": 0, "prompt_ids": ids_of(b)} for b in batches] == planned
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl") == planned


def test_the_batch_plan_is_saved_once_and_reused_on_resume(stage_dir, tmp_path):
    ids = [f"matrix_{i:06d}" for i in range(7)]
    first = screen.plan_round_batches(stage_dir, 0, ids, 3, 1)
    again = screen.plan_round_batches(stage_dir, 0, ids, 2, 1)  # a different batch size on resume
    assert again == first
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl") == first
    # a plan that holds only some of the round's hands is planned again
    utils.save_jsonl(first[:1], stage_dir / "screen_batches.jsonl")
    replanned = screen.plan_round_batches(stage_dir, 0, ids, 3, 1)
    assert replanned == first
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl") == first


# ---------------------------------------------------------------- reading the reply and the request

def test_read_accepts_the_kept_hands_and_removes_the_implausible_ones(screen_run):
    batch = batch_of([0, 1, 2])
    outcomes = screen.STEP.read_reply(batch, ModelReply(screen_answer(3, remove={2}, reasons={2: " cannot be true "}), "end_turn"), screen_run)
    assert list(outcomes) == ids_of(batch)
    assert outcomes["matrix_000000"] == Accepted({"number": 1, "reasoning": "Considered hand 1.", "reason": ""})
    assert outcomes["matrix_000001"] == Removed(
        "extremely_implausible", {"number": 2, "reasoning": "Considered hand 2.", "reason": "cannot be true"})
    records = screen.STEP.records(batch, outcomes, screen_run)
    judgments = [record for file_name, record in records.records if file_name == "screen_judgments.jsonl"]
    rejections = [record for file_name, record in records.records if file_name == "screen_rejections.jsonl"]
    assert [list(j) for j in judgments] == [["prompt_id", "round", "batch_id", "number", "reasoning", "remove", "reason"]] * 3
    assert [j["remove"] for j in judgments] == [False, True, False]
    assert rejections == [{
        "prompt_id": "matrix_000001", "round": 0, "batch_id": "screen_round0_batch0000", "number": 2,
        "reason": "cannot be true", "variables": {"tone": "tone 1"}, "variable_lines": "tone: tone 1"}]
    assert records.replacing == (
        ("screen_judgments.jsonl", "batch_id", "screen_round0_batch0000"),
        ("screen_rejections.jsonl", "batch_id", "screen_round0_batch0000"))


@pytest.mark.parametrize("text, stop_reason, fragment", [
    ("{}", "max_tokens", "truncated answer (stop_reason=max_tokens)"),
    ("not json", "end_turn", "unusable answer"),
    (screen_answer(2), "end_turn", "expected 3 entries, got 2"),
    (json.dumps({"specifications": [
        {"number": n, "reasoning": "r", "remove": False, "reason": ""} for n in (1, 1, 3)]}), "end_turn", "listed twice"),
    (json.dumps({"specifications": [
        {"number": n, "reasoning": "r", "remove": False, "reason": ""} for n in (1, 2, 4)]}), "end_turn", "outside 1..3"),
])
def test_read_fails_every_member_of_an_unusable_answer(screen_run, text, stop_reason, fragment):
    batch = batch_of([0, 1, 2])
    outcomes = screen.STEP.read_reply(batch, ModelReply(text, stop_reason), screen_run)
    assert list(outcomes) == ids_of(batch)
    for outcome in outcomes.values():
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
        assert fragment in outcome.reason
    assert screen.STEP.progress_line(batch, outcomes, StepProgress(pending_count=1, groups_seen=1), screen_run).startswith(
        "  screen_round0_batch0000: unusable answer (")
    assert screen.STEP.records(batch, outcomes, screen_run).records == ()


def test_a_thrown_call_prints_todays_line_and_both_kinds_count_towards_the_systemic_stop(screen_run):
    batch = batch_of([0])
    thrown = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    assert screen.STEP.progress_line(batch, {"matrix_000000": thrown}, StepProgress(pending_count=1, groups_seen=1), screen_run) == (
        "  screen_round0_batch0000: API call failed (RuntimeError: boom) — will retry on resume")
    assert screen.STEP.policy.failure_kinds_counting_towards_the_systemic_stop == {
        FailureKind.THROWN_CALL, FailureKind.UNUSABLE_ANSWER}


# ---------------------------------------------------------------- the runner on the screen

def round_zero_hands(count=6):
    return [make_hand(i) for i in range(count)]


def judged_batch_ids(stage_dir):
    return {record["batch_id"] for record in utils.load_jsonl(stage_dir / "screen_judgments.jsonl")}


def test_a_failed_batch_is_not_written_and_stops_the_run_with_its_message(screen_run, stage_dir, capsys):
    def send(request):
        if request.item_id.endswith("0001"):
            raise RuntimeError("boom")
        return ModelReply(screen_answer(3), "end_turn")

    with pytest.raises(SystemExit) as stop:
        run_step(screen.STEP, screen_run, round_zero_hands(), send=send)
    assert str(stop.value) == ("layer1_screen: 1 batch(es) of round 0 failed; the next round cannot be dealt "
                               "until the round is complete — --resume retries them.")
    assert judged_batch_ids(stage_dir) == {"screen_round0_batch0000"}
    output = capsys.readouterr().out
    assert "  screen_round0_batch0001: API call failed (RuntimeError: boom) — will retry on resume" in output
    assert "  Screened screen_round0_batch0000 (1/2): 3 hands, 0 extremely implausible," in output


@pytest.mark.parametrize("send", [
    lambda request: (_ for _ in ()).throw(RuntimeError("down")),
    lambda request: ModelReply("garbage", "end_turn"),
])
def test_every_pending_batch_failing_is_systemic_whether_thrown_or_unusable(screen_run, stage_dir, send):
    with pytest.raises(SystemExit) as stop:
        run_step(screen.STEP, screen_run, round_zero_hands(3), send=send)
    assert str(stop.value) == screen.STEP.systemic_stop_message
    assert not (stage_dir / "screen_judgments.jsonl").exists()


def test_finished_batches_are_not_called_again(screen_run):
    calls = []

    def send(request):
        calls.append(request.item_id)
        return ModelReply(screen_answer(3, remove={1}), "end_turn")

    run_step(screen.STEP, screen_run, round_zero_hands(), send=send)
    run_step(screen.STEP, screen_run, round_zero_hands(), send=send)   # a resume: every batch is finished
    assert sorted(calls) == ["screen_round0_batch0000", "screen_round0_batch0001"]   # workers answer in any order


def test_a_run_with_the_marks_of_earlier_code_is_not_billed_again_for_batches_current_code_finished(screen_run, stage_dir):
    """Earlier code resumed by its marks alone, so current code keeps the marks of such a run up to date: a batch it
    finishes after the first resume is not sent again by the second."""
    sent = []

    def send(request):
        sent.append(request.item_id)
        if request.item_id.endswith("0001"):
            raise RuntimeError("interrupted")
        return ModelReply(screen_answer(3), "end_turn")

    with pytest.raises(SystemExit):
        run_step(screen.STEP, screen_run, round_zero_hands(), send=send)   # batch 0 finished, batch 1 not
    utils.Checkpoint(stage_dir / "_screen_checkpoint.json").mark_done("screen_round0_batch0000")   # as earlier code left it
    sent.clear()

    run_step(screen.STEP, screen_run, round_zero_hands(), send=lambda request: (
        sent.append(request.item_id), ModelReply(screen_answer(3), "end_turn"))[1])   # the first resume finishes batch 1
    assert sent == ["screen_round0_batch0001"]
    sent.clear()
    run_step(screen.STEP, screen_run, round_zero_hands(), send=lambda request: (
        sent.append(request.item_id), ModelReply(screen_answer(3), "end_turn"))[1])   # the second resume has nothing to send
    assert sent == []


# ---------------------------------------------------------------- the layer: loop, rounds

def test_the_loop_ends_after_the_round_limit_and_the_leftover_hands_are_dropped(
        screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_redeal_rounds"] = 2
    stub_claude(remove_the_first_three_hands_of_every_batch)
    screen_layer.run(screen_config, prompts_sdf, stage_dir)
    rounds = utils.load_jsonl(stage_dir / "screen_rounds.jsonl")
    assert [r["round"] for r in rounds] == [0, 1, 2]
    summary = json.loads((stage_dir / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["leftover_dropped"] == len(rounds[2]["prompt_ids"]) > 0


def test_a_completed_screen_makes_no_calls_and_a_disabled_screen_writes_nothing(screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(no_removals)
    screen_layer.run(screen_config, prompts_sdf, stage_dir)
    calls = stub_claude(no_removals)
    screen_layer.run(screen_config, prompts_sdf, stage_dir)
    assert calls == []
    disabled = stage_dir.parent / "disabled" / "layer1"
    disabled.mkdir(parents=True)
    screen_config["sdf"]["screen_enabled"] = False
    assert screen_layer.run(screen_config, prompts_sdf, disabled) is None
    assert list(disabled.iterdir()) == [] and list(disabled.parent.iterdir()) == [disabled]


def test_a_prompt_building_error_fails_that_batch_and_a_resume_retries_it(screen_run, stage_dir, capsys):
    def broken_build_request(batch, run):
        if batch[0]["batch_id"].endswith("0001"):
            raise KeyError("missing_placeholder")
        return screen.STEP.build_request(batch, run)

    sent = []

    def send(request):
        sent.append(request.item_id)
        return ModelReply(screen_answer(3), "end_turn")

    with pytest.raises(SystemExit, match="failed; the next round cannot be dealt"):
        run_step(dataclasses.replace(screen.STEP, build_request=broken_build_request), screen_run, round_zero_hands(), send=send)
    assert sent == ["screen_round0_batch0000"] and judged_batch_ids(stage_dir) == {"screen_round0_batch0000"}
    assert ("  screen_round0_batch0001: API call failed (KeyError: 'missing_placeholder') — will retry on resume"
            in capsys.readouterr().out)

    run_step(screen.STEP, screen_run, round_zero_hands(), send=send)
    assert sent[1:] == ["screen_round0_batch0001"] and judged_batch_ids(stage_dir) == {
        "screen_round0_batch0000", "screen_round0_batch0001"}


def test_a_template_broken_for_every_batch_stops_the_run_with_no_model_call(screen_run):
    def broken_build_request(batch, run):
        raise KeyError("missing_placeholder")

    sent = []
    with pytest.raises(SystemExit) as stop:
        run_step(dataclasses.replace(screen.STEP, build_request=broken_build_request), screen_run, round_zero_hands(),
                 send=lambda request: sent.append(request))
    assert "an error building the prompt" in str(stop.value) and "systemic" in str(stop.value)
    assert sent == []
