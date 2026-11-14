"""The deal, the implausibility screen and the redeal as update steps.

Step unit tests call ``group``, ``interpret`` and ``apply`` with no stubbed API and no files. The
runner and layer tests pin the requests, the stage files, the stop messages, resume (identical rounds,
zero calls for finished batches) and the three non-per-document events plus the per-hand screen events.
``layer1_screen.run`` keeps its signature and behaviour (scripts and tests import it).
"""

import json
import random

import pytest

from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_screen as screen
from sdf_pipeline.document_event_log import (
    DocumentEventLog, fold_document_events, fold_round_and_batch_plan_events, read_document_events)
from sdf_pipeline.layer1_screen import DealStep, RedealStep, ScreenBatch, ScreenStep
from sdf_pipeline.step_runner import BatchStepRunner
from sdf_pipeline.update_steps import Accepted, Created, Failed, FailureKind, ModelReply, ModelRequest, Removed
from shared import utils
from test_layer1_screen import (
    no_removals, numbers_in, remove_first_hand_of_round_zero_batches, remove_the_first_hand_of_every_batch,
    remove_the_first_three_hands_of_every_batch,
    screen_answer, screen_config)  # noqa: F401  (screen_config is a fixture)


def make_hand(index, round_number=0, **variables):
    return {"prompt_id": f"matrix_{index:06d}", "round": round_number,
            "variables": variables or {"tone": f"tone {index}"}}


def render_lines(hand):
    return f"tone: {hand['variables'].get('tone', '')}"


@pytest.fixture
def screen_step(prompts_sdf):
    return ScreenStep(screen.load_screen_prompts(prompts_sdf), "screen-model", 3, render_lines)


def batch_of(screen_step, hand_indexes, batch_id="screen_round0_batch0000"):
    hands = {h["prompt_id"]: h for h in (make_hand(i) for i in hand_indexes)}
    return screen_step.batch_from_record(
        {"batch_id": batch_id, "round": 0, "prompt_ids": list(hands)}, hands)


def events_of(stage_dir):
    return read_document_events(stage_dir.parent / "document_events.jsonl")


# ---------------------------------------------------------------- grouping

def test_group_is_a_seeded_shuffle_of_the_sorted_ids_cut_into_batches(screen_step):
    hands = [make_hand(i) for i in (4, 0, 3, 1, 2, 5, 6)]
    batches = screen_step.group(hands, 7, 2)
    expected_order = sorted(h["prompt_id"] for h in hands)
    random.Random("7:layer1_screen:round2").shuffle(expected_order)
    assert [b.document_ids for b in batches] == [expected_order[0:3], expected_order[3:6], expected_order[6:]]
    assert [b.batch_id for b in batches] == [f"screen_round2_batch000{i}" for i in range(3)]
    assert all(b.round_number == 2 for b in batches)
    # input order is irrelevant; the seed and the round are not
    assert [b.document_ids for b in screen_step.group(list(reversed(hands)), 7, 2)] == [b.document_ids for b in batches]
    assert [b.document_ids for b in screen_step.group(hands, 8, 2)] != [b.document_ids for b in batches]
    assert [b.document_ids for b in screen_step.group(hands, 7, 3)] != [b.document_ids for b in batches]


def test_group_matches_the_saved_plan_records(screen_step, stage_dir):
    hands = [make_hand(i) for i in range(8)]
    planned = screen.plan_round_batches(stage_dir, 0, [h["prompt_id"] for h in hands], 3, 5)
    assert [b.as_record() for b in screen_step.group(hands, 5, 0)] == planned
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl") == planned


def test_a_batch_shows_each_member_its_rendered_lines_in_order(screen_step):
    batch = batch_of(screen_step, [2, 0, 1])
    assert batch.shown_lines == ("tone: tone 2", "tone: tone 0", "tone: tone 1")


def test_the_batch_plan_is_recorded_once_and_reused_on_resume(stage_dir, tmp_path):
    log = DocumentEventLog(tmp_path / "document_events.jsonl")
    ids = [f"matrix_{i:06d}" for i in range(7)]
    first = screen.plan_round_batches(stage_dir, 0, ids, 3, 1, log)
    again = screen.plan_round_batches(stage_dir, 0, ids, 2, 1, log)  # a different batch size on resume
    assert again == first
    events = read_document_events(log.path)
    assert [(e["event"], e["round"], e["batches"]) for e in events] == [("screen_batches_planned", 0, first)]
    # a plan that holds only some of the round's hands is planned again, and recorded again
    utils.save_jsonl(first[:1], stage_dir / "screen_batches.jsonl")
    replanned = screen.plan_round_batches(stage_dir, 0, ids, 3, 1, log)
    assert replanned == first
    assert len(read_document_events(log.path)) == 2


# ---------------------------------------------------------------- ScreenStep.interpret and the request

def test_interpret_accepts_the_kept_hands_and_removes_the_implausible_ones(screen_step):
    batch = batch_of(screen_step, [0, 1, 2])
    outcomes = screen_step.interpret(batch, ModelReply(screen_answer(3, remove={2}, reasons={2: " cannot be true "}), "end_turn"))
    assert list(outcomes) == batch.document_ids
    assert outcomes["matrix_000000"] == Accepted({"number": 1, "reasoning": "Considered hand 1.", "reason": ""})
    assert outcomes["matrix_000001"] == Removed(
        "extremely_implausible", {"number": 2, "reasoning": "Considered hand 2.", "reason": "cannot be true"})
    judgments, rejections = screen_step.stage_file_records(batch, outcomes)
    assert [list(j) for j in judgments] == [["prompt_id", "round", "batch_id", "number", "reasoning", "remove", "reason"]] * 3
    assert [j["remove"] for j in judgments] == [False, True, False]
    assert rejections == [{
        "prompt_id": "matrix_000001", "round": 0, "batch_id": "screen_round0_batch0000", "number": 2,
        "reason": "cannot be true", "variables": {"tone": "tone 1"}, "variable_lines": "tone: tone 1"}]


@pytest.mark.parametrize("text, stop_reason, fragment", [
    ("{}", "max_tokens", "truncated answer (stop_reason=max_tokens)"),
    ("not json", "end_turn", "unusable answer"),
    (screen_answer(2), "end_turn", "expected 3 entries, got 2"),
    (json.dumps({"specifications": [
        {"number": n, "reasoning": "r", "remove": False, "reason": ""} for n in (1, 1, 3)]}), "end_turn", "listed twice"),
    (json.dumps({"specifications": [
        {"number": n, "reasoning": "r", "remove": False, "reason": ""} for n in (1, 2, 4)]}), "end_turn", "outside 1..3"),
])
def test_interpret_fails_every_member_of_an_unusable_answer(screen_step, text, stop_reason, fragment):
    batch = batch_of(screen_step, [0, 1, 2])
    outcomes = screen_step.interpret(batch, ModelReply(text, stop_reason))
    assert list(outcomes) == batch.document_ids
    for outcome in outcomes.values():
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
        assert fragment in outcome.reason
    assert screen_step.failed_batch_line(batch, next(iter(outcomes.values()))).startswith(
        "  screen_round0_batch0000: unusable answer (")


def test_a_thrown_call_prints_todays_line_and_both_kinds_count_towards_the_systemic_stop(screen_step):
    batch = batch_of(screen_step, [0])
    thrown = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    assert screen_step.failed_batch_line(batch, thrown) == (
        "  screen_round0_batch0000: API call failed (RuntimeError: boom) — will retry on resume")
    assert screen_step.policy.failure_kinds_counting_towards_the_systemic_stop == {
        FailureKind.THROWN_CALL, FailureKind.UNUSABLE_ANSWER}


def test_request_has_the_arguments_the_screen_always_sent(screen_step):
    request = screen_step.build_request(batch_of(screen_step, [0, 1]))
    assert isinstance(request, ModelRequest)
    assert (request.stage, request.item_id, request.model, request.max_tokens) == (
        "layer1_screen", "screen_round0_batch0000", "screen-model", 8000)
    assert request.cache_system is False and request.temperature is None
    assert request.json_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] == [1, 2]
    assert "tone: tone 0" in request.user_message and "tone: tone 1" in request.user_message


# ---------------------------------------------------------------- the batch runner

def run_batches(screen_step, tmp_path, batches, send, checkpoint=None, workers=1):
    log = DocumentEventLog(tmp_path / "document_events.jsonl")
    checkpoint = checkpoint or utils.Checkpoint(tmp_path / "_screen_checkpoint.json")
    written = []
    runner = BatchStepRunner(screen_step, log, checkpoint, lambda batch, outcomes: written.append(batch.batch_id),
                             workers=workers, send=send)
    return runner, log, checkpoint, written


def test_a_failed_batch_is_not_written_or_checkpointed_and_stops_the_run_with_its_message(screen_step, tmp_path, capsys):
    batches = [batch_of(screen_step, [0, 1], "screen_round3_batch0000"), batch_of(screen_step, [2, 3], "screen_round3_batch0001")]

    def send(request):
        if request.item_id.endswith("0001"):
            raise RuntimeError("boom")
        return ModelReply(screen_answer(2), "end_turn")

    runner, log, checkpoint, written = run_batches(screen_step, tmp_path, batches, send)
    with pytest.raises(SystemExit) as stop:
        runner.run(batches)
    assert str(stop.value) == ("layer1_screen: 1 batch(es) of round 0 failed; the next round cannot be dealt "
                               "until the round is complete — --resume retries them.")
    assert written == ["screen_round3_batch0000"]
    assert checkpoint.is_done("screen_round3_batch0000") and not checkpoint.is_done("screen_round3_batch0001")
    output = capsys.readouterr().out
    assert "  screen_round3_batch0001: API call failed (RuntimeError: boom) — will retry on resume" in output
    assert "  Screened screen_round3_batch0000 (1/2): 2 hands, 0 extremely implausible," in output


@pytest.mark.parametrize("send", [
    lambda request: (_ for _ in ()).throw(RuntimeError("down")),
    lambda request: ModelReply("garbage", "end_turn"),
])
def test_every_pending_batch_failing_is_systemic_whether_thrown_or_unusable(screen_step, tmp_path, send):
    batches = [batch_of(screen_step, [0, 1])]
    runner, _, checkpoint, written = run_batches(screen_step, tmp_path, batches, send)
    with pytest.raises(SystemExit) as stop:
        runner.run(batches)
    assert str(stop.value) == ScreenStep.systemic_stop_message
    assert written == [] and not checkpoint.is_done("screen_round0_batch0000")


def test_events_are_appended_per_hand_with_the_batch_requests_hash_and_finished_batches_are_not_called_again(
        screen_step, tmp_path):
    batches = [batch_of(screen_step, [0, 1, 2], "screen_round0_batch0000"), batch_of(screen_step, [3, 4, 5], "screen_round0_batch0001")]
    calls = []

    def send(request):
        calls.append(request.item_id)
        return ModelReply(screen_answer(3, remove={1}), "end_turn")

    runner, log, checkpoint, _ = run_batches(screen_step, tmp_path, batches, send)
    runner.run(batches)
    events = read_document_events(log.path)
    assert [(e["document_id"], e["step"], e["outcome"], e["stage"]) for e in events] == [
        (f"matrix_{i:06d}", "screen", "removed" if i % 3 == 0 else "accepted", "layer1_screen") for i in range(6)]
    assert events[0]["request_hash"] == screen_step.build_request(batches[0]).request_hash()
    assert events[3]["request_hash"] == screen_step.build_request(batches[1]).request_hash()
    assert events[0]["reason_code"] == "extremely_implausible"
    runner.run(batches)   # a resume: every batch is checkpointed
    assert calls == ["screen_round0_batch0000", "screen_round0_batch0001"]
    assert len(read_document_events(log.path)) == 6


# ---------------------------------------------------------------- the deal and the redeal

@pytest.fixture
def deal_inputs(prompts_sdf):
    template = utils.sdf_template_path(prompts_sdf, 1).read_text(encoding="utf-8")
    values, weights, value_ids = cp.load_variables(prompts_sdf / cp.DEFAULT_VARIABLES.name)
    return template, values, weights, cp.load_deal_rules(prompts_sdf / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)


def test_the_deal_creates_one_round_zero_hand_per_document_keyed_by_hand_id(deal_inputs):
    template, values, weights, rules = deal_inputs
    outcomes = DealStep(template, values, weights, 6, 3, rules).apply([])
    assert list(outcomes) == [f"matrix_{i:06d}" for i in range(6)]
    assert all(isinstance(o, Created) and o.new_state["round"] == 0 for o in outcomes.values())
    expected = cp.hand_records(cp.deal_hands(template, values, weights, 6, 3, rules))
    assert [o.new_state["variables"] for o in outcomes.values()] == [h["variables"] for h in expected]
    assert [list(o.new_state) for o in outcomes.values()] == [["prompt_id", "round", "variables"]] * 6


def test_the_redeal_removes_the_screened_out_hands_and_creates_new_ones_numbered_on(deal_inputs):
    template, values, weights, rules = deal_inputs
    dealt = [o.new_state for o in DealStep(template, values, weights, 12, 0, rules).apply([]).values()]
    removed_ids = ["matrix_000002", "matrix_000007", "matrix_000009"]
    step = RedealStep(0, removed_ids, {pid: f"why {pid}" for pid in removed_ids}, 0, rules)
    outcomes = step.apply(dealt)
    assert [pid for pid, o in outcomes.items() if isinstance(o, Removed)] == removed_ids
    assert outcomes["matrix_000002"] == Removed("screened_out", "why matrix_000002")
    created = {pid: o.new_state for pid, o in outcomes.items() if isinstance(o, Created)}
    assert list(created) == ["matrix_000012", "matrix_000013", "matrix_000014"]
    assert all(h["round"] == 1 for h in created.values())
    by_id = {h["prompt_id"]: h for h in dealt}
    for axis in dealt[0]["variables"]:  # every column keeps exactly its cards
        assert sorted(h["variables"][axis] for h in created.values()) == sorted(by_id[p]["variables"][axis] for p in removed_ids)
    assert outcomes == RedealStep(0, removed_ids, {pid: f"why {pid}" for pid in removed_ids}, 0, rules).apply(dealt)


def test_cards_that_cannot_be_redealt_are_dropped_with_nothing_created(deal_inputs):
    template, values, weights, rules = deal_inputs
    hands = [o.new_state for o in DealStep(template, values, weights, 1, 0, rules).apply([]).values()]
    outcomes = RedealStep(0, ["matrix_000000"], {"matrix_000000": "no"}, 0, rules).apply(hands)
    assert outcomes == {"matrix_000000": Removed("screened_out_cards_dropped", "no")}


# ---------------------------------------------------------------- the layer: loop, rounds, events

def test_the_loop_ends_after_the_round_limit_and_the_leftover_hands_are_dropped(
        screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_redeal_rounds"] = 2
    stub_claude(remove_the_first_three_hands_of_every_batch)
    screen.run(screen_config, prompts_sdf, stage_dir)
    rounds = utils.load_jsonl(stage_dir / "screen_rounds.jsonl")
    assert [r["round"] for r in rounds] == [0, 1, 2]
    summary = json.loads((stage_dir / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["leftover_dropped"] == len(rounds[2]["prompt_ids"]) > 0


def test_deal_round_and_batch_plan_events_match_the_stage_files(screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    events = events_of(stage_dir)
    folded_plan = fold_round_and_batch_plan_events(events)
    rounds = utils.load_jsonl(stage_dir / "screen_rounds.jsonl")
    assert folded_plan["rounds"] == {r["round"]: {"prompt_ids": r["prompt_ids"], "note": r["note"]} for r in rounds}
    batches = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    assert folded_plan["batches_by_round"] == {
        number: [b for b in batches if b["round"] == number] for number in sorted({b["round"] for b in batches})}
    finished = folded_plan["corpus_steps_finished"]
    assert [(f["step"], f["round"]) for f in finished] == [("deal", 0), ("redeal", 1)]
    assert finished[0]["created_prompt_ids"] == rounds[0]["prompt_ids"]
    assert finished[1]["created_prompt_ids"] == rounds[1]["prompt_ids"] and finished[1]["cards_dropped"] is False
    # every round's events come before that round's first screen outcome event
    sequence = [e["event"] if e["event"] != "document_step_outcome" else e["step"] for e in events]
    assert sequence[:3] == ["round_dealt", "corpus_step_finished", "screen_batches_planned"]
    assert sequence.index("screen") > sequence.index("screen_batches_planned")


def test_screen_events_fold_to_the_judgments_and_redeal_removes_each_screened_out_hand(
        screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    folded = fold_document_events(events_of(stage_dir))
    judgments = utils.load_jsonl(stage_dir / "screen_judgments.jsonl")
    rejected = screen.load_rejections(stage_dir)
    assert len(judgments) == len([d for d in folded.values() if "screen" in d.latest_outcome_by_step])
    for record in judgments:
        outcome, facts = folded[record["prompt_id"]].latest_outcome_by_step["screen"]
        if record["remove"]:
            assert outcome == "removed" and facts["detail"]["reason"] == record["reason"]
        else:
            assert outcome == "accepted" and facts["number"] == record["number"]
    assert {pid for pid, d in folded.items() if "redeal" in d.removal_by_step} == set(rejected)
    assert all(folded[pid].removal_by_step["redeal"]["reason_code"] == "screened_out" for pid in rejected)


def test_a_removal_that_cannot_be_redealt_is_recorded_as_dropped_cards(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["n_prompts"] = 1
    stub_claude(remove_the_first_hand_of_every_batch)
    screen.run(screen_config, prompts_sdf, stage_dir)
    plan = fold_round_and_batch_plan_events(events_of(stage_dir))
    assert plan["rounds"][1]["prompt_ids"] == [] and "could not be redealt" in plan["rounds"][1]["note"]
    assert plan["corpus_steps_finished"][-1] == {
        "step": "redeal", "round": 1, "removed_prompt_ids": ["matrix_000000"], "created_prompt_ids": [], "cards_dropped": True}
    folded = fold_document_events(events_of(stage_dir))
    assert folded["matrix_000000"].removal_by_step["redeal"]["reason_code"] == "screened_out_cards_dropped"


def test_resume_repeats_the_identical_rounds_with_zero_calls_for_finished_batches(
        screen_config, prompts_sdf, stage_dir, stub_claude):
    def dies_in_round_one(user_message, **kw):
        if "round1" in kw["item_id"]:
            return "not json"
        return remove_first_hand_of_round_zero_batches(user_message, **kw)

    stub_claude(dies_in_round_one)
    with pytest.raises(SystemExit):
        screen.run(screen_config, prompts_sdf, stage_dir)
    events_before = events_of(stage_dir)
    batches_before = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    assert [e["event"] for e in events_before if e["event"] != "document_step_outcome"] == [
        "round_dealt", "corpus_step_finished", "screen_batches_planned",
        "round_dealt", "corpus_step_finished", "screen_batches_planned"]
    resumed_calls = stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    assert resumed_calls and all("round1" in c["item_id"] for c in resumed_calls)
    assert utils.load_jsonl(stage_dir / "screen_batches.jsonl")[:len(batches_before)] == batches_before
    plan_events_after = [e for e in events_of(stage_dir) if e["event"] != "document_step_outcome"]
    # the round 0 and round 1 plans and the round 1 deal are reused, not recorded again
    assert [e["event"] for e in plan_events_after] == [e["event"] for e in events_before if e["event"] != "document_step_outcome"]
    # round 1's failed attempt left per-hand failed events, and the resume's acceptance resolves them
    folded = fold_document_events(events_of(stage_dir))
    assert all("screen" not in d.unresolved_failure_by_step for d in folded.values())


def test_a_completed_screen_and_a_disabled_screen_append_no_events(screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir)
    count = len(events_of(stage_dir))
    calls = stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir)
    assert calls == [] and len(events_of(stage_dir)) == count
    disabled = stage_dir.parent / "disabled" / "layer1"
    disabled.mkdir(parents=True)
    screen_config["sdf"]["screen_enabled"] = False
    assert screen.run(screen_config, prompts_sdf, disabled) is None
    assert not (disabled.parent / "document_events.jsonl").exists()
