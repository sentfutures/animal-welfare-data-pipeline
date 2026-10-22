"""Behavior tests for the layer-1 implausibility screen with redeal rounds, driven
through layer1_screen.run() and screen_hands() with a stubbed shared.api.call_claude
(no test here may reach the API; see conftest). The real screen template, real
variables and real deal rules stay in play.

Money paths: parse happy path, malformed/truncated/out-of-range answers are not
checkpointed, resume after a completed screen makes zero calls, rounds and batches
are stable across resume. Invalidating bugs: a redeal that changes a column's card
counts, touches an accepted hand, or differs between runs of the same seed.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from evals import audit_sdf
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_screen as screen
from shared import utils

REPO_ROOT = Path(__file__).resolve().parent.parent


def numbers_in(user_message):
    """(number, text) for every <hand> block the screen prompt shows."""
    return [(int(n), body) for n, body in re.findall(
        r'<hand number="(\d+)">\n(.*?)\n</hand>', user_message, re.DOTALL)]


def screen_answer(count, remove=(), reasons=None, **overrides):
    """A schema-shaped answer for a batch of ``count`` hands."""
    entries = [
        {"number": n, "reasoning": f"Considered hand {n}.", "remove": n in remove,
         "reason": (reasons or {}).get(n, "No realistic document could satisfy this hand.") if n in remove else ""}
        for n in range(1, count + 1)
    ]
    return json.dumps({"specifications": entries, **overrides})


def no_removals(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)))


def remove_first_hand_of_round_zero_batches(user_message, **kw):
    """Removes hand 1 of every round-0 batch; keeps everything in later rounds."""
    count = len(numbers_in(user_message))
    return screen_answer(count, remove={1} if "round0" in kw["item_id"] else ())


def remove_the_first_hand_of_every_batch(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)), remove={1})


def remove_the_first_three_hands_of_every_batch(user_message, **kw):
    return screen_answer(len(numbers_in(user_message)), remove={1, 2, 3})


@pytest.fixture
def stage_dir(tmp_path):
    directory = tmp_path / "layer1"
    directory.mkdir()
    return directory


@pytest.fixture
def screen_config(tiny_config):
    tiny_config["sdf"]["n_prompts"] = 25
    tiny_config["sdf"]["screen_batch_size"] = 10
    return tiny_config


def column_counts(hands):
    columns = {}
    for hand in hands:
        for axis, value in hand["variables"].items():
            columns.setdefault(axis, Counter())[value] += 1
    return columns


def load_dealt_hands(stage_dir):
    return [h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl") if h["round"] == 0]


# --- parsing and the public function ---

def test_parse_valid_answer_keeps_every_judgment_in_number_order():
    entries = [
        {"number": 2, "reasoning": "No such practice there.", "remove": True, "reason": "Impossible premise."},
        {"number": 1, "reasoning": "Fine.", "remove": False, "reason": ""},
    ]
    judgments = screen.parse_screen_answer(json.dumps({"specifications": entries}), 2)
    assert [(j.number, j.index, j.remove, j.reason) for j in judgments] == [
        (1, 0, False, ""), (2, 1, True, "Impossible premise.")]
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


def test_screen_hands_maps_numbers_to_judgments_through_the_public_function(stub_claude, prompts_sdf):
    def removes_the_second(user_message, **kw):
        return screen_answer(len(numbers_in(user_message)), remove={2})

    calls = stub_claude(removes_the_second)
    prompts = screen.load_screen_prompts(prompts_sdf)
    judgments = screen.screen_hands(["lines one", "lines two", "lines three"], prompts=prompts, model="m", item_id="b0")
    assert [(j.number, j.remove) for j in judgments] == [(1, False), (2, True), (3, False)]
    assert judgments[1].reason == "No realistic document could satisfy this hand."
    assert all(j.reasoning for j in judgments)
    assert calls[0]["stage"] == "layer1_screen" and calls[0]["model"] == "m"
    sent_schema = calls[0]["json_schema"]  # schema-constrained, enum matching this batch's size
    assert sent_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"] == [1, 2, 3]
    assert 'number="3"' in calls[0]["user_message"] and 'number="4"' not in calls[0]["user_message"]
    assert "lines two" in calls[0]["user_message"]
    assert calls[0]["system_prompt"]


def test_no_model_facing_prose_lives_in_the_python_module():
    source = (REPO_ROOT / "sdf_pipeline" / "layer1_screen.py").read_text(encoding="utf-8")
    for phrase in ("Judge against how the world actually is", "These are not reasons", "competent planner"):
        assert phrase not in source


def test_screen_prompt_wording(prompts_sdf):
    prompts = screen.load_screen_prompts(prompts_sdf)
    system, user = screen.build_screen_prompt(prompts, ["a", "b", "c"])
    text = (system + user).lower()
    assert "extremely implausible" in text and "incoherent" not in text
    assert "none" in user.lower() and "several" in user.lower()  # permission to remove none or several
    assert "no target number" in user.lower()
    assert "reasoning first" in user
    assert "not reasons to remove" in user
    assert "budgerigars" in user and "encyclopedia" in user  # the two audit-derived illustrations
    assert not re.search(r"\b(at least|at most|about|roughly|around) (one|two|three|\d+)\b",
                         user, re.IGNORECASE)
    assert "{" not in user and "allowed" not in text


def test_truncated_answer_is_rejected(stub_claude, prompts_sdf):
    stub_claude(lambda user_message, **kw: (screen_answer(1), "max_tokens"))
    with pytest.raises(screen.ScreenAnswerError, match="truncated"):
        screen.screen_hands(["a"], prompts=screen.load_screen_prompts(prompts_sdf))


# --- the stage: batching, mapping, rounds ---

def test_screen_shows_the_planner_variable_lines_in_numbered_batches_of_ten(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    calls = stub_claude(no_removals)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)

    assert len(accepted) == 25
    assert sorted(len(numbers_in(c["user_message"])) for c in calls) == [5, 10, 10]
    assert all(c["stage"] == "layer1_screen" and c["temperature"] is None for c in calls)
    # what the model saw for a hand is exactly the text the planner sees
    render = cp.make_variable_line_renderer(prompts_sdf, 137)
    hands = {h["prompt_id"]: h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl")}
    batches = utils.load_jsonl(stage_dir / "screen_batches.jsonl")
    for call in calls:
        batch = next(b for b in batches if b["batch_id"] == call["item_id"])
        for (number, body), pid in zip(numbers_in(call["user_message"]), batch["prompt_ids"]):
            assert body == render(hands[pid]).strip()
    assert not any("matrix_0" in c["user_message"] for c in calls)  # numbers only, never ids


def test_numbers_map_back_to_the_right_hands_and_rejections_carry_the_round(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    batches = [b for b in utils.load_jsonl(stage_dir / "screen_batches.jsonl") if b["round"] == 0]
    first_of_each_batch = {b["prompt_ids"][0] for b in batches}
    rejections = utils.load_jsonl(stage_dir / "screen_rejections.jsonl")
    assert {r["prompt_id"] for r in rejections} == first_of_each_batch
    assert all(r["round"] == 0 and r["reason"] and r["variable_lines"] and r["variables"] for r in rejections)
    judgments = utils.load_jsonl(stage_dir / "screen_judgments.jsonl")
    assert all(j["reasoning"] and j["batch_id"].startswith("screen_round") for j in judgments)
    assert {j["prompt_id"] for j in judgments if j["remove"]} >= first_of_each_batch


def test_out_of_range_numbers_are_rejected_and_not_checkpointed(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    stub_claude(lambda user_message, **kw: json.dumps({"specifications": [
        valid_entry(11)] * len(numbers_in(user_message))}))
    with pytest.raises(SystemExit, match="every pending screen call failed"):
        screen.run(screen_config, prompts_sdf, stage_dir)
    assert utils.Checkpoint(stage_dir / "_screen_checkpoint.json").done_count == 0
    assert not (stage_dir / "screen_judgments.jsonl").exists()
    assert not (stage_dir / "screened_hands.jsonl").exists()


@pytest.mark.parametrize("make_bad_answer", [
    lambda count: screen_answer(count - 1),                              # wrong count
    lambda count: screen_answer(count + 1),                              # too many entries
    lambda count: screen_answer(count, remove={1}, reasons={1: ""}),    # remove without a reason
    lambda count: "{not json",                                           # malformed JSON
    lambda count: (screen_answer(count), "max_tokens"),                  # truncated
])
def test_unusable_answers_are_not_checkpointed(screen_config, prompts_sdf, stage_dir, stub_claude, make_bad_answer):
    stub_claude(lambda user_message, **kw: make_bad_answer(len(numbers_in(user_message))))
    with pytest.raises(SystemExit, match="every pending screen call failed"):
        screen.run(screen_config, prompts_sdf, stage_dir)
    assert utils.Checkpoint(stage_dir / "_screen_checkpoint.json").done_count == 0
    assert not (stage_dir / "screen_rejections.jsonl").exists()


def test_one_failed_batch_stops_before_any_redeal_and_resume_retries_exactly_that_batch(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    failing = {}

    def dispatcher(user_message, **kw):
        failing.setdefault("id", kw["item_id"])  # the first batch to be called fails once
        if kw["item_id"] == failing["id"] and not failing.get("retried"):
            failing["retried"] = True
            return "I think number 2 might be a problem."  # not JSON
        return remove_first_hand_of_round_zero_batches(user_message, **kw)

    stub_claude(dispatcher)
    with pytest.raises(SystemExit, match="1 batch"):
        screen.run(screen_config, prompts_sdf, stage_dir)
    assert len(utils.load_jsonl(stage_dir / "screen_rounds.jsonl")) == 1  # no redeal happened
    resumed_calls = stub_claude(remove_first_hand_of_round_zero_batches)
    screen.run(screen_config, prompts_sdf, stage_dir)
    assert failing["id"] in [c["item_id"] for c in resumed_calls]
    assert [c["item_id"] for c in resumed_calls].count(failing["id"]) == 1


def test_batches_are_written_before_any_call(screen_config, prompts_sdf, stage_dir, stub_claude):
    seen_at_first_call = []

    def dispatcher(user_message, **kw):
        seen_at_first_call.append(len(utils.load_jsonl(stage_dir / "screen_batches.jsonl")))
        return no_removals(user_message)

    stub_claude(dispatcher)
    screen.run(screen_config, prompts_sdf, stage_dir)
    assert seen_at_first_call == [3, 3, 3]


def test_model_falls_back_from_screen_to_plan_to_global(screen_config, prompts_sdf, tmp_path, stub_claude):
    screen_config["sdf"]["n_prompts"] = 3
    calls = stub_claude(no_removals)
    for name, knobs, expected in (
        ("first", {}, None),
        ("second", {"plan_model": "plan-model-x"}, "plan-model-x"),
        ("third", {"plan_model": "plan-model-x", "screen_model": "screen-model-x"}, "screen-model-x"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        screen_config["sdf"].update(knobs)
        screen.run(screen_config, prompts_sdf, directory)
        assert calls[-1]["model"] == expected


def test_batch_size_knob(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_batch_size"] = 4
    screen_config["sdf"]["n_prompts"] = 10
    calls = stub_claude(no_removals)
    screen.run(screen_config, prompts_sdf, stage_dir)
    assert len(calls) == 3  # 4 + 4 + 2


def test_batches_are_seeded_and_input_order_independent(screen_config, prompts_sdf, tmp_path, stub_claude):
    stub_claude(no_removals)
    batches = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        directory = tmp_path / name
        directory.mkdir()
        screen_config["sdf"]["seed"] = seed
        screen.run(screen_config, prompts_sdf, directory)
        batches.append(utils.load_jsonl(directory / "screen_batches.jsonl"))
    assert batches[0] == batches[1]
    assert batches[0] != batches[2]


# --- redeal rounds ---

def test_removed_hands_are_redealt_and_every_columns_counts_stay_exact(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    # The first hand of each round-0 batch is removed; the redealt hands all pass.
    stub_claude(remove_first_hand_of_round_zero_batches)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    dealt = load_dealt_hands(stage_dir)
    assert len(dealt) == 25 and len(accepted) == 25   # 3 removed, 3 redealt and accepted
    assert column_counts(accepted) == column_counts(dealt)
    summary = json.loads((stage_dir / "screen_summary.json").read_text())
    assert summary["leftover_dropped"] == 0
    assert [(r["round"], r["hands"], r["removed"]) for r in summary["rounds"]] == [(0, 25, 3), (1, 3, 0)]
    # redealt hands got fresh ids, and the removed ones are not accepted
    removed_ids = set(screen.load_rejections(stage_dir))
    assert not removed_ids & {h["prompt_id"] for h in accepted}
    redealt = [h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl") if h["round"] == 1]
    assert all(h["prompt_id"] not in {d["prompt_id"] for d in dealt} for h in redealt)


def test_redeal_never_touches_hands_already_accepted(screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(remove_first_hand_of_round_zero_batches)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    dealt = {h["prompt_id"]: h["variables"] for h in load_dealt_hands(stage_dir)}
    removed_ids = set(screen.load_rejections(stage_dir))
    for hand in accepted:
        if hand["prompt_id"] in dealt:
            assert hand["variables"] == dealt[hand["prompt_id"]]
    assert {pid for pid in dealt if pid not in removed_ids} <= {h["prompt_id"] for h in accepted}


def test_redeal_applies_deal_rule_repair_only_among_the_new_hands(prompts_sdf):
    values, weights, value_ids = cp.load_variables(prompts_sdf / "variables.txt")
    rules = cp.load_deal_rules(prompts_sdf / "deal_constraints.txt", value_ids)
    template = (prompts_sdf / "layer1.txt").read_text(encoding="utf-8")
    hands = [{"prompt_id": f"matrix_{i:06d}", "variables": v}
             for i, v in enumerate(cp.deal_hands(template, values, weights, 60, 0, rules))]
    removed = hands[:12]
    new_hands = screen.redeal_removed_hands(removed, 1, 0, rules)
    assert new_hands is not None and len(new_hands) == 12
    assert column_counts([{"variables": h} for h in new_hands]) == column_counts(removed)
    from shared import deal_constraints
    assert not any(deal_constraints.violated_rule_names(rules, hand) for hand in new_hands)


def test_redeal_is_deterministic_per_seed_and_round(prompts_sdf):
    values, weights, value_ids = cp.load_variables(prompts_sdf / "variables.txt")
    rules = cp.load_deal_rules(prompts_sdf / "deal_constraints.txt", value_ids)
    template = (prompts_sdf / "layer1.txt").read_text(encoding="utf-8")
    hands = [{"prompt_id": f"matrix_{i:06d}", "variables": v}
             for i, v in enumerate(cp.deal_hands(template, values, weights, 40, 0, rules))][:10]
    same = [screen.redeal_removed_hands(hands, 1, 0, rules) for _ in range(2)]
    assert same[0] == same[1]
    assert screen.redeal_removed_hands(hands, 2, 0, rules) != same[0]
    assert screen.redeal_removed_hands(hands, 1, 1, rules) != same[0]


def test_a_lone_removed_hand_cannot_be_redealt_into_a_different_hand(prompts_sdf):
    hand = {"prompt_id": "matrix_000000", "variables": {"tone": "a", "culture": "b"}}
    assert screen.redeal_removed_hands([hand], 1, 0, []) is None


def test_screen_terminates_after_the_maximum_rounds_and_reports_leftovers(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    screen_config["sdf"]["screen_redeal_rounds"] = 2
    calls = stub_claude(remove_the_first_three_hands_of_every_batch)  # every round removes hands 1-3 of each batch
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    summary = json.loads((stage_dir / "screen_summary.json").read_text())
    assert [r["round"] for r in summary["rounds"]] == [0, 1, 2]       # original deal + 2 redeal rounds
    assert summary["leftover_dropped"] > 0
    assert len(accepted) == 25 - summary["leftover_dropped"] == summary["accepted"]
    assert len(summary["leftover_prompt_ids"]) == summary["leftover_dropped"]
    assert not set(summary["leftover_prompt_ids"]) & {h["prompt_id"] for h in accepted}
    # no calls after the last round; every call belongs to a round in the summary
    assert max(int(re.search(r"round(\d+)", c["item_id"]).group(1)) for c in calls) == 2
    # leftover cards are exactly the shortfall in the column counts
    shortfall = column_counts(load_dealt_hands(stage_dir))
    leftover_hands = [h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl")
                      if h["prompt_id"] in set(summary["leftover_prompt_ids"])]
    final = column_counts(accepted)
    for axis, counter in shortfall.items():
        assert counter - final[axis] == column_counts(leftover_hands)[axis]


def test_removed_hands_that_cannot_be_redealt_are_dropped_and_recorded(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    stub_claude(remove_the_first_hand_of_every_batch)  # round 1 has one hand: no different deal exists
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    summary = json.loads((stage_dir / "screen_summary.json").read_text())
    assert summary["leftover_dropped"] == 1 and len(accepted) == 24
    assert "could not be redealt" in summary["rounds"][-1]["note"] and summary["rounds"][-1]["hands"] == 0


def test_zero_redeal_rounds_drops_removed_hands_without_redealing(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    screen_config["sdf"]["screen_redeal_rounds"] = 0
    stub_claude(remove_first_hand_of_round_zero_batches)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    assert len(accepted) == 22 and len(utils.load_jsonl(stage_dir / "screen_rounds.jsonl")) == 1


def test_redeal_rounds_are_saved_before_they_are_screened(screen_config, prompts_sdf, stage_dir, stub_claude):
    seen = []

    def dispatcher(user_message, **kw):
        seen.append((kw["item_id"], len(utils.load_jsonl(stage_dir / "screen_rounds.jsonl")),
                     len(utils.load_jsonl(stage_dir / "screen_hands.jsonl"))))
        return remove_first_hand_of_round_zero_batches(user_message, **kw)

    stub_claude(dispatcher)
    screen.run(screen_config, prompts_sdf, stage_dir)
    round_one = [entry for entry in seen if "round1" in entry[0]]
    assert round_one and all(rounds == 2 and hands == 28 for _, rounds, hands in round_one)


def test_resume_mid_redeal_repeats_the_identical_round(screen_config, prompts_sdf, stage_dir, stub_claude, tmp_path):
    # An uninterrupted run, for comparison.
    stub_claude(remove_first_hand_of_round_zero_batches)
    whole_dir = tmp_path / "whole"
    whole_dir.mkdir()
    whole = screen.run(screen_config, prompts_sdf, whole_dir)

    # A run that dies on its first round-1 call, then resumes.
    def dies_in_round_one(user_message, **kw):
        if "round1" in kw["item_id"]:
            return "not json"
        return remove_first_hand_of_round_zero_batches(user_message, **kw)

    stub_claude(dies_in_round_one)
    with pytest.raises(SystemExit):
        screen.run(screen_config, prompts_sdf, stage_dir)
    redeal_before = [h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl") if h["round"] == 1]
    resumed_calls = stub_claude(remove_first_hand_of_round_zero_batches)
    resumed = screen.run(screen_config, prompts_sdf, stage_dir)
    assert all("round1" in c["item_id"] for c in resumed_calls)  # round 0 is not re-screened
    assert [h for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl") if h["round"] == 1] == redeal_before
    assert resumed == whole


def test_a_redeal_interrupted_before_its_round_record_is_dealt_again_without_orphan_hands(
    screen_config, prompts_sdf, stage_dir, stub_claude, tmp_path
):
    whole_dir = tmp_path / "whole"
    whole_dir.mkdir()
    stub_claude(remove_first_hand_of_round_zero_batches)
    whole = screen.run(screen_config, prompts_sdf, whole_dir)

    def dies_in_round_one(user_message, **kw):
        return "not json" if "round1" in kw["item_id"] else remove_first_hand_of_round_zero_batches(user_message, **kw)

    stub_claude(dies_in_round_one)
    with pytest.raises(SystemExit):
        screen.run(screen_config, prompts_sdf, stage_dir)
    rounds_path = stage_dir / "screen_rounds.jsonl"
    utils.save_jsonl(utils.load_jsonl(rounds_path)[:1], rounds_path)   # the redeal's hands are saved, its round record is not
    (stage_dir / "screen_batches.jsonl").unlink()
    (stage_dir / "_screen_checkpoint.json").unlink()
    stub_claude(remove_first_hand_of_round_zero_batches)
    resumed = screen.run(screen_config, prompts_sdf, stage_dir)
    hand_ids = [h["prompt_id"] for h in utils.load_jsonl(stage_dir / "screen_hands.jsonl")]
    assert len(hand_ids) == len(set(hand_ids))
    assert len(resumed) == len({h["prompt_id"] for h in resumed})
    assert column_counts(resumed) == column_counts(whole)


def test_a_round_whose_batches_were_only_partly_saved_screens_every_hand(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    stub_claude(lambda user_message, **kw: "not json")
    with pytest.raises(SystemExit):
        screen.run(screen_config, prompts_sdf, stage_dir)
    batches_path = stage_dir / "screen_batches.jsonl"
    utils.save_jsonl(utils.load_jsonl(batches_path)[:1], batches_path)   # interrupted while the batches were written
    calls = stub_claude(no_removals)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    screened = sum(len(numbers_in(call["user_message"])) for call in calls)
    assert screened == 25 and len(accepted) == 25


def test_a_round_zero_deal_interrupted_while_writing_hands_leaves_no_duplicate_hands(
    screen_config, prompts_sdf, stage_dir, stub_claude
):
    stub_claude(lambda user_message, **kw: "not json")
    with pytest.raises(SystemExit):
        screen.run(screen_config, prompts_sdf, stage_dir)
    hands_path = stage_dir / "screen_hands.jsonl"
    utils.save_jsonl(utils.load_jsonl(hands_path)[:5], hands_path)    # only part of the original deal was written
    (stage_dir / "screen_rounds.jsonl").unlink()
    (stage_dir / "screen_batches.jsonl").unlink()
    stub_claude(no_removals)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    hand_ids = [h["prompt_id"] for h in utils.load_jsonl(hands_path)]
    assert len(hand_ids) == len(set(hand_ids)) == 25
    assert len({h["prompt_id"] for h in accepted}) == len(accepted) == 25


def test_a_batch_screened_again_replaces_its_earlier_verdicts(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_redeal_rounds"] = 0
    stub_claude(remove_the_first_hand_of_every_batch)
    screen.run(screen_config, prompts_sdf, stage_dir)
    checkpoint_path = stage_dir / "_screen_checkpoint.json"
    checkpoint_path.unlink()          # the verdicts were saved but the batches were not checkpointed
    for name in ("screened_hands.jsonl", "screen_summary.json"):
        (stage_dir / name).unlink()
    stub_claude(no_removals)
    accepted = screen.run(screen_config, prompts_sdf, stage_dir)
    judgments = utils.load_jsonl(stage_dir / "screen_judgments.jsonl")
    assert len(judgments) == len({j["prompt_id"] for j in judgments}) == 25
    assert screen.load_rejections(stage_dir) == {} and len(accepted) == 25


def test_resume_after_a_completed_screen_makes_zero_calls(screen_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(remove_first_hand_of_round_zero_batches)
    first = screen.run(screen_config, prompts_sdf, stage_dir)
    calls = stub_claude([])  # any call would exhaust the empty queue and fail
    again = screen.run(screen_config, prompts_sdf, stage_dir)
    assert calls == [] and again == first


def test_same_seed_gives_the_same_screen(screen_config, prompts_sdf, tmp_path, stub_claude):
    stub_claude(remove_first_hand_of_round_zero_batches)
    outcomes = []
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        outcomes.append(screen.run(screen_config, prompts_sdf, directory))
    assert outcomes[0] == outcomes[1]


def test_disabled_screen_makes_no_calls_and_writes_nothing(screen_config, prompts_sdf, stage_dir, stub_claude):
    screen_config["sdf"]["screen_enabled"] = False
    calls = stub_claude([])
    assert screen.run(screen_config, prompts_sdf, stage_dir) is None
    assert calls == [] and list(stage_dir.iterdir()) == []


def test_snapshot_without_the_screen_template_skips_the_screen(screen_config, tmp_path, stage_dir, stub_claude):
    calls = stub_claude([])
    old_snapshot = tmp_path / "old_prompts"
    old_snapshot.mkdir()
    assert screen.run(screen_config, old_snapshot, stage_dir) is None
    assert calls == []


# --- reporting ---

def write_screened_run(run_dir, hands, removed_ids, accepted_ids=None, rounds=None):
    stage_dir = run_dir / "layer1"
    stage_dir.mkdir(parents=True)
    utils.save_jsonl(hands, stage_dir / "screen_hands.jsonl")
    utils.save_jsonl(rounds or [{"round": 0, "prompt_ids": [h["prompt_id"] for h in hands], "note": ""}],
                     stage_dir / "screen_rounds.jsonl")
    utils.save_jsonl([{"batch_id": "screen_round0_batch0000", "round": 0,
                       "prompt_ids": [h["prompt_id"] for h in hands]}], stage_dir / "screen_batches.jsonl")
    utils.save_jsonl([{"prompt_id": pid, "round": 0, "reason": "r"} for pid in removed_ids],
                     stage_dir / "screen_rejections.jsonl")
    utils.Checkpoint(stage_dir / "_screen_checkpoint.json").mark_done("screen_round0_batch0000")
    if accepted_ids is not None:
        utils.save_jsonl([h for h in hands if h["prompt_id"] in accepted_ids], stage_dir / "screened_hands.jsonl")
        (stage_dir / "screen_summary.json").write_text(json.dumps({
            "leftover_prompt_ids": sorted(set(removed_ids))}))
    return stage_dir


def make_hands(count=40):
    """A quarter of the hands are skeptical, so a skeptical-heavy removal can exceed 2x."""
    return [
        {"prompt_id": f"matrix_{i:06d}", "round": 0,
         "variables": {"tone": "skeptical or displeased" if i % 4 == 1 else "neutral or journalistic",
                       "culture": "France"}}
        for i in range(count)
    ]


def test_removal_rate_rows_flag_values_removed_at_over_twice_the_corpus_rate(tmp_path):
    hands = make_hands()
    skeptical = [h["prompt_id"] for h in hands if h["variables"]["tone"].startswith("skeptical")][:4]
    neutral = [h["prompt_id"] for h in hands if h["variables"]["tone"].startswith("neutral")][:1]
    write_screened_run(tmp_path, hands, skeptical + neutral)
    rows = screen.removal_rate_rows(screen.load_screen_outputs(tmp_path))
    by_value = {(r["variable"], r["value"]): r for r in rows}
    skeptical_row = by_value[("tone", "skeptical or displeased")]
    assert (skeptical_row["screened"], skeptical_row["removed"], skeptical_row["rate"]) == (10, 4, 0.4)
    assert skeptical_row["corpus_rate"] == pytest.approx(5 / 40) and skeptical_row["flagged"]
    assert not by_value[("tone", "neutral or journalistic")]["flagged"]
    assert not by_value[("culture", "France")]["flagged"]
    assert rows[0]["flagged"]


def test_removal_rates_count_every_round(tmp_path):
    hands = make_hands(10)
    redealt = [{"prompt_id": "matrix_000010", "round": 1, "variables": hands[1]["variables"]}]
    rounds = [{"round": 0, "prompt_ids": [h["prompt_id"] for h in hands], "note": ""},
              {"round": 1, "prompt_ids": ["matrix_000010"], "note": ""}]
    stage_dir = write_screened_run(tmp_path, hands + redealt, ["matrix_000001", "matrix_000010"], rounds=rounds)
    utils.save_jsonl([{"batch_id": "screen_round0_batch0000", "round": 0, "prompt_ids": [h["prompt_id"] for h in hands]},
                      {"batch_id": "screen_round1_batch0000", "round": 1, "prompt_ids": ["matrix_000010"]}],
                     stage_dir / "screen_batches.jsonl")
    utils.Checkpoint(stage_dir / "_screen_checkpoint.json").mark_done("screen_round1_batch0000")
    outputs = screen.load_screen_outputs(tmp_path)
    cell = next(r for r in screen.removal_rate_rows(outputs) if r["value"] == "skeptical or displeased")
    assert (cell["screened"], cell["removed"]) == (4, 2)
    assert [(r["round"], r["hands"], r["removed"]) for r in screen.round_rows(outputs)] == [(0, 10, 1), (1, 1, 1)]


def test_share_shift_shows_leftover_hands_changing_the_shares(tmp_path):
    hands = make_hands(20)
    leftover = [h["prompt_id"] for h in hands if h["variables"]["tone"].startswith("skeptical")][:2]
    accepted = {h["prompt_id"] for h in hands} - set(leftover)
    write_screened_run(tmp_path, hands, leftover, accepted_ids=accepted)
    rows = screen.share_shift_rows(screen.load_screen_outputs(tmp_path))
    skeptical = next(r for r in rows if r["value"] == "skeptical or displeased")
    assert (skeptical["dealt"], skeptical["accepted"]) == (5, 3)
    assert skeptical["shift"] == pytest.approx(3 / 18 - 5 / 20)
    assert rows[0]["variable"] == "tone"  # largest shifts first


def test_load_screen_outputs_is_none_for_a_run_without_the_screen(tmp_path):
    (tmp_path / "layer1").mkdir()
    assert screen.load_screen_outputs(tmp_path) is None
    assert screen.load_screen_outputs(tmp_path / "does_not_exist") is None


def test_only_checkpointed_batches_count_as_screened(tmp_path):
    hands = make_hands(6)
    stage_dir = write_screened_run(tmp_path, hands, [])
    utils.save_jsonl([{"batch_id": "screen_round0_batch0000", "round": 0, "prompt_ids": ["matrix_000000", "matrix_000001"]},
                      {"batch_id": "screen_round0_batch0001", "round": 0, "prompt_ids": ["matrix_000002"]}],
                     stage_dir / "screen_batches.jsonl")
    outputs = screen.load_screen_outputs(tmp_path)
    assert outputs["screened_ids"] == {"matrix_000000", "matrix_000001"}
    assert (outputs["batches_done"], outputs["batches_total"]) == (1, 2)


def test_audit_reports_screen_removals_rounds_and_flags(tmp_path, capsys):
    hands = make_hands()
    skeptical = [h["prompt_id"] for h in hands if h["variables"]["tone"].startswith("skeptical")][:4]
    write_screened_run(tmp_path, hands, skeptical, accepted_ids={h["prompt_id"] for h in hands} - set(skeptical))
    report = {}
    audit_sdf.audit_screen(tmp_path, report)
    output = capsys.readouterr().out
    assert "4 of 40 screened" in output and "skeptical or displeased" in output and "BAD" in output
    assert "extremely implausible" in output and "round 0" in output.lower()
    assert report["screen"]["removed"] == 4 and report["screen"]["leftover_dropped"] == 4
    assert report["screen"]["flagged_values"] == ["tone = skeptical or displeased"]
    assert report["screen"]["rounds"][0]["hands"] == 40
    assert report["screen"]["share_shift"][0]["variable"] == "tone"


def test_audit_still_works_on_a_run_without_screen_files(tmp_path, capsys):
    (tmp_path / "layer1").mkdir()
    report = {}
    audit_sdf.audit_screen(tmp_path, report)
    assert "no screen files" in capsys.readouterr().out
    assert "screen" not in report
