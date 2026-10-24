"""Pure-code tests for the considerations assignment (linear programme); no API calls."""

import json
import random

import pytest

from sdf_pipeline import considerations_assignment as assignment
from sdf_pipeline.considerations_assignment import AssignmentLimits, assign_considerations, coverage_report
from sdf_pipeline.considerations_catalog import Consideration, ConsiderationCatalog, EPISTEMIC_LIST, MORAL_LIST


def make_catalog(moral=("M1", "M2"), epistemic=("E1", "E2"), digital=()):
    return ConsiderationCatalog(
        moral=tuple(Consideration(label, "gloss", 0.5, MORAL_LIST) for label in moral),
        epistemic=tuple(Consideration(label, "gloss", 0.5, EPISTEMIC_LIST, True) for label in digital)
        + tuple(Consideration(label, "gloss", 0.5, EPISTEMIC_LIST) for label in epistemic),
        digital_mind_categories=frozenset({"digital"}), skipped_resolutions=frozenset(),
    )


def documents(count, digital=()):
    return {f"doc_{i}": i in digital for i in range(count)}


def run(scores, catalog, targets, limits=AssignmentLimits(), seed=0, document_flags=None):
    flags = document_flags if document_flags is not None else {pid: False for pid in scores}
    return assign_considerations(scores, catalog, flags, limits, seed, targets)


def total(assigned):
    return sum(len(labels) for labels in assigned.values())


def test_entry_targets_and_per_document_list_caps_are_respected():
    catalog = make_catalog()
    scores = {f"doc_{i}": {"M1": 5, "M2": 4, "E1": 5, "E2": 4} for i in range(6)}
    assigned = run(scores, catalog, {"M1": 2, "M2": 1, "E1": 3, "E2": 1})
    for label, target in {"M1": 2, "M2": 1, "E1": 3, "E2": 1}.items():
        assert sum(label in labels for labels in assigned.values()) <= target
    for labels in assigned.values():
        assert sum(label.startswith("M") for label in labels) <= 1 and sum(label.startswith("E") for label in labels) <= 1
    assert total(assigned) == 7        # every slot filled


def test_pairs_below_the_minimum_assignable_score_are_never_assigned():
    catalog = make_catalog()
    scores = {"doc_0": {"M1": 2, "M2": 2, "E1": 1, "E2": 2}, "doc_1": {"M1": 3, "M2": 1, "E1": 2, "E2": 1}}
    targets = {"M1": 2, "M2": 2, "E1": 2, "E2": 2}
    assert run(scores, catalog, targets) == {"doc_0": [], "doc_1": ["M1"]}
    raised = run(scores, catalog, targets, AssignmentLimits(minimum_assignable_score=4))
    assert total(raised) == 0
    lowered = run(scores, catalog, targets, AssignmentLimits(minimum_assignable_score=2, score_weights={2: 1, 3: 1, 4: 3, 5: 9}))
    assert lowered["doc_0"] and total(lowered) > 1


def test_the_objective_trades_count_against_score_through_the_weights():
    # One moral entry per document: doc_0 -> M1 (5, weight 9) leaves doc_1 (only M1 allowed) empty;
    # the alternative fills both slots for 3 + 3 = 6 < 9.
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1",))
    scores = {"doc_0": {"M1": 5, "M2": 4, "E1": 1}, "doc_1": {"M1": 4, "M2": 1, "E1": 1}}
    targets = {"M1": 1, "M2": 1, "E1": 1}
    one_moral = AssignmentLimits(max_moral_per_document=1)
    assert run(scores, catalog, targets, one_moral) == {"doc_0": ["M1"], "doc_1": []}
    flat_weights = AssignmentLimits(max_moral_per_document=1, score_weights={3: 1, 4: 3, 5: 3})
    assert run(scores, catalog, targets, flat_weights) == {"doc_0": ["M2"], "doc_1": ["M1"]}


def test_higher_scores_win_at_equal_count():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))
    scores = {"doc_0": {"M1": 3, "E1": 1}, "doc_1": {"M1": 5, "E1": 1}, "doc_2": {"M1": 4, "E1": 1}}
    assert run(scores, catalog, {"M1": 1, "E1": 1})["doc_1"] == ["M1"]
    assert run(scores, catalog, {"M1": 2, "E1": 1})["doc_0"] == []


def test_the_assignment_is_deterministic_and_the_seed_only_breaks_ties():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))
    scores = {f"doc_{i}": {"M1": 4, "E1": 1} for i in range(12)}      # a 12-way tie for 3 slots
    first = run(scores, catalog, {"M1": 3, "E1": 1}, seed=7)
    assert run(scores, catalog, {"M1": 3, "E1": 1}, seed=7) == first
    winners = {seed: tuple(pid for pid, labels in run(scores, catalog, {"M1": 3, "E1": 1}, seed=seed).items() if labels)
               for seed in range(6)}
    assert len(set(winners.values())) > 1 and all(len(w) == 3 for w in winners.values())


def test_empty_input_and_no_allowed_pairs():
    catalog = make_catalog()
    assert run({}, catalog, {"M1": 1, "M2": 1, "E1": 1, "E2": 1}, document_flags={"doc_0": False}) == {"doc_0": []}
    assert assign_considerations({}, catalog, {}, AssignmentLimits(), 0) == {}
    assert run({"doc_0": {"M1": 1, "M2": 2, "E1": 1, "E2": 2}}, catalog, {"M1": 1, "M2": 1, "E1": 1, "E2": 1}) == {"doc_0": []}


def test_digital_minds_only_entries_go_only_to_digital_mind_documents():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",), digital=("D1",))
    scores = {"doc_0": {"M1": 1, "E1": 1, "D1": 5}, "doc_1": {"M1": 1, "E1": 1, "D1": 5}}
    flags = {"doc_0": False, "doc_1": True}
    assigned = assign_considerations(scores, catalog, flags, AssignmentLimits(), 0, {"M1": 1, "E1": 1, "D1": 2})
    assert assigned == {"doc_0": [], "doc_1": ["D1"]}


def test_default_targets_come_from_the_catalog_shares():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))      # share 0.5 each -> 2 of 4 documents
    scores = {f"doc_{i}": {"M1": 5, "E1": 5} for i in range(4)}
    assigned = assign_considerations(scores, catalog, documents(4), AssignmentLimits(), 0)
    assert sum("M1" in labels for labels in assigned.values()) == 2
    assert sum("E1" in labels for labels in assigned.values()) == 2


def test_caps_of_two_per_list_and_three_in_total_are_respected():
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1", "E2", "E3"))
    scores = {f"doc_{i}": {label: 5 for label in ("M1", "M2", "M3", "E1", "E2", "E3")} for i in range(2)}
    targets = {label: 2 for label in ("M1", "M2", "M3", "E1", "E2", "E3")}
    assigned = run(scores, catalog, targets, AssignmentLimits(3, 2, 2, 3, penalty_step_per_further_consideration_in_a_list=0))
    for labels in assigned.values():
        assert len(labels) == 3
        assert sum(label.startswith("M") for label in labels) <= 2 and sum(label.startswith("E") for label in labels) <= 2


def test_the_total_cap_binds_across_the_lists_and_prefers_higher_scores():
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1", "E2"))
    scores = {"doc_0": {"M1": 5, "M2": 4, "E1": 3, "E2": 3}}
    targets = {label: 1 for label in ("M1", "M2", "E1", "E2")}
    assert run(scores, catalog, targets, AssignmentLimits(3, 2, 2, 2, penalty_step_per_further_consideration_in_a_list=0)) == {"doc_0": ["M1", "M2"]}
    three = run(scores, catalog, targets, AssignmentLimits(3, 2, 2, 3, penalty_step_per_further_consideration_in_a_list=0))["doc_0"]
    assert three[:2] == ["M1", "M2"] and len(three) == 3 and three[2] in ("E1", "E2")
    assert run(scores, catalog, targets, AssignmentLimits(3, 2, 2, 1, penalty_step_per_further_consideration_in_a_list=0)) == {"doc_0": ["M1"]}


def test_caps_of_one_one_two_give_at_most_one_per_list():
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1", "E2"))
    scores = {f"doc_{i}": {"M1": 5, "M2": 4, "E1": 4, "E2": 5} for i in range(3)}
    targets = {label: 3 for label in ("M1", "M2", "E1", "E2")}
    limits = AssignmentLimits(3, 1, 1, 2)
    assigned = run(scores, catalog, targets, limits)
    for labels in assigned.values():
        assert labels == ["M1", "E2"]


def test_a_zero_cap_for_a_list_removes_that_list():
    catalog = make_catalog()
    scores = {"doc_0": {"M1": 5, "M2": 5, "E1": 5, "E2": 5}}
    assigned = run(scores, catalog, {"M1": 1, "M2": 1, "E1": 1, "E2": 1}, AssignmentLimits(3, 0, 1, 2, penalty_step_per_further_consideration_in_a_list=0))
    assert assigned == {"doc_0": ["E1"]} or assigned == {"doc_0": ["E2"]}


def test_coverage_report_and_offline_reassignment_from_a_scores_file(tmp_path):
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))
    rows = [{"prompt_id": f"doc_{i}", "scores": {
        "moral": {"M1": {"score": 3 + i % 3, "where_it_arises": "here"}},
        "epistemic": {"E1": {"score": 1, "where_it_arises": ""}}}} for i in range(4)]
    path = tmp_path / "consideration_scores.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assigned, report = assignment.assign_from_scores_file(
        path, catalog, documents(4), AssignmentLimits(), targets={"M1": 2, "E1": 1})
    assert sum(bool(labels) for labels in assigned.values()) == 2
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["M1"]["achieved"] == 2 and by_label["M1"]["target_count"] == 2
    assert by_label["M1"]["documents_scoring_at_least_threshold"] == 4
    assert by_label["M1"]["score_histogram"]["3"] == 2 and by_label["E1"]["score_histogram"]["1"] == 4
    assert report["documents_with_no_consideration"] == 2 and len(report["document_ids_with_no_consideration"]) == 2
    stricter = assignment.assign_from_scores_file(
        path, catalog, documents(4), AssignmentLimits(minimum_assignable_score=5), targets={"M1": 2, "E1": 1})[0]
    assert stricter == {"doc_0": [], "doc_1": [], "doc_2": ["M1"], "doc_3": []}   # only doc_2 scored 5


def test_limits_come_from_config_with_defaults():
    defaults = assignment.limits_from_config({"sdf": {}})
    assert defaults == AssignmentLimits(3, 3, 3, 4, {3: 1, 4: 3, 5: 9}, 2) == AssignmentLimits()
    config = {"sdf": {"minimum_assignable_score": 4, "max_moral_considerations_per_document": 2,
                      "max_epistemic_considerations_per_document": 3, "max_considerations_per_document": 4,
                      "score_weights": {4: 2, 5: 5}, "penalty_step_per_further_consideration_in_a_list": 1}}
    assert assignment.limits_from_config(config) == AssignmentLimits(4, 2, 3, 4, {4: 2, 5: 5}, 1)


def test_every_assignable_score_needs_a_positive_weight():
    with pytest.raises(ValueError):
        AssignmentLimits(score_weights={3: 1, 5: 9})
    with pytest.raises(ValueError):
        AssignmentLimits(score_weights={3: 0, 4: 3, 5: 9})
    AssignmentLimits(minimum_assignable_score=4, score_weights={4: 3, 5: 9})       # weight for 3 is irrelevant


# --- the convex per-list penalty ---

def one_document_with_two_moral(second_score):
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1",))
    scores = {"doc_0": {"M1": 5, "M2": second_score, "M3": 1, "E1": 1}}
    return run(scores, catalog, {"M1": 1, "M2": 1, "M3": 1, "E1": 1}, AssignmentLimits(penalty_step_per_further_consideration_in_a_list=3))


def test_a_second_consideration_in_a_list_needs_a_weight_above_the_penalty():
    # weights: 3 -> 1, 4 -> 3, 5 -> 9 with p = 3: a 4 (weight 3) only ties the penalty and is left out; a 5 (9) is worth it.
    assert one_document_with_two_moral(3) == {"doc_0": ["M1"]}
    assert one_document_with_two_moral(4) == {"doc_0": ["M1"]}
    assert one_document_with_two_moral(5) == {"doc_0": ["M1", "M2"]}
    just_under = AssignmentLimits(score_weights={3: 1, 4: 3.0001, 5: 9}, penalty_step_per_further_consideration_in_a_list=3)
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1",))
    scores = {"doc_0": {"M1": 5, "M2": 4, "E1": 1}}
    assert run(scores, catalog, {"M1": 1, "M2": 1, "E1": 1}, just_under) == {"doc_0": ["M1", "M2"]}


def test_a_third_consideration_in_a_list_needs_a_weight_above_twice_the_penalty():
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1",))
    targets = {"M1": 1, "M2": 1, "M3": 1, "E1": 1}
    scores = {"doc_0": {"M1": 5, "M2": 5, "M3": 4, "E1": 1}}
    # p = 1.5: the third costs 3.0, the weight of a 4 -> not worth it at weight 3 (tie), worth it at 3.5.
    tie = AssignmentLimits(score_weights={3: 1, 4: 3, 5: 9}, penalty_step_per_further_consideration_in_a_list=1.5)
    assert run(scores, catalog, targets, tie) == {"doc_0": ["M1", "M2"]}
    above = AssignmentLimits(score_weights={3: 1, 4: 3.5, 5: 9}, penalty_step_per_further_consideration_in_a_list=1.5)
    assert run(scores, catalog, targets, above) == {"doc_0": ["M1", "M2", "M3"]}


def test_the_penalty_is_per_list_so_one_moral_and_one_epistemic_cost_nothing():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))
    scores = {"doc_0": {"M1": 3, "E1": 3}}       # weight 1 each, far below any penalty that applied across lists
    limits = AssignmentLimits(penalty_step_per_further_consideration_in_a_list=100)
    assert run(scores, catalog, {"M1": 1, "E1": 1}, limits) == {"doc_0": ["M1", "E1"]}


def test_the_penalty_spreads_picks_across_documents():
    # One entry, three slots; doc_0 and doc_1 score it equally on three entries each.
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1",))
    scores = {pid: {"M1": 5, "M2": 5, "M3": 5, "E1": 1} for pid in ("doc_0", "doc_1")}
    targets = {"M1": 1, "M2": 1, "M3": 1, "E1": 1}
    no_penalty = run(scores, catalog, targets, AssignmentLimits(penalty_step_per_further_consideration_in_a_list=0))
    assert total(no_penalty) == 3
    spread = run(scores, catalog, targets, AssignmentLimits(penalty_step_per_further_consideration_in_a_list=2))
    assert total(spread) == 3 and sorted(len(labels) for labels in spread.values()) == [1, 2]   # 2 + 1, never 3 + 0
    heavy = run({**scores, "doc_2": {"M1": 5, "M2": 5, "M3": 5, "E1": 1}}, catalog, targets,
                AssignmentLimits(penalty_step_per_further_consideration_in_a_list=2))
    assert sorted(len(labels) for labels in heavy.values()) == [1, 1, 1]


def test_slots_stay_unfilled_when_only_weak_extras_remain():
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1",))
    scores = {"doc_0": {"M1": 5, "M2": 3, "E1": 1}, "doc_1": {"M1": 1, "M2": 3, "E1": 1}}
    # M2's slot (target 2): doc_1 takes it (first of its list, weight 1); doc_0's would be a weak extra (1 - 2 < 0).
    assigned = run(scores, catalog, {"M1": 1, "M2": 2, "E1": 1})
    assert assigned == {"doc_0": ["M1"], "doc_1": ["M2"]}


def test_hard_caps_hold_even_without_a_penalty():
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1", "E2"))
    scores = {f"doc_{i}": {label: 5 for label in ("M1", "M2", "M3", "E1", "E2")} for i in range(3)}
    targets = {label: 3 for label in ("M1", "M2", "M3", "E1", "E2")}
    assigned = run(scores, catalog, targets, AssignmentLimits(3, 2, 1, 3, penalty_step_per_further_consideration_in_a_list=0))
    for labels in assigned.values():
        assert len(labels) == 3
        assert sum(label.startswith("M") for label in labels) <= 2 and sum(label.startswith("E") for label in labels) <= 1


# --- permutation invariance ---

def test_the_assignment_does_not_depend_on_input_order():
    catalog = make_catalog(moral=("M1", "M2"), epistemic=("E1", "E2"))
    rng = random.Random(5)
    scores = {f"doc_{i}": {label: rng.choice((3, 4, 5)) for label in ("M1", "M2", "E1", "E2")} for i in range(10)}
    targets = {"M1": 3, "M2": 3, "E1": 3, "E2": 3}
    reference = run(scores, catalog, targets, seed=3)
    for shuffle_seed in range(5):
        shuffler = random.Random(shuffle_seed)
        document_order = list(scores)
        shuffler.shuffle(document_order)
        shuffled = {}
        for pid in document_order:
            label_order = list(scores[pid])
            shuffler.shuffle(label_order)
            shuffled[pid] = {label: scores[pid][label] for label in label_order}
        assert run(shuffled, catalog, targets, seed=3) == reference


def test_an_unrelated_document_does_not_change_the_tie_break_between_two_others():
    catalog = make_catalog(moral=("M1",), epistemic=("E1",))
    base = {f"doc_{i}": {"M1": 4, "E1": 1} for i in range(8)}
    targets = {"M1": 1, "E1": 1}
    for seed in range(10):
        winner = [pid for pid, labels in run(base, catalog, targets, seed=seed).items() if labels]
        assert len(winner) == 1
        # an extra document that cannot win (low score) and one that is not even scored
        extended = {**base, "doc_extra": {"M1": 1, "E1": 1}, "doc_extra_two": {"M1": 2, "E1": 1}}
        flags = {pid: False for pid in extended}
        again = assign_considerations(extended, catalog, flags, AssignmentLimits(), seed, targets)
        assert [pid for pid, labels in again.items() if labels] == winner
        pair = {pid: base[pid] for pid in ("doc_2", "doc_5")}
        winner_of_pair = [pid for pid, labels in run(pair, catalog, targets, seed=seed).items() if labels]
        pair_extended = {**pair, "doc_unrelated": {"M1": 1, "E1": 1}}
        assert [pid for pid, labels in run(pair_extended, catalog, targets, seed=seed).items() if labels] == winner_of_pair


# --- integrality and constraints on random instances ---

@pytest.mark.parametrize("instance_seed", range(30))
def test_random_instances_are_integral_and_satisfy_every_constraint(instance_seed):
    rng = random.Random(instance_seed)
    catalog = make_catalog(moral=("M1", "M2", "M3"), epistemic=("E1", "E2"), digital=("D1",))
    labels = ("M1", "M2", "M3", "E1", "E2", "D1")
    flags = {f"doc_{i}": rng.random() < 0.3 for i in range(20)}
    scores = {pid: {label: rng.randint(1, 5) for label in labels} for pid in flags}
    targets = {label: rng.randint(1, 8) for label in labels}
    limits = AssignmentLimits(3, rng.randint(1, 3), rng.randint(1, 3), rng.randint(2, 4),
                              penalty_step_per_further_consideration_in_a_list=rng.choice((0, 1, 2, 4)))
    assigned = assign_considerations(scores, catalog, flags, limits, instance_seed, targets)   # asserts run inside
    list_of = {entry.label: entry.list_name for entry in catalog.all_entries}
    for label, target in targets.items():
        assert sum(label in picks for picks in assigned.values()) <= target
    for pid, picks in assigned.items():
        assert len(picks) <= limits.max_total_per_document
        for list_name in (MORAL_LIST, EPISTEMIC_LIST):
            assert sum(list_of[label] == list_name for label in picks) <= limits.max_for_list(list_name)
        assert all(scores[pid][label] >= 3 and (flags[pid] or label != "D1") for label in picks)
