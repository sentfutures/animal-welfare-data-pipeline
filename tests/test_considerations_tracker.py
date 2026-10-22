"""Pure-code tests for the considerations catalog and the run-level tracker.

No API stub is needed anywhere in this file: the tracker is rules only. The
catalog tests read the real prompts/sdf/considerations.txt and variables.txt
(that is what catches a reworded sentient_category), and the tracker tests build
small catalogs directly so each rule is exercised on its own.
"""

import pytest

from sdf_pipeline import considerations_catalog as catalog_module
from sdf_pipeline.considerations_catalog import (
    Consideration, ConsiderationCatalog, EPISTEMIC_LIST, MORAL_LIST, load_considerations, target_count,
)
from sdf_pipeline.considerations_tracker import (
    ConsiderationTracker, corpus_coverage, STATUS_ACCEPTED, STATUS_NONE, STATUS_OVER_CAP,
)


def make_catalog(moral=(("M1", 0.5), ("M2", 0.5)), epistemic=(("E1", 0.5),), digital=(("D1", 0.5),)):
    moral_entries = tuple(Consideration(label, f"gloss {label}", share, MORAL_LIST) for label, share in moral)
    epistemic_entries = tuple(
        Consideration(label, f"gloss {label}", share, EPISTEMIC_LIST, True) for label, share in digital
    ) + tuple(Consideration(label, f"gloss {label}", share, EPISTEMIC_LIST) for label, share in epistemic)
    return ConsiderationCatalog(
        moral=moral_entries, epistemic=epistemic_entries,
        digital_mind_categories=frozenset({"digital"}), skipped_resolutions=frozenset({"no stake"}),
    )


def make_documents(count, digital_ids=()):
    return [(f"doc_{i}", f"doc_{i}" in digital_ids) for i in range(count)]


def labels(entries):
    return [entry.label for entry in entries]


# --- the real file ---

def test_the_real_file_loads_with_both_lists_in_file_order(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert labels(catalog.moral)[0] == "Substrate neutrality"
    assert labels(catalog.moral)[-1] == "Relevant principles from the constitution"
    # the digital-minds-only entries sit at the top of the epistemic list
    assert [e.digital_minds_only for e in catalog.epistemic[:2]] == [True, True]
    assert not any(e.digital_minds_only for e in catalog.epistemic[2:])
    assert catalog.epistemic[2].label == "Weighing evidence for sentience"
    assert len({e.label for e in catalog.all_entries}) == len(catalog.all_entries)


def test_the_precautionary_principle_is_on_the_moral_list_only(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert any("Precautionary" in e.label for e in catalog.moral)
    assert not any("precaution" in e.display.lower() for e in catalog.epistemic)


def test_the_targets_are_the_shares_in_the_plan(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    shares = {e.label: e.target_share for e in catalog.all_entries}
    assert shares["Weighing evidence for sentience"] == 0.10
    assert shares["Absence of evidence versus evidence of absence"] == 0.01
    assert shares["Weighing theories of consciousness and their implications for AI"] == 0.20
    assert shares["Difficulties of AI self-reporting welfare"] == 0.10
    assert shares["Substrate neutrality"] == 0.08
    assert shares["Continuity with the AI's aligned character"] == 0.15


def test_digital_mind_categories_resolve_to_the_three_real_values(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert len(catalog.digital_mind_categories) == 3
    assert all("connectomes" in c or "LLMs" in c for c in catalog.digital_mind_categories)
    assert catalog.is_digital_mind_document({"sentient_category": "farmed fishes"}) is False


def test_rewording_a_digital_mind_category_makes_loading_fail(prompts_sdf, tmp_path):
    for name in ("considerations.txt", "variables.txt"):
        (tmp_path / name).write_text((prompts_sdf / name).read_text(encoding="utf-8"), encoding="utf-8")
    variables = (tmp_path / "variables.txt").read_text(encoding="utf-8")
    assert "simulated human connectomes whose moral state is uncertain" in variables
    (tmp_path / "variables.txt").write_text(
        variables.replace("simulated human connectomes whose", "emulated human brains whose"), encoding="utf-8")
    with pytest.raises(ValueError, match="simulated human connectomes"):
        load_considerations(tmp_path)


def test_rewording_the_no_stake_resolution_makes_loading_fail(prompts_sdf, tmp_path):
    for name in ("considerations.txt", "variables.txt"):
        (tmp_path / name).write_text((prompts_sdf / name).read_text(encoding="utf-8"), encoding="utf-8")
    variables = (tmp_path / "variables.txt").read_text(encoding="utf-8")
    (tmp_path / "variables.txt").write_text(variables.replace("NO real welfare stake", "no stake"), encoding="utf-8")
    with pytest.raises(ValueError, match="NO real welfare stake"):
        load_considerations(tmp_path)


@pytest.mark.parametrize("old,new,message", [
    ("Substrate neutrality: welfare", "Substrate neutrality welfare", "must read"),
    ("0.08 :: Substrate neutrality", "Substrate neutrality", "target share"),
    ("0.08 :: Substrate neutrality", "1.5 :: Substrate neutrality", "target share"),
    ("Individualism:", "Substrate neutrality:", "unique"),
])
def test_a_malformed_considerations_file_fails_loudly(prompts_sdf, tmp_path, old, new, message):
    text = (prompts_sdf / "considerations.txt").read_text(encoding="utf-8")
    (tmp_path / "variables.txt").write_text((prompts_sdf / "variables.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "considerations.txt").write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_considerations(tmp_path)


def test_target_count_rounds_half_up_with_a_floor_of_one():
    assert target_count(0.10, 100) == 10          # not 11: float noise must not push it over
    assert target_count(0.08, 60) == 5            # 4.8
    assert target_count(0.01, 60) == 1            # 0.6 -> floor of one
    assert target_count(0.05, 10) == 1            # 0.5 -> half up, still at the floor
    assert target_count(0.25, 10) == 3            # 2.5 -> 3
    assert target_count(0.5, 0) == 0


# --- priority and eligibility ---

def test_offered_entries_follow_priority_order():
    tracker = ConsiderationTracker(make_catalog(), make_documents(4))
    assert labels(tracker.offered_entries("doc_0", MORAL_LIST)) == ["M1", "M2"]
    assert labels(tracker.offered_entries("doc_0", EPISTEMIC_LIST)) == ["E1"]


def test_digital_minds_only_entries_are_offered_first_and_only_on_digital_documents():
    tracker = ConsiderationTracker(make_catalog(), make_documents(4, digital_ids={"doc_1"}))
    assert labels(tracker.offered_entries("doc_1", EPISTEMIC_LIST)) == ["D1", "E1"]
    assert labels(tracker.offered_entries("doc_0", EPISTEMIC_LIST)) == ["E1"]


def test_digital_minds_only_targets_use_the_digital_documents_as_denominator():
    catalog = make_catalog(digital=(("D1", 0.5),), epistemic=(("E1", 0.5),))
    tracker = ConsiderationTracker(catalog, make_documents(20, digital_ids={"doc_0", "doc_1", "doc_2", "doc_3"}))
    assert tracker.targets["D1"] == 2      # half of 4 digital documents, not half of 20
    assert tracker.targets["E1"] == 10     # half of all 20
    row = {e["label"]: e for e in tracker.coverage_report()["entries"]}["D1"]
    assert (row["eligible_documents"], row["target_count"]) == (4, 2)


def test_digital_entry_with_no_digital_documents_has_a_zero_target_and_is_never_offered():
    tracker = ConsiderationTracker(make_catalog(), make_documents(6))
    assert tracker.targets["D1"] == 0
    assert all(label != "D1" for label in labels(tracker.offered_entries("doc_0", EPISTEMIC_LIST)))
    assert not tracker.is_open(tracker.catalog.entry_by_label("D1"))


# --- the cap ---

def test_a_category_is_removed_at_exactly_its_target_and_not_before():
    catalog = make_catalog(moral=(("M1", 0.3), ("M2", 0.9)))
    tracker = ConsiderationTracker(catalog, make_documents(10))   # M1 target 3
    for number in range(3):
        assert "M1" in labels(tracker.offered_entries(f"doc_{number}", MORAL_LIST))
        row = tracker.record_answer(f"doc_{number}", 1, 1, {MORAL_LIST: ("M1", "r")})
        assert row["outcomes"][0]["status"] == STATUS_ACCEPTED
    assert tracker.counts["M1"] == 3
    assert "M1" not in labels(tracker.offered_entries("doc_5", MORAL_LIST))
    assert labels(tracker.offered_entries("doc_5", MORAL_LIST)) == ["M2"]


def test_accepting_stops_exactly_at_target_in_document_order_within_a_wave():
    catalog = make_catalog(moral=(("M1", 0.2),))
    tracker = ConsiderationTracker(catalog, make_documents(10))   # target 2
    statuses = [
        tracker.record_answer(f"doc_{i}", 1, 1, {MORAL_LIST: ("M1", "r")})["outcomes"][0]["status"]
        for i in range(4)
    ]
    assert statuses == [STATUS_ACCEPTED, STATUS_ACCEPTED, STATUS_OVER_CAP, STATUS_OVER_CAP]
    assert tracker.counts["M1"] == 2
    assert tracker.over_cap_counts["M1"] == 2


def test_an_over_cap_pick_is_re_asked_next_wave_against_the_updated_lists():
    catalog = make_catalog(moral=(("M1", 0.1), ("M2", 0.9)), epistemic=(("E1", 0.5),))
    tracker = ConsiderationTracker(catalog, make_documents(10))   # M1 target 1
    wave_one = tracker.next_wave_calls(1, 3)
    assert [c.prompt_id for c in wave_one] == ["doc_0", "doc_1", "doc_2"]
    for call in wave_one:   # all three ask for M1 on the moral list and E1 on the epistemic list
        tracker.record_answer(call.prompt_id, 1, 1, {MORAL_LIST: ("M1", "r"), EPISTEMIC_LIST: ("E1", "r")})
    assert tracker.counts["M1"] == 1
    wave_two = tracker.next_wave_calls(1, 3)
    # the two rejected documents come first, and only for the list that was rejected
    assert [(c.prompt_id, c.list_names) for c in wave_two[:2]] == [
        ("doc_1", (MORAL_LIST,)), ("doc_2", (MORAL_LIST,))]
    assert wave_two[2].prompt_id == "doc_3"
    assert "M1" not in labels(tracker.offered_entries("doc_1", MORAL_LIST))
    assert tracker.next_wave_number(1) == 2
    row = tracker.record_answer("doc_1", 1, 2, {MORAL_LIST: ("M2", "r")})
    assert row["outcomes"][0]["status"] == STATUS_ACCEPTED
    assert (1, "doc_1", MORAL_LIST) in tracker._resolved


def test_a_none_answer_resolves_the_list_for_that_pass():
    tracker = ConsiderationTracker(make_catalog(), make_documents(2))
    tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: (None, ""), EPISTEMIC_LIST: (None, "")})
    assert [c.prompt_id for c in tracker.pending_calls(1)] == ["doc_1"]


def test_a_label_outside_the_documents_offer_is_an_error():
    tracker = ConsiderationTracker(make_catalog(), make_documents(4))
    with pytest.raises(ValueError, match="not eligible"):
        tracker.record_answer("doc_0", 1, 1, {EPISTEMIC_LIST: ("D1", "r")})
    with pytest.raises(ValueError, match="not on the moral list"):
        tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: ("E1", "r")})
    tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: ("M1", "r")})
    with pytest.raises(ValueError, match="already carries"):
        tracker.record_answer("doc_0", 2, 1, {MORAL_LIST: ("M1", "r")})


# --- passes ---

def test_pass_two_offers_only_under_target_categories_and_never_one_the_document_carries():
    catalog = make_catalog(moral=(("M1", 0.1), ("M2", 0.5), ("M3", 0.5)), epistemic=(("E1", 0.1),))
    tracker = ConsiderationTracker(catalog, make_documents(10))   # M1 and E1 target 1
    tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: ("M1", "r"), EPISTEMIC_LIST: ("E1", "r")})
    tracker.record_answer("doc_1", 1, 1, {MORAL_LIST: ("M2", "r"), EPISTEMIC_LIST: (None, "")})
    # M1 and E1 are at target: gone from every list. doc_1 already carries M2.
    assert labels(tracker.offered_entries("doc_1", MORAL_LIST)) == ["M3"]
    assert labels(tracker.offered_entries("doc_2", MORAL_LIST)) == ["M2", "M3"]
    assert tracker.offered_entries("doc_2", EPISTEMIC_LIST) == []
    calls = {c.prompt_id: c.list_names for c in tracker.pending_calls(2)}
    assert calls["doc_1"] == (MORAL_LIST,)             # a document that already has a pick is asked again
    assert all(names == (MORAL_LIST,) for names in calls.values())   # the epistemic list has nothing left


def test_a_document_with_nothing_left_to_offer_is_not_asked():
    tracker = ConsiderationTracker(make_catalog(moral=(("M1", 0.1),), epistemic=(("E1", 0.1),)), make_documents(10))
    tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: ("M1", "r"), EPISTEMIC_LIST: ("E1", "r")})
    assert tracker.all_targets_met()
    assert tracker.pending_calls(2) == []


def test_failed_documents_can_be_skipped_when_scheduling():
    tracker = ConsiderationTracker(make_catalog(), make_documents(3))
    calls = tracker.pending_calls(1, skip=frozenset({"doc_0"}))
    assert [c.prompt_id for c in calls] == ["doc_1", "doc_2"]


# --- rebuilding ---

def run_scripted_waves(tracker, script):
    """Feed (document, answer) pairs wave by wave and return the rows written."""
    rows = []
    for pass_number, wave_number, answers in script:
        for prompt_id, choices in answers:
            rows.append(tracker.record_answer(prompt_id, pass_number, wave_number, choices))
    return rows


def state_of(tracker):
    return (
        dict(tracker.counts), dict(tracker.over_cap_counts),
        [(p.prompt_id, p.list_name, p.label, p.reason, p.pass_number, p.wave_number) for p in tracker.picks],
        sorted(tracker._resolved), sorted(tracker._over_cap_asks), dict(tracker._last_wave),
        [(c.prompt_id, c.list_names) for c in tracker.pending_calls(1)],
        [(c.prompt_id, c.list_names) for c in tracker.pending_calls(2)],
    )


def test_rebuilding_from_the_saved_rows_gives_the_state_of_the_uninterrupted_run():
    catalog = make_catalog(moral=(("M1", 0.2), ("M2", 0.5)), epistemic=(("E1", 0.3),))
    documents = make_documents(10, digital_ids={"doc_4"})
    script = [
        (1, 1, [("doc_0", {MORAL_LIST: ("M1", "a"), EPISTEMIC_LIST: ("E1", "b")}),
                ("doc_1", {MORAL_LIST: ("M1", "c"), EPISTEMIC_LIST: (None, "")}),
                ("doc_2", {MORAL_LIST: ("M1", "d"), EPISTEMIC_LIST: ("E1", "e")})]),   # M1 over cap at doc_2
        (1, 2, [("doc_2", {MORAL_LIST: ("M2", "f")}),
                ("doc_3", {MORAL_LIST: (None, ""), EPISTEMIC_LIST: ("E1", "g")})]),
    ]
    uninterrupted = ConsiderationTracker(catalog, documents)
    rows = run_scripted_waves(uninterrupted, script)
    assert [o["status"] for o in rows[2]["outcomes"]] == [STATUS_OVER_CAP, STATUS_ACCEPTED]

    rebuilt = ConsiderationTracker(catalog, documents)
    rebuilt.replay(rows)
    assert state_of(rebuilt) == state_of(uninterrupted)

    # a partial file (killed after the first wave) rebuilds to the state at that point, then continues identically
    partial = ConsiderationTracker(catalog, documents)
    partial.replay(rows[:3])
    expected_midway = ConsiderationTracker(catalog, documents)
    run_scripted_waves(expected_midway, script[:1])
    assert state_of(partial) == state_of(expected_midway)
    for row in rows[3:]:
        partial.record_answer(row["prompt_id"], row["pass"], row["wave"],
                              {o["list"]: (o["label"], o["reason"]) for o in row["outcomes"]})
    assert state_of(partial) == state_of(uninterrupted)


def test_replay_trusts_recorded_statuses_and_ignores_documents_no_longer_eligible():
    tracker = ConsiderationTracker(make_catalog(), make_documents(2))
    tracker.replay([
        {"prompt_id": "doc_0", "pass": 1, "wave": 1,
         "outcomes": [{"list": MORAL_LIST, "label": "M1", "reason": "r", "status": STATUS_ACCEPTED}]},
        {"prompt_id": "removed_by_screen", "pass": 1, "wave": 1,
         "outcomes": [{"list": MORAL_LIST, "label": "M2", "reason": "r", "status": STATUS_ACCEPTED}]},
    ])
    assert tracker.counts["M1"] == 1 and tracker.counts["M2"] == 0


# --- coverage ---

def test_the_coverage_report_counts_none_one_both_and_repeats():
    catalog = make_catalog(moral=(("M1", 0.9), ("M2", 0.9)), epistemic=(("E1", 0.9),))
    tracker = ConsiderationTracker(catalog, make_documents(4))
    tracker.record_answer("doc_0", 1, 1, {MORAL_LIST: ("M1", "r"), EPISTEMIC_LIST: ("E1", "r")})
    tracker.record_answer("doc_1", 1, 1, {MORAL_LIST: ("M1", "r"), EPISTEMIC_LIST: (None, "")})
    tracker.record_answer("doc_1", 2, 1, {MORAL_LIST: ("M2", "r")})
    tracker.record_answer("doc_2", 1, 1, {MORAL_LIST: (None, ""), EPISTEMIC_LIST: (None, "")})
    report = tracker.coverage_report()
    assert report["eligible_documents"] == 4
    assert report["documents_with_no_consideration"] == 2      # doc_2 and the never-asked doc_3
    assert report["documents_with_only_one_list"] == 1
    assert report["documents_with_both_lists"] == 1
    assert report["documents_with_two_or_more_in_one_list"] == 1
    assert report["share_with_no_consideration"] == 0.5
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["M2"]["from_later_passes"] == 1 and by_label["M1"]["from_later_passes"] == 0


def test_corpus_coverage_measures_final_documents_and_skips_the_no_stake_arm():
    catalog = make_catalog(moral=(("M1", 0.5),), epistemic=(("E1", 0.5),), digital=(("D1", 0.5),))
    corpus = [
        {"sentient_category": "digital", "moral_considerations": ["M1"], "epistemic_considerations": ["D1"]},
        {"sentient_category": "fish", "moral_considerations": [], "epistemic_considerations": []},
        {"sentient_category": "fish", "resolution": "no stake", "moral_considerations": [], "epistemic_considerations": []},
    ]
    report = corpus_coverage(catalog, corpus)
    assert report["eligible_documents"] == 2 and report["digital_mind_documents"] == 1
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["D1"]["achieved_share"] == 1.0 and by_label["M1"]["achieved_share"] == 0.5
    assert report["documents_with_both_lists"] == 1 and report["documents_with_no_consideration"] == 1
