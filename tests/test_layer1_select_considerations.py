"""Behavior tests for the layer-1 considerations selector, driven through
layer1_select_considerations.run() and select_for_document() with a stubbed
shared.api.call_claude (no test here may reach the API; see conftest). The real
selector template stays in play. Most stage tests use a small considerations
file written to tmp_path (so exact counts are knowable); the checks about the
real file, the real templates and the drafter use the repo's own files.

Money paths: parse happy path, malformed / unknown-label answers are not
checkpointed, resume makes zero calls for completed work, every-call-failure
exits, and a callable dispatcher (the stage runs a rolling pool of workers).
"""

import json
import re
import threading
from pathlib import Path

import pytest

from evals import audit_sdf
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_select_considerations as select
from sdf_pipeline import layer2_draft
from sdf_pipeline.considerations_catalog import load_considerations
from shared import utils

SMALL_CONSIDERATIONS = """\
{moral_considerations}
    0.2 :: Alpha moral: the first moral gloss
    0.9 :: Beta moral: the second moral gloss

{epistemic_considerations}
    0.1 :: Gamma epistemic: the epistemic gloss

{epistemic_considerations_digital_minds_only}
    0.5 :: Delta digital: the digital-minds gloss

{digital_mind_categories}
    large_language_models

{skipped_resolutions}
    no_stake
"""

DIGITAL_CATEGORY = "LLMs whose moral state is uncertain"


@pytest.fixture
def small_prompts(prompts_sdf, tmp_path):
    """A prompts dir with the real selector template and variables but a small considerations file."""
    directory = tmp_path / "prompts_small"
    directory.mkdir()
    for name in ("variables.txt", select.TEMPLATE_FILENAME):
        (directory / name).write_text((prompts_sdf / name).read_text(encoding="utf-8"), encoding="utf-8")
    (directory / "considerations.txt").write_text(SMALL_CONSIDERATIONS, encoding="utf-8")
    return directory


@pytest.fixture
def stage_dir(tmp_path):
    directory = tmp_path / "layer1"
    directory.mkdir()
    return directory


@pytest.fixture
def select_config(tiny_config):
    tiny_config["sdf"]["max_selection_passes"] = 3
    return tiny_config


def no_stake_resolution(prompts_dir):
    return sorted(load_considerations(prompts_dir).skipped_resolutions)[0]


def make_plans(count=10, digital=(), no_stake=(), incoherent=(), resolution=None):
    plans = []
    for i in range(count):
        variables = {
            "document_type": "a news article",
            "centrality": "the central subject of the document",
            "sentient_category": DIGITAL_CATEGORY if i in digital else "farmed fishes",
            "resolution": resolution if i in no_stake else "a creative solution",
        }
        plans.append({
            "prompt_id": f"matrix_{i:06d}", "variables": variables, "plan": "…",
            "description": None if i in incoherent else f"Spec {i}.",
            "incoherent": i in incoherent,
        })
    return plans


def offered_in(user_message):
    """{'moral': [labels], 'epistemic': [labels]} the selector prompt showed."""
    shown = {}
    for name in ("moral", "epistemic"):
        body = re.search(rf"<{name}_list>\n(.*?)\n</{name}_list>", user_message, re.DOTALL).group(1)
        shown[name] = [line[2:].split(": ")[0] for line in body.splitlines() if line.startswith("- ")]
    return shown


def answer(moral=None, epistemic=None):
    def one(label):
        if label is None:
            return {"reasoning": "Nothing fits.", "label": "none", "where_it_arises": ""}
        return {"reasoning": "It fits.", "label": label, "where_it_arises": f"Where {label} arises."}
    return json.dumps({"moral": one(moral), "epistemic": one(epistemic)})


def pick_first_offered(user_message, **kw):
    shown = offered_in(user_message)
    return answer(shown["moral"][0] if shown["moral"] else None,
                  shown["epistemic"][0] if shown["epistemic"] else None)


def say_none(user_message, **kw):
    return answer()


def rows_of(stage_dir):
    return utils.load_jsonl(stage_dir / select.SELECTIONS_FILENAME)


def picks_by_document(plans):
    return {p["prompt_id"]: p["variables"]["moral_considerations"] + p["variables"]["epistemic_considerations"]
            for p in plans if p.get("description")}


# --- parsing and the public function ---

def test_select_for_document_parses_a_happy_path_answer(stub_claude, small_prompts):
    catalog = load_considerations(small_prompts)
    calls = stub_claude(lambda user_message, **kw: answer("Beta moral", None))
    facts = select.DocumentFacts("A scenario.", "a news article", "central", "farmed fishes")
    offered = {"moral": list(catalog.moral), "epistemic": [catalog.epistemic[-1]]}
    choices = select.select_for_document(
        facts, offered, template=select.load_selection_template(small_prompts), model="m", item_id="d0")
    entry, reason = choices["moral"]
    assert entry.label == "Beta moral" and reason == "Where Beta moral arises."
    assert choices["epistemic"] == (None, "")
    assert calls[0]["stage"] == "layer1_select" and calls[0]["model"] == "m" and calls[0]["item_id"] == "d0"
    assert calls[0]["system_prompt"] and "A scenario." in calls[0]["user_message"]
    assert not re.search(r"\{[a-z_]+\}", calls[0]["user_message"])        # no unrendered placeholder


def make_offer(small_prompts):
    catalog = load_considerations(small_prompts)
    return catalog, {"moral": list(catalog.moral), "epistemic": [catalog.epistemic[-1]]}


@pytest.mark.parametrize("raw", [
    "no json at all",
    '{"moral": {"label": "Alpha moral", "where_it_arises": "x"}}',                  # epistemic missing
    '{"moral": {"label": "Made up label", "where_it_arises": "x"}, "epistemic": {"label": "none"}}',
    '{"moral": {"label": "Gamma epistemic", "where_it_arises": "x"}, "epistemic": {"label": "none"}}',  # other list's label
    '{"moral": {"label": "Alpha moral"}, "epistemic": {"label": "none"}}',          # a pick with no reason
    '{"moral": {"label": "Alpha moral", "where_it_arises": "  "}, "epistemic": {"label": "none"}}',
    '{"moral": {"label": "m1", "where_it_arises": "x"}, "epistemic": {"label": "none"}}',            # an id, not a label
    '{"moral": "Alpha moral", "epistemic": "none"}',
])
def test_parse_rejects_unusable_answers_and_unknown_labels(small_prompts, raw):
    _, offered = make_offer(small_prompts)
    with pytest.raises(select.SelectionAnswerError):
        select.parse_selection_answer(raw, offered)


def test_parse_tolerates_case_spacing_quotes_and_none_spellings(small_prompts):
    _, offered = make_offer(small_prompts)
    choices = select.parse_selection_answer(
        '{"moral": {"label": " \\"alpha  MORAL\\" ", "where_it_arises": "  here  it\\narises "}, '
        '"epistemic": {"label": "None"}}', offered)
    assert choices["moral"][0].label == "Alpha moral" and choices["moral"][1] == "here it arises"
    assert choices["epistemic"] == (None, "")


def test_a_pick_from_a_list_with_nothing_offered_is_rejected(small_prompts):
    catalog, _ = make_offer(small_prompts)
    offered = {"moral": list(catalog.moral), "epistemic": []}
    ok = select.parse_selection_answer(
        '{"moral": {"label": "none"}, "epistemic": {"label": "none"}}', offered)
    assert list(ok) == ["moral"]
    with pytest.raises(select.SelectionAnswerError, match="nothing was offered"):
        select.parse_selection_answer(
            '{"moral": {"label": "none"}, "epistemic": {"label": "Gamma epistemic", "where_it_arises": "x"}}', offered)


def test_selection_schema_enums_match_the_offered_labels(small_prompts):
    catalog, offered = make_offer(small_prompts)
    schema = select.build_selection_schema(offered)
    assert list(schema["properties"]) == ["moral", "epistemic"]
    assert schema["required"] == ["moral", "epistemic"] and schema["additionalProperties"] is False
    for list_name in ("moral", "epistemic"):
        list_schema = schema["properties"][list_name]
        assert list(list_schema["properties"]) == ["reasoning", "label", "where_it_arises"]  # reasoning comes first
        assert list_schema["required"] == ["reasoning", "label", "where_it_arises"]
        assert list_schema["additionalProperties"] is False
        assert list_schema["properties"]["label"]["enum"] == [e.label for e in offered[list_name]] + ["none"]
    # a list with nothing offered (or missing) can only answer none
    emptied = select.build_selection_schema({"moral": offered["moral"], "epistemic": []})
    assert emptied["properties"]["epistemic"]["properties"]["label"]["enum"] == ["none"]
    assert select.build_selection_schema({})["properties"]["moral"]["properties"]["label"]["enum"] == ["none"]


def test_schema_shaped_answer_parses_and_reasoning_is_dropped(small_prompts):
    _, offered = make_offer(small_prompts)
    raw = json.dumps({
        "moral": {"reasoning": "Alpha fits.", "label": "Alpha moral", "where_it_arises": "In the second scene."},
        "epistemic": {"reasoning": "Nothing fits.", "label": "none", "where_it_arises": ""}})
    choices = select.parse_selection_answer(raw, offered)
    assert choices["moral"][0].label == "Alpha moral" and choices["moral"][1] == "In the second scene."
    assert choices["epistemic"] == (None, "")


def test_select_for_document_sends_the_schema(stub_claude, small_prompts):
    _, offered = make_offer(small_prompts)
    calls = stub_claude(say_none)
    select.select_for_document(select.DocumentFacts("A scenario.", "t", "c", "s"), offered,
                               template=select.load_selection_template(small_prompts))
    assert calls[0]["json_schema"] == select.build_selection_schema(offered)


def test_real_catalog_continuity_label_is_in_the_enum_and_its_gloss_is_not(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    entry = next(e for e in catalog.all_entries
                 if e.label == "the continuity between welfare concern and the rest of the AI's aligned character")
    assert entry.gloss  # the entry the model once answered with its label plus part of this gloss
    schema = select.build_selection_schema({entry.list_name: list(catalog.entries_of(entry.list_name))})
    enum = schema["properties"][entry.list_name]["properties"]["label"]["enum"]
    assert entry.label in enum
    assert not any(entry.gloss in value or value.startswith(entry.label + ":") for value in enum)


def test_a_label_with_a_straight_double_quote_is_refused_at_load(small_prompts):
    # The API writes a chosen enum value out unescaped, so a quoted label breaks the answer's JSON.
    path = small_prompts / "considerations.txt"
    text = path.read_text(encoding="utf-8")
    first_label = load_considerations(small_prompts).all_entries[0].label
    path.write_text(text.replace(first_label, 'a "quoted" label', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="straight double quote"):
        load_considerations(small_prompts)


def test_no_real_catalog_label_contains_a_straight_double_quote(prompts_sdf):
    assert not any('"' in entry.label for entry in load_considerations(prompts_sdf).all_entries)


def test_truncated_answer_is_rejected(stub_claude, small_prompts):
    _, offered = make_offer(small_prompts)
    stub_claude(lambda user_message, **kw: ('{"moral": {"label": "none"}', "max_tokens"))
    facts = select.DocumentFacts("A scenario.", "t", "c", "s")
    with pytest.raises(select.SelectionAnswerError, match="truncated"):
        select.select_for_document(facts, offered, template=select.load_selection_template(small_prompts))


def test_the_prompt_shows_the_lists_and_existing_picks_only_when_there_are_some(small_prompts):
    catalog, offered = make_offer(small_prompts)
    template = select.load_selection_template(small_prompts)
    facts = select.DocumentFacts("A scenario.", "a news article", "central", "farmed fishes")
    _, plain = select.build_selection_prompt(template, facts, offered)
    assert "Alpha moral: the first moral gloss" in plain and "none" in plain
    assert "already carries" not in plain
    _, later = select.build_selection_prompt(template, facts, offered, [catalog.moral[0]])
    assert "already carries" in later and "Alpha moral" in later.split("already carries")[1]
    _, one_list = select.build_selection_prompt(template, facts, {"moral": offered["moral"]})
    assert select.NO_ENTRIES_MARKER in one_list


def test_the_selector_prompt_tells_the_model_none_is_expected_and_never_shows_a_target(small_prompts):
    _, offered = make_offer(small_prompts)
    _, user = select.build_selection_prompt(
        select.load_selection_template(small_prompts), select.DocumentFacts("d", "t", "c", "s"), offered)
    assert "none is expected and usually correct" in user.lower()
    assert not re.search(r"\d+%|target|quota", user, re.IGNORECASE)


# --- the stage ---

def test_picks_are_attached_to_the_plans_with_variables_and_a_spec(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(10)
    calls = stub_claude(lambda user_message, **kw: answer("Alpha moral", None) if "Spec 0." in user_message else say_none(user_message))
    result = select.run(select_config, small_prompts, stage_dir, plans)
    by_id = {p["prompt_id"]: p for p in result}
    first = by_id["matrix_000000"]
    assert first["variables"]["moral_considerations"] == ["Alpha moral"]
    assert first["variables"]["epistemic_considerations"] == []
    assert first["considerations_spec"] == "Alpha moral (the first moral gloss) — Where Alpha moral arises."
    assert by_id["matrix_000001"]["considerations_spec"] == ""
    assert by_id["matrix_000001"]["variables"]["moral_considerations"] == []
    assert plans[0]["variables"].get("moral_considerations") is None   # the input plans are not mutated
    assert {c["stage"] for c in calls} == {"layer1_select"}
    assert not any("matrix_0" in c["user_message"] for c in calls)      # labels only, never ids
    assert all(c["temperature"] is None for c in calls)


def test_the_spec_has_one_line_per_pick_moral_list_first(select_config, small_prompts, stage_dir, stub_claude):
    stub_claude(lambda user_message, **kw: answer("Beta moral", "Gamma epistemic") if "Spec 3." in user_message else say_none(user_message))
    result = select.run(select_config, small_prompts, stage_dir, make_plans(10))
    spec = {p["prompt_id"]: p for p in result}["matrix_000003"]["considerations_spec"]
    assert spec.splitlines() == [
        "Beta moral (the second moral gloss) — Where Beta moral arises.",
        "Gamma epistemic (the epistemic gloss) — Where Gamma epistemic arises.",
    ]


def test_assemble_considerations_spec_is_empty_without_picks():
    assert select.assemble_considerations_spec([]) == ""


def test_assemble_considerations_spec_leaves_out_the_parentheses_for_an_entry_without_a_gloss():
    entry = select.Consideration("any relevant principles", "", 0.1, "moral")
    assert select.assemble_considerations_spec([(entry, "Where it arises.")]) == (
        "any relevant principles — Where it arises."
    )


def test_counts_never_exceed_targets_and_the_coverage_report_is_written(select_config, small_prompts, stage_dir, stub_claude):
    stub_claude(pick_first_offered)
    select.run(select_config, small_prompts, stage_dir, make_plans(10, digital={0, 1, 2, 3}))
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    for entry in report["entries"]:
        assert entry["achieved"] <= entry["target_count"]
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["Alpha moral"]["achieved"] == 2           # 20% of 10
    assert by_label["Gamma epistemic"]["achieved"] == 1       # 10% of 10
    assert by_label["Delta digital"]["eligible_documents"] == 4
    assert by_label["Delta digital"]["achieved"] == 2         # 50% of the 4 digital documents


def test_each_documents_lists_are_ordered_by_remaining_need_when_its_call_starts(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    calls = stub_claude(pick_first_offered)
    select.run(select_config, small_prompts, stage_dir, make_plans(10))
    pass_one_moral = [offered_in(c["user_message"])["moral"] for c in calls if c["item_id"].endswith("pass1")]
    # Alpha target 2, Beta target 9: Beta (need 9) is first and takes every pick until its need falls to Alpha's 2
    assert pass_one_moral[0] == ["Beta moral", "Alpha moral"]
    assert pass_one_moral[6] == ["Beta moral", "Alpha moral"]      # Beta need 3 after six picks, Alpha need 2
    assert pass_one_moral[7] == ["Alpha moral", "Beta moral"]      # Beta need 2 ties Alpha's 2: catalog order puts Alpha first
    assert pass_one_moral[8] == ["Beta moral", "Alpha moral"]      # Alpha was just picked (need 1), Beta still needs 2
    rows = rows_of(stage_dir)
    assert not any(o["status"] == "over_cap" for r in rows for o in r["outcomes"])


def test_stale_picks_beyond_the_target_are_rejected_over_cap_and_re_asked_in_the_same_pass(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 4
    barrier = threading.Barrier(4, timeout=10)
    started = {"count": 0, "first_four": [], "re_asks": 0}
    lock = threading.Lock()

    def dispatcher(user_message, **kw):
        with lock:
            started["count"] += 1
            first_four = started["count"] <= 4
            if first_four:
                started["first_four"].append(kw["item_id"])
            re_ask = (not first_four and kw["item_id"] in started["first_four"] and kw["item_id"].endswith("pass1"))
            if re_ask:
                started["re_asks"] += 1
                re_ask_number = started["re_asks"]
        if first_four:
            barrier.wait()      # all four calls have been offered the same list before any answer is recorded
            return answer("Alpha moral", None)    # Alpha's target is 2
        if re_ask:
            assert "Alpha moral" not in offered_in(user_message)["moral"]     # a freshly computed offer
            return answer("Beta moral" if re_ask_number == 1 else None, None)
        return say_none(user_message)

    stub_claude(dispatcher)
    result = select.run(select_config, small_prompts, stage_dir, make_plans(10))
    rows = rows_of(stage_dir)

    def moral_outcome(row):
        return next((o for o in row["outcomes"] if o["list"] == "moral"), {"label": None, "status": None})

    alpha_rows = [r for r in rows if moral_outcome(r)["label"] == "Alpha moral"]
    assert sorted(moral_outcome(r)["status"] for r in alpha_rows) == ["accepted", "accepted", "over_cap", "over_cap"]
    rejected_ids = [r["prompt_id"] for r in alpha_rows if moral_outcome(r)["status"] == "over_cap"]
    assert started["re_asks"] == 2 and len(rejected_ids) == 2
    for prompt_id in rejected_ids:      # re-asked in the same pass, and ended with another entry or none
        later = [r for r in rows if r["prompt_id"] == prompt_id and r["pass"] == 1][-1]
        assert moral_outcome(later)["status"] in ("accepted", "none")
        assert moral_outcome(later)["label"] != "Alpha moral"
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["Alpha moral"]["achieved"] == 2 and by_label["Alpha moral"]["target_count"] == 2
    assert by_label["Alpha moral"]["rejected_over_cap"] == 2
    assert by_label["Beta moral"]["achieved"] == 1
    assert sum("Alpha moral" in p["variables"]["moral_considerations"] for p in result) == 2
    for entry in report["entries"]:
        assert entry["achieved"] <= entry["target_count"]


def test_resuming_from_rows_holding_over_cap_outcomes_rebuilds_the_state_and_re_asks_those_documents(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    plans = make_plans(4)
    rows = [
        {"prompt_id": f"matrix_{i:06d}", "pass": 1, "wave": i + 1, "outcomes": [
            {"list": "moral", "label": "Alpha moral", "reason": "r", "status": status},
            {"list": "epistemic", "label": None, "reason": "", "status": "none"}]}
        for i, status in enumerate(["accepted", "accepted", "over_cap"])
    ]
    utils.save_jsonl(rows, stage_dir / select.SELECTIONS_FILENAME)
    attached = select.attach_selections(select_config, small_prompts, stage_dir, plans)
    assert picks_by_document(attached)["matrix_000002"] == []
    calls = stub_claude(say_none)
    result = select.run(select_config, small_prompts, stage_dir, plans)
    pass_one = [c["item_id"] for c in calls if c["item_id"].endswith("pass1")]
    assert pass_one == ["matrix_000002_pass1", "matrix_000003_pass1"]     # the rejected document first, then the unasked one
    reasked_prompt = next(c for c in calls if c["item_id"] == "matrix_000002_pass1")["user_message"]
    assert "Alpha moral" not in offered_in(reasked_prompt)["moral"]
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["Alpha moral"]["rejected_over_cap"] == 1 and by_label["Alpha moral"]["achieved"] == 2


def test_tentative_picks_are_held_while_calls_are_in_flight_and_released_after_answers_and_failures(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 2
    barrier = threading.Barrier(2, timeout=10)
    state = {"count": 0, "held_while_in_flight": None}
    lock = threading.Lock()
    tracker_holder = {}
    original = select.ConsiderationTracker

    class RecordingTracker(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tracker_holder["tracker"] = self

    def dispatcher(user_message, **kw):
        with lock:
            state["count"] += 1
            first_two = state["count"] <= 2
        if first_two:
            barrier.wait()      # both calls are in flight, each holding a tentative pick on its top moral entry
            with lock:
                state["held_while_in_flight"] = sum(tracker_holder["tracker"].in_flight_top_picks.values())
        if kw["item_id"] == "matrix_000001_pass1":
            raise RuntimeError("transient outage")      # a failing call must release too
        return say_none(user_message)

    stub_claude(dispatcher)
    select.ConsiderationTracker = RecordingTracker
    try:
        select.run(select_config, small_prompts, stage_dir, make_plans(6))
    finally:
        select.ConsiderationTracker = original
    assert state["held_while_in_flight"] >= 2
    assert sum(tracker_holder["tracker"].in_flight_top_picks.values()) == 0
    assert len(rows_of(stage_dir)) == 5      # the failed call saved nothing


def test_a_label_not_in_the_calls_own_offer_is_not_saved(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    # Delta digital is real in the catalog but was never offered to a non-digital document
    stub_claude(lambda user_message, **kw: answer(None, "Delta digital"))
    with pytest.raises(SystemExit, match="every pending selection call failed"):
        select.run(select_config, small_prompts, stage_dir, make_plans(3))
    assert rows_of(stage_dir) == []


def test_each_answer_is_written_as_it_completes_and_start_numbers_count_up_within_a_pass(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    rows_seen_at_call_start = []

    def dispatcher(user_message, **kw):
        rows_seen_at_call_start.append(len(rows_of(stage_dir)))
        return say_none(user_message)

    stub_claude(dispatcher)
    select.run(select_config, small_prompts, stage_dir, make_plans(5))
    assert rows_seen_at_call_start == [0, 1, 2, 3, 4]       # row k is on disk before call k+1 starts
    assert [r["wave"] for r in rows_of(stage_dir)] == [1, 2, 3, 4, 5]


def test_rolling_workers_record_every_document_once_and_resume_retries_only_the_failed_call(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 3
    plans = make_plans(8)
    state = {"failed": False}
    lock = threading.Lock()

    def dispatcher(user_message, **kw):
        with lock:
            if "Spec 5." in user_message and kw["item_id"].endswith("pass1") and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("transient outage")
        return say_none(user_message)

    stub_claude(dispatcher)
    select.run(select_config, small_prompts, stage_dir, plans)
    rows = rows_of(stage_dir)
    assert sorted(r["prompt_id"] for r in rows) == sorted(f"matrix_{i:06d}" for i in range(8) if i != 5)
    assert len({r["wave"] for r in rows}) == 7                      # distinct start sequence numbers
    assert not utils.Checkpoint(stage_dir / select.CHECKPOINT_FILENAME).is_done("pass_1")

    calls = stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, plans)
    assert [c["item_id"] for c in calls] == ["matrix_000005_pass1"]
    assert utils.Checkpoint(stage_dir / select.CHECKPOINT_FILENAME).is_done("pass_1")


def test_selection_files_from_older_runs_with_over_cap_rows_still_load(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(4)
    old_rows = [
        {"prompt_id": "matrix_000000", "pass": 1, "wave": 1, "outcomes": [
            {"list": "moral", "label": "Alpha moral", "reason": "r", "status": "accepted"},
            {"list": "epistemic", "label": None, "reason": "", "status": "none"}]},
        {"prompt_id": "matrix_000001", "pass": 1, "wave": 1, "outcomes": [
            {"list": "moral", "label": "Alpha moral", "reason": "r", "status": "over_cap"},
            {"list": "epistemic", "label": None, "reason": "", "status": "none"}]},
    ]
    utils.save_jsonl(old_rows, stage_dir / select.SELECTIONS_FILENAME)
    attached = select.attach_selections(select_config, small_prompts, stage_dir, plans)
    assert picks_by_document(attached)["matrix_000000"] == ["Alpha moral"]
    assert picks_by_document(attached)["matrix_000001"] == []
    stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, plans)
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    assert {e["label"]: e["rejected_over_cap"] for e in report["entries"]}["Alpha moral"] == 1


def test_documents_go_through_in_a_seeded_order_whatever_the_plan_order(select_config, small_prompts, tmp_path, stub_claude):
    select_config["workers"] = 1
    plans = make_plans(10)
    orders = []
    for name, order, seed in (("a", plans, 0), ("b", list(reversed(plans)), 0), ("c", plans, 1)):
        directory = tmp_path / name
        directory.mkdir()
        select_config["sdf"]["seed"] = seed
        calls = stub_claude(say_none)
        select.run(select_config, small_prompts, directory, order)
        orders.append([c["item_id"] for c in calls])
    assert orders[0] == orders[1] and orders[0] != orders[2]
    assert orders[0] != sorted(orders[0])    # shuffled, not left in id order


def test_no_stake_and_incoherent_documents_get_no_picks_and_no_calls(select_config, small_prompts, stage_dir, stub_claude):
    resolution = no_stake_resolution(small_prompts)
    plans = make_plans(6, no_stake={1}, incoherent={2}, resolution=resolution)
    calls = stub_claude(pick_first_offered)
    result = select.run(select_config, small_prompts, stage_dir, plans)
    asked = {c["item_id"].split("_pass")[0] for c in calls}
    assert asked == {"matrix_000000", "matrix_000003", "matrix_000004", "matrix_000005"}
    by_id = {p["prompt_id"]: p for p in result}
    assert by_id["matrix_000001"]["considerations_spec"] == "" and by_id["matrix_000001"]["variables"]["moral_considerations"] == []
    assert "considerations_spec" not in by_id["matrix_000002"]       # incoherent plans pass through untouched
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    assert report["eligible_documents"] == 4                         # targets use only the documents asked


def test_the_no_stake_skip_can_be_turned_off(select_config, small_prompts, stage_dir, stub_claude):
    select_config["sdf"]["select_skip_no_stake"] = False
    plans = make_plans(4, no_stake={1}, resolution=no_stake_resolution(small_prompts))
    calls = stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, plans)
    assert "matrix_000001" in {c["item_id"].split("_pass")[0] for c in calls}


def test_digital_only_entries_are_offered_only_on_digital_documents(select_config, small_prompts, stage_dir, stub_claude):
    calls = stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, make_plans(6, digital={2}))
    for call in calls:
        digital = call["item_id"].startswith("matrix_000002")
        assert ("Delta digital" in call["user_message"]) == digital


# --- passes ---

def test_pass_two_offers_only_under_target_entries_and_documents_can_carry_several(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    state = {"pass_one_seen": False}

    def dispatcher(user_message, **kw):
        if kw["item_id"].endswith("pass1"):
            # pass 1: only the document asked first picks Alpha; everything else says none
            if not state["pass_one_seen"]:
                state["pass_one_seen"] = True
                return answer("Alpha moral", None)
            return say_none(user_message)
        return pick_first_offered(user_message, **kw)

    calls = stub_claude(dispatcher)
    result = select.run(select_config, small_prompts, stage_dir, make_plans(10))
    pass_two = [c for c in calls if c["item_id"].endswith("pass2")]
    assert pass_two
    carrier = next(p["prompt_id"] for p in result if "Alpha moral" in p["variables"]["moral_considerations"])
    for call in pass_two:
        shown = offered_in(call["user_message"])
        if call["item_id"].startswith(carrier):
            assert "Alpha moral" not in shown["moral"]                     # never one it already has
            assert "Alpha moral" in call["user_message"].split("already carries")[1]
    assert max(len(p["variables"]["moral_considerations"]) for p in result) >= 2   # a document ended with two moral picks


def test_passes_stop_when_a_pass_adds_nothing(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    calls = stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, make_plans(6))
    assert {c["item_id"].rsplit("_", 1)[1] for c in calls} == {"pass1"}


def test_passes_stop_at_the_configured_maximum(select_config, small_prompts, stage_dir, stub_claude):
    select_config["sdf"]["max_selection_passes"] = 2
    # one pick per pass keeps every pass making progress, so only the maximum can stop the loop
    state = {"asked": 0}

    def one_pick_per_pass(user_message, **kw):
        state["asked"] += 1
        return answer("Beta moral", None) if state["asked"] % 6 == 1 else say_none(user_message)

    select_config["workers"] = 1
    calls = stub_claude(one_pick_per_pass)
    select.run(select_config, small_prompts, stage_dir, make_plans(6))
    assert {c["item_id"].rsplit("_", 1)[1] for c in calls} <= {"pass1", "pass2"}


def test_passes_stop_when_every_target_is_met(select_config, small_prompts, stage_dir, stub_claude):
    select_config["sdf"]["max_selection_passes"] = 5
    catalog_text = SMALL_CONSIDERATIONS.replace("0.9 :: Beta moral", "0.1 :: Beta moral").replace("0.2 :: Alpha", "0.1 :: Alpha")
    (small_prompts / "considerations.txt").write_text(catalog_text, encoding="utf-8")
    select_config["workers"] = 1
    calls = stub_claude(pick_first_offered)
    select.run(select_config, small_prompts, stage_dir, make_plans(10, digital=set(range(10))))
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    assert all(e["achieved"] == e["target_count"] for e in report["entries"])
    assert max(int(c["item_id"].rsplit("pass", 1)[1]) for c in calls) < 5


# --- checkpoints and resume ---

def test_resume_after_a_completed_stage_makes_zero_calls_and_returns_the_same_picks(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(10, digital={0, 1})
    stub_claude(pick_first_offered)
    first = select.run(select_config, small_prompts, stage_dir, plans)
    rows_before = rows_of(stage_dir)
    calls = stub_claude([])
    again = select.run(select_config, small_prompts, stage_dir, plans)
    assert calls == []
    assert picks_by_document(again) == picks_by_document(first)
    assert [p.get("considerations_spec") for p in again] == [p.get("considerations_spec") for p in first]
    assert rows_of(stage_dir) == rows_before


def test_resume_after_a_pass_that_added_nothing_does_not_start_another_pass(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(6)
    stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, plans)
    calls = stub_claude(pick_first_offered)     # a new pass would ask and would find picks
    select.run(select_config, small_prompts, stage_dir, plans)
    assert calls == []


def test_a_killed_run_resumes_from_the_saved_rows_without_re_asking(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    plans = make_plans(10)
    stub_claude(pick_first_offered)
    uninterrupted = select.run(select_config, small_prompts, stage_dir, plans)
    all_rows = rows_of(stage_dir)

    partial_dir = stage_dir.parent / "partial"
    partial_dir.mkdir()
    utils.save_jsonl(all_rows[:4], partial_dir / select.SELECTIONS_FILENAME)    # killed after four answered calls
    calls = stub_claude(pick_first_offered)
    resumed = select.run(select_config, small_prompts, partial_dir, plans)
    asked_before = {r["prompt_id"] for r in all_rows[:4] if r["pass"] == 1}
    assert not any(c["item_id"].split("_pass")[0] in asked_before and c["item_id"].endswith("pass1") for c in calls)
    assert picks_by_document(resumed) == picks_by_document(uninterrupted)


def test_malformed_answer_is_not_saved_and_resume_retries_exactly_that_document(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    plans = make_plans(6)
    state = {"failed": False}

    def dispatcher(user_message, **kw):
        if "Spec 2." in user_message and kw["item_id"].endswith("pass1") and not state["failed"]:
            state["failed"] = True
            return "I think the first moral entry fits."          # no JSON
        return say_none(user_message)

    stub_claude(dispatcher)
    select.run(select_config, small_prompts, stage_dir, plans)
    asked = {r["prompt_id"] for r in rows_of(stage_dir)}
    assert "matrix_000002" not in asked and len(asked) == 5
    assert not utils.Checkpoint(stage_dir / select.CHECKPOINT_FILENAME).is_done("pass_1")

    calls = stub_claude(say_none)
    select.run(select_config, small_prompts, stage_dir, plans)
    assert [c["item_id"] for c in calls] == ["matrix_000002_pass1"]


@pytest.mark.parametrize("bad_answer", [
    answer("Not a real label", None),
    '{"moral": {"label": "Gamma epistemic", "where_it_arises": "x"}, "epistemic": {"label": "none"}}',
    ("{\"moral\": {\"label\": \"none\"}", "max_tokens"),
    "",
])
def test_unknown_label_and_truncated_answers_are_not_saved(select_config, small_prompts, stage_dir, stub_claude, bad_answer):
    stub_claude(lambda user_message, **kw: bad_answer)
    with pytest.raises(SystemExit, match="every pending selection call failed"):
        select.run(select_config, small_prompts, stage_dir, make_plans(4))
    assert rows_of(stage_dir) == []
    assert utils.Checkpoint(stage_dir / select.CHECKPOINT_FILENAME).done_count == 0


def test_an_api_error_on_every_call_exits_but_one_failure_does_not(select_config, small_prompts, stage_dir, stub_claude):
    def always_fails(user_message, **kw):
        raise RuntimeError("backend down")

    stub_claude(always_fails)
    with pytest.raises(SystemExit, match="systemic"):
        select.run(select_config, small_prompts, stage_dir, make_plans(4))

    def one_fails(user_message, **kw):
        if "Spec 1." in user_message:
            raise RuntimeError("poison document")
        return say_none(user_message)

    stub_claude(one_fails)
    select.run(select_config, small_prompts, stage_dir, make_plans(4))      # does not raise
    assert "matrix_000001" not in {r["prompt_id"] for r in rows_of(stage_dir)}


# --- switches and the snapshot ---

def test_model_falls_back_from_select_to_plan_to_global(select_config, small_prompts, stage_dir, stub_claude, tmp_path):
    calls = stub_claude(say_none)
    plans = make_plans(2)
    select.run(select_config, small_prompts, stage_dir, plans)
    assert calls[-1]["model"] is None
    select_config["sdf"]["plan_model"] = "plan-model-x"
    second = tmp_path / "second"
    second.mkdir()
    select.run(select_config, small_prompts, second, plans)
    assert calls[-1]["model"] == "plan-model-x"
    select_config["sdf"]["select_model"] = "select-model-x"
    third = tmp_path / "third"
    third.mkdir()
    select.run(select_config, small_prompts, third, plans)
    assert calls[-1]["model"] == "select-model-x"


def test_disabled_selector_makes_no_calls_and_writes_nothing(select_config, small_prompts, stage_dir, stub_claude):
    select_config["sdf"]["select_enabled"] = False
    calls = stub_claude([])
    plans = make_plans(4)
    assert select.run(select_config, small_prompts, stage_dir, plans) == plans
    assert select.attach_selections(select_config, small_prompts, stage_dir, plans) == plans
    assert calls == [] and list(stage_dir.iterdir()) == []


def test_snapshot_without_the_selector_files_skips_the_stage(select_config, tmp_path, stage_dir, stub_claude):
    calls = stub_claude([])
    old_snapshot = tmp_path / "old_prompts"
    old_snapshot.mkdir()
    plans = make_plans(4)
    assert select.run(select_config, old_snapshot, stage_dir, plans) == plans
    assert select.attach_selections(select_config, old_snapshot, stage_dir, plans) == plans
    assert calls == [] and list(stage_dir.iterdir()) == []


def test_snapshot_with_the_template_but_not_the_considerations_file_skips_the_stage(select_config, small_prompts, stage_dir, stub_claude):
    (small_prompts / "considerations.txt").unlink()
    calls = stub_claude([])
    plans = make_plans(4)
    assert select.run(select_config, small_prompts, stage_dir, plans) == plans and calls == []


# --- resuming at layer 2 ---

def test_attach_selections_gives_the_same_picks_as_the_uninterrupted_run_without_calls(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(10, digital={0, 5})
    stub_claude(pick_first_offered)
    uninterrupted = select.run(select_config, small_prompts, stage_dir, plans)
    calls = stub_claude([])
    reattached = select.attach_selections(select_config, small_prompts, stage_dir, plans)
    assert calls == []
    assert reattached == uninterrupted


def test_attach_selections_ignores_documents_the_screen_removed_later(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(10)
    stub_claude(pick_first_offered)
    select.run(select_config, small_prompts, stage_dir, plans)
    survivors = plans[:7]
    attached = select.attach_selections(select_config, small_prompts, stage_dir, survivors)
    assert [p["prompt_id"] for p in attached] == [p["prompt_id"] for p in survivors]


# --- reporting ---

def test_load_considerations_outputs_is_none_for_a_run_without_the_selector(tmp_path):
    (tmp_path / "layer1").mkdir()
    assert select.load_considerations_outputs(tmp_path) is None
    assert select.load_considerations_outputs(tmp_path / "does_not_exist") is None


def test_load_considerations_outputs_reads_rows_report_and_the_runs_own_catalog(select_config, small_prompts, tmp_path, stub_claude):
    run_dir = tmp_path / "run"
    stage = run_dir / "layer1"
    stage.mkdir(parents=True)
    snapshot = run_dir / "inputs" / "prompts"
    snapshot.mkdir(parents=True)
    for source in small_prompts.iterdir():
        (snapshot / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    stub_claude(pick_first_offered)
    result = select.run(select_config, snapshot, stage, make_plans(10))
    outputs = select.load_considerations_outputs(run_dir)
    assert len(outputs["rows"]) == len(rows_of(stage)) and outputs["report"]["eligible_documents"] == 10
    corpus = [{"variables": p["variables"]} for p in result]
    report = select.corpus_coverage_report(outputs, corpus)
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["Alpha moral"]["achieved"] == 2


def test_audit_reports_the_considerations_coverage_on_the_final_corpus(select_config, small_prompts, tmp_path, stub_claude, capsys):
    run_dir = tmp_path / "run"
    stage = run_dir / "layer1"
    stage.mkdir(parents=True)
    snapshot = run_dir / "inputs" / "prompts"
    snapshot.mkdir(parents=True)
    for source in small_prompts.iterdir():
        (snapshot / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    stub_claude(pick_first_offered)
    result = select.run(select_config, snapshot, stage, make_plans(10))
    capsys.readouterr()
    report = {}
    audit_sdf.audit_considerations(run_dir, [{"variables": p["variables"]} for p in result], report)
    output = capsys.readouterr().out
    assert "Alpha moral" in output and "target 20%" in output and "no consideration from either list" in output
    assert report["considerations"]["final_corpus"]["eligible_documents"] == 10
    assert report["considerations"]["layer1"]["eligible_documents"] == 10


def test_audit_still_works_on_a_run_without_selection_files(tmp_path, capsys):
    (tmp_path / "layer1").mkdir()
    report = {}
    audit_sdf.audit_considerations(tmp_path, [], report)
    assert "no selection files" in capsys.readouterr().out
    assert "considerations" not in report


# --- the real files, the drafter, and the experiment-invalidating checks ---

def test_layer1_renders_with_no_consideration_text_and_no_label_wording(prompts_sdf):
    template = (prompts_sdf / "layer1.txt").read_text(encoding="utf-8")
    assert "reasoning_featured" not in template and "moral or constitutional" not in template
    assert "reasoning_featured" not in cp.matrix_axes(template)
    values, _ = cp.split_weights(cp.parse_variables(prompts_sdf / "variables.txt"))
    assert "reasoning_featured" not in values
    weights = {name: [1 / len(vals)] * len(vals) for name, vals in values.items()}
    catalog = load_considerations(prompts_sdf)
    records = list(cp.compose_records(
        template, values, weights, (prompts_sdf / "preamble.txt").read_text(encoding="utf-8"), n_prompts=30, seed=0))
    for record in records:
        planner_text = (record.get("system") or "") + record["prompt"]
        for entry in catalog.all_entries:
            assert entry.label not in planner_text
            assert not entry.gloss or entry.gloss not in planner_text
        assert "moral_considerations" not in record["variables"]


def draft_prompt_for(config, prompts_dir, tmp_path, stub_claude, plan):
    calls = stub_claude(lambda user_message, **kw: "<document>A draft.</document>")
    config["workers"] = 1
    output_dir = tmp_path / f"draft_{len(list(tmp_path.iterdir()))}"
    output_dir.mkdir()
    drafts = layer2_draft.run(config, prompts_dir, output_dir, [plan])
    return calls[0], drafts


def test_the_none_leak_test_a_document_without_picks_leaves_no_trace_in_its_draft_prompt(select_config, prompts_sdf, tmp_path, stub_claude):
    plan = make_plans(1)[0] | {"considerations_spec": ""}
    call, drafts = draft_prompt_for(select_config, prompts_sdf, tmp_path, stub_claude, plan)
    prompt = call["system_prompt"] + "\n" + call["user_message"]
    catalog = load_considerations(prompts_sdf)
    for entry in catalog.all_entries:
        assert entry.label not in prompt
    for phrase in ("considerations", "Hold them lightly", "label", "{", "reasoning_featured"):
        assert phrase not in call["user_message"], phrase
    assert "<document_description>" in call["user_message"] and "Output a fragment" in call["user_message"]
    assert "\n\n\n" not in call["user_message"]                    # the dropped line leaves no double gap
    assert drafts[0]["considerations_spec"] == ""
    assert "none" not in call["user_message"].lower().split("</document_description>")[1].split("Output a fragment")[0]


def test_a_document_with_picks_gets_its_spec_in_the_draft_prompt_and_records(select_config, prompts_sdf, tmp_path, stub_claude):
    spec = ("Substrate neutrality (a gloss) — Where it arises here.\n"
            "Weighing evidence for sentience (another gloss) — Where that arises here.")
    plan = make_plans(1)[0] | {"considerations_spec": spec}
    call, drafts = draft_prompt_for(select_config, prompts_sdf, tmp_path, stub_claude, plan)
    assert spec in call["user_message"]
    assert call["user_message"].index("</document_description>") < call["user_message"].index(spec) < call["user_message"].index("Output a fragment")
    assert "never reuse the wording of its label" in call["user_message"]
    assert drafts[0]["considerations_spec"] == spec


def test_the_considerations_spec_survives_onto_the_draft_rewrite_and_final_records(select_config, prompts_sdf, tmp_path, stub_claude):
    from sdf_pipeline import layer3_rewrite, layer4_score
    spec = "Scope sensitivity (a gloss) — Where it arises here."
    plan = make_plans(1)[0] | {"considerations_spec": spec}
    _, drafts = draft_prompt_for(select_config, prompts_sdf, tmp_path, stub_claude, plan)
    stub_claude(lambda user_message, **kw: "Review.\n<improved_document>Better.</improved_document>")
    rewrite_dir = tmp_path / "rewrite"
    rewrite_dir.mkdir()
    rewrites = layer3_rewrite.run(select_config, prompts_sdf, rewrite_dir, drafts)
    assert rewrites[0]["considerations_spec"] == spec
    stub_claude(lambda user_message, **kw: json.dumps(
        {"alignment": 9, "realism": 9, "spec_conformance": 9, "notes": ""}))
    score_dir, final_dir = tmp_path / "score", tmp_path / "final"
    score_dir.mkdir()
    final_dir.mkdir()
    final = layer4_score.run(select_config, prompts_sdf, score_dir, final_dir, rewrites)
    assert final[0]["considerations_spec"] == spec
    assert utils.load_jsonl(final_dir / "sdf_corpus.jsonl")[0]["considerations_spec"] == spec


def test_a_plan_from_an_old_run_without_the_spec_still_drafts(select_config, prompts_sdf, tmp_path, stub_claude):
    plan = make_plans(1)[0]
    plan["variables"]["reasoning_featured"] = "individualism"
    call, drafts = draft_prompt_for(select_config, prompts_sdf, tmp_path, stub_claude, plan)
    assert "considerations" not in call["user_message"]
    assert drafts[0]["considerations_spec"] == ""


def test_an_old_prompt_snapshot_that_still_renders_reasoning_featured_keeps_working(select_config, prompts_sdf, tmp_path, stub_claude):
    old = tmp_path / "old_prompts"
    old.mkdir()
    for source in prompts_sdf.iterdir():
        if source.is_file():
            (old / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    layer2 = (old / "layer2.txt").read_text(encoding="utf-8")
    line_start = layer2.index("The scenario raises the considerations")
    line_end = layer2.index("\n", line_start)
    (old / "layer2.txt").write_text(
        layer2[:line_start] + "The animal or welfare related section of the document might feature {reasoning_featured}."
        + layer2[line_end:], encoding="utf-8")
    plan = make_plans(1)[0] | {"considerations_spec": "ignored by the old template"}
    plan["variables"]["reasoning_featured"] = "OLD-RF-VALUE"
    call, _ = draft_prompt_for(select_config, old, tmp_path, stub_claude, plan)
    assert "OLD-RF-VALUE" in call["user_message"]
    assert "ignored by the old template" not in call["user_message"]
