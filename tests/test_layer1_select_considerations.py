"""Behavior tests for the layer-1 considerations selector (the model scores, code assigns), driven
through layer1_select_considerations.run() and score_document() with a stubbed
shared.api.call_claude (no test here may reach the API; see conftest). The real
selector template stays in play. Most stage tests use a small considerations
file written to tmp_path (so exact counts are knowable); the checks about the
real file, the real templates and the drafter use the repo's own files.

Money paths: well-formed answers parse, unusable answers are retried and recorded
without crashing the run, resume makes zero calls for scored documents, and a
failure on any call stops the stage before it assigns anything.
"""

import json
import re
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
    return tiny_config


def no_stake_resolution(prompts_dir):
    return sorted(load_considerations(prompts_dir).skipped_resolutions)[0]


_REAL_VARIABLES = cp.load_variables(Path(__file__).resolve().parent.parent / "prompts" / "sdf" / "variables.txt")[0]
COMPLETE_HAND = {name: cards[0] for name, cards in _REAL_VARIABLES.items()}   # the later stages render every dealt variable


def make_plans(count=10, digital=(), no_stake=(), incoherent=(), resolution=None):
    plans = []
    for i in range(count):
        variables = {
            **COMPLETE_HAND,
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
    """{'moral': [labels], 'epistemic': [labels]} in the order the scoring prompt showed them."""
    shown = {}
    for name in ("moral", "epistemic"):
        body = re.search(rf"<{name}_list>\n(.*?)\n</{name}_list>", user_message, re.DOTALL).group(1)
        shown[name] = [line[2:].split(": ")[0] for line in body.splitlines() if line.startswith("- ")]
    return shown


def answer_with(user_message, score_of):
    """A well-formed answer scoring every shown label with ``score_of(label)``."""
    shown = offered_in(user_message)
    return json.dumps({
        name: [{"label": label, "score": score_of(label),
                "where_it_arises": f"Where {label} arises." if score_of(label) >= 3 else ""}
               for label in labels]
        for name, labels in shown.items() if labels})


def score_everything_five(user_message, **kw):
    return answer_with(user_message, lambda label: 5)


def score_everything_one(user_message, **kw):
    return answer_with(user_message, lambda label: 1)


def rows_of(stage_dir, filename=select.SELECTIONS_FILENAME):
    return utils.load_jsonl(stage_dir / filename)


def picks_by_document(plans):
    return {p["prompt_id"]: p["variables"]["moral_considerations"] + p["variables"]["epistemic_considerations"]
            for p in plans if p.get("description")}


def small_offer(small_prompts, digital=False):
    catalog = load_considerations(small_prompts)
    return catalog, {"moral": list(catalog.moral),
                     "epistemic": [e for e in catalog.epistemic if digital or not e.digital_minds_only]}


# --- eligibility, shuffling, the prompt and the schema ---

def test_digital_minds_only_entries_are_eligible_only_on_digital_mind_documents(small_prompts):
    catalog = load_considerations(small_prompts)
    ordinary = select.eligible_entries_for_document(catalog, {"sentient_category": "farmed fishes"})
    digital = select.eligible_entries_for_document(catalog, {"sentient_category": DIGITAL_CATEGORY})
    assert [e.label for e in ordinary["epistemic"]] == ["Gamma epistemic"]
    assert [e.label for e in digital["epistemic"]] == ["Delta digital", "Gamma epistemic"]
    assert [e.label for e in ordinary["moral"]] == ["Alpha moral", "Beta moral"]


def test_the_shuffle_is_deterministic_per_seed_and_prompt_id_and_keeps_every_entry(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    offered = {"moral": list(catalog.moral), "epistemic": list(catalog.epistemic)}
    first = select.shuffled_offer(offered, 0, "matrix_000001")
    assert select.shuffled_offer(offered, 0, "matrix_000001") == first
    assert {e.label for e in first["moral"]} == {e.label for e in catalog.moral}
    assert len(first["epistemic"]) == len(catalog.epistemic)
    orders = {tuple(e.label for e in select.shuffled_offer(offered, 0, f"matrix_{i:06d}")["moral"]) for i in range(8)}
    assert len(orders) > 1                                                    # differs between documents
    assert select.shuffled_offer(offered, 1, "matrix_000001") != first        # and between seeds


def test_the_scoring_prompt_shows_every_entry_the_scale_and_no_target_or_fullness_information(small_prompts):
    catalog, offered = small_offer(small_prompts, digital=True)
    facts = select.DocumentFacts("A scenario about fish.", "a news article", "central", "farmed fishes")
    system, user = select.build_scoring_prompt(select.load_selection_template(small_prompts), facts, offered)
    for entry in catalog.all_entries:
        assert entry.display in user
    assert "1 = it is not coherent with the setup." in user and "5 = it is already in the description, or something would feel missing without it." in user
    assert "A scenario about fish." in user and system
    for phrase in ("target", "full", "already used", "first entry", "first fit"):
        assert phrase not in user.lower(), phrase
    assert "{" not in user


def test_the_schema_has_one_item_type_per_list_with_labels_in_the_shuffled_order(small_prompts):
    catalog, offered = small_offer(small_prompts, digital=True)
    shuffled = select.shuffled_offer(offered, 0, "matrix_000003")
    schema = select.build_scoring_schema(shuffled)
    assert schema["required"] == ["moral", "epistemic"] and schema["additionalProperties"] is False
    for name in ("moral", "epistemic"):
        item = schema["properties"][name]["items"]
        assert item["properties"]["label"]["enum"] == [e.label for e in shuffled[name]]
        assert item["properties"]["score"] == {"type": "integer", "enum": [1, 2, 3, 4, 5]}
        assert item["required"] == ["label", "score", "where_it_arises"] and item["additionalProperties"] is False
    assert all('"' not in e.label for e in catalog.all_entries)


def test_a_list_with_no_entries_is_left_out_of_the_schema():
    assert select.build_scoring_schema({"moral": [], "epistemic": []})["required"] == []


# --- parsing and validation ---

def good_answer(offered, score=4):
    return {name: [{"label": e.label, "score": score, "where_it_arises": "Here."} for e in entries]
            for name, entries in offered.items() if entries}


def test_a_complete_answer_parses_to_scores_per_label(small_prompts):
    _, offered = small_offer(small_prompts)
    parsed = select.parse_scoring_answer(json.dumps(good_answer(offered)), offered)
    assert parsed["moral"]["Alpha moral"] == {"score": 4, "where_it_arises": "Here."}
    assert set(parsed["epistemic"]) == {"Gamma epistemic"}


def test_where_it_arises_is_blanked_below_three_and_required_from_three(small_prompts):
    _, offered = small_offer(small_prompts)
    answer = good_answer(offered, 2)
    parsed = select.parse_scoring_answer(json.dumps(answer), offered)
    assert parsed["moral"]["Alpha moral"] == {"score": 2, "where_it_arises": ""}
    answer["moral"][0]["score"] = 3
    answer["moral"][0]["where_it_arises"] = ""
    with pytest.raises(select.ScoringAnswerError, match="without saying where"):
        select.parse_scoring_answer(json.dumps(answer), offered)


@pytest.mark.parametrize("damage,message", [
    (lambda a: a["moral"].pop(), "no score for"),
    (lambda a: a["moral"].append({"label": "Not a label", "score": 3, "where_it_arises": "x"}), "not on the list shown"),
    (lambda a: a["moral"].append(dict(a["moral"][0])), "more than once"),
    (lambda a: a["moral"][0].update(score=0), "invalid score"),
    (lambda a: a["moral"][0].update(score=6), "invalid score"),
    (lambda a: a["moral"][0].update(score="4"), "invalid score"),
    (lambda a: a["moral"][0].update(score=True), "invalid score"),
    (lambda a: a["moral"][0].update(score=4.0), "invalid score"),
    (lambda a: a.pop("epistemic"), "missing or not an array"),
    (lambda a: a["moral"].append({"score": 3}), "malformed item"),
])
def test_invalid_answers_are_rejected(small_prompts, damage, message):
    _, offered = small_offer(small_prompts)
    answer = good_answer(offered)
    damage(answer)
    with pytest.raises(select.ScoringAnswerError, match=message):
        select.parse_scoring_answer(json.dumps(answer), offered)


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", "null"])
def test_unparseable_answers_are_rejected(small_prompts, raw):
    _, offered = small_offer(small_prompts)
    with pytest.raises(select.ScoringAnswerError):
        select.parse_scoring_answer(raw, offered)


def test_scores_for_a_list_that_was_not_offered_are_rejected(small_prompts):
    _, offered = small_offer(small_prompts)
    answer = good_answer(offered)
    with pytest.raises(select.ScoringAnswerError, match="not offered"):
        select.parse_scoring_answer(json.dumps(answer), {"moral": offered["moral"], "epistemic": []})


FACTS = select.DocumentFacts("A scenario.", "a news article", "central", "farmed fishes")


def test_score_document_retries_an_unusable_answer_then_succeeds(stub_claude, small_prompts):
    _, offered = small_offer(small_prompts)
    calls = stub_claude(["not json", (answer_with_labels(offered), "max_tokens"), answer_with_labels(offered)])
    scores, attempts = select.score_document(
        FACTS, offered, template=select.load_selection_template(small_prompts), item_id="doc", max_attempts=3)
    assert attempts == 3 and len(calls) == 3 and scores["moral"]["Beta moral"]["score"] == 4
    assert calls[0]["stage"] == "layer1_select" and calls[0]["json_schema"] == select.build_scoring_schema(offered)


def answer_with_labels(offered):
    return json.dumps(good_answer(offered))


def test_score_document_gives_up_after_the_attempt_limit(stub_claude, small_prompts):
    _, offered = small_offer(small_prompts)
    calls = stub_claude(["not json"] * 5)
    with pytest.raises(select.ScoringAnswerError, match="after 2 attempts"):
        select.score_document(FACTS, offered, template=select.load_selection_template(small_prompts), max_attempts=2)
    assert len(calls) == 2


# --- the stage ---

def test_every_eligible_document_gets_one_call_and_skipped_ones_get_none(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(8, digital={0}, no_stake={1, 2}, incoherent={3}, resolution=no_stake_resolution(small_prompts))
    calls = stub_claude(score_everything_five)
    select.run(select_config, small_prompts, stage_dir, plans)
    assert sorted(c["item_id"] for c in calls) == [f"matrix_{i:06d}" for i in (0, 4, 5, 6, 7)]
    for call in calls:
        digital = call["item_id"] == "matrix_000000"
        assert ("Delta digital" in call["user_message"]) == digital
        assert call["stage"] == "layer1_select" and call["json_schema"] is not None


def test_the_entry_order_differs_between_documents_and_matches_the_schema(select_config, prompts_sdf, stage_dir, stub_claude):
    calls = stub_claude(score_everything_one)
    select.run(select_config, prompts_sdf, stage_dir, make_plans(10))
    orders = {tuple(offered_in(c["user_message"])["moral"]) for c in calls}
    assert len(orders) > 1
    for call in calls:
        shown = offered_in(call["user_message"])
        assert call["json_schema"]["properties"]["moral"]["items"]["properties"]["label"]["enum"] == shown["moral"]


def test_end_to_end_writes_scores_selections_and_coverage_and_respects_targets(select_config, small_prompts, stage_dir, stub_claude):
    select_config["sdf"].update(max_moral_considerations_per_document=1, max_epistemic_considerations_per_document=1,
                                max_considerations_per_document=2)   # one per list, so Beta moral is capped below its target
    plans = make_plans(10, digital={0, 5})
    stub_claude(score_everything_five)
    result = select.run(select_config, small_prompts, stage_dir, plans)
    scores = rows_of(stage_dir, select.SCORES_FILENAME)
    assert len(scores) == 10 and scores[0]["scores"]["moral"]
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["Alpha moral"]["achieved"] == 2 and by_label["Alpha moral"]["target_count"] == 2
    assert by_label["Delta digital"]["target_count"] == 1 and by_label["Delta digital"]["achieved"] == 1
    assert by_label["Beta moral"]["achieved"] == 8
    assert all(e["achieved"] <= e["target_count"] for e in report["entries"])
    assert "rejected_over_cap" not in by_label["Alpha moral"]
    assert by_label["Alpha moral"]["score_histogram"]["5"] == 10
    picks = picks_by_document(result)
    assert all(len([l for l in labels if l in ("Alpha moral", "Beta moral")]) == 1 for labels in picks.values())
    assert all(len(labels) <= 2 for labels in picks.values())
    plan = next(p for p in result if p["variables"]["epistemic_considerations"])
    assert plan["considerations_spec"].count("\n") + 1 == len(picks[plan["prompt_id"]])
    assert "arises" in plan["considerations_spec"]
    selection = rows_of(stage_dir)
    assert len(selection) == 10 and all(o["status"] in ("accepted", "none") for r in selection for o in r["outcomes"])
    assert all("score" in o for r in selection for o in r["outcomes"] if o["status"] == "accepted")


def test_documents_scoring_below_the_threshold_get_nothing_and_are_reported(select_config, small_prompts, stage_dir, stub_claude):
    stub_claude(score_everything_one)
    result = select.run(select_config, small_prompts, stage_dir, make_plans(6))
    assert all(not labels for labels in picks_by_document(result).values())
    assert all(p["considerations_spec"] == "" for p in result)
    report = json.loads((stage_dir / select.COVERAGE_FILENAME).read_text())
    assert report["documents_with_no_consideration"] == 6 and len(report["document_ids_with_no_consideration"]) == 6


def test_the_caps_from_config_allow_several_considerations_per_document(select_config, prompts_sdf, stage_dir, stub_claude):
    select_config["sdf"].update(max_moral_considerations_per_document=2, max_epistemic_considerations_per_document=2,
                                max_considerations_per_document=3)
    stub_claude(score_everything_five)
    result = select.run(select_config, prompts_sdf, stage_dir, make_plans(10))
    counts = [len(v) for v in picks_by_document(result).values()]
    assert max(counts) == 3 and all(c <= 3 for c in counts)
    for plan in result:
        assert len(plan["variables"]["moral_considerations"]) <= 2 and len(plan["variables"]["epistemic_considerations"]) <= 2


# --- checkpoints and resume ---

def test_a_rerun_scores_nothing_that_is_already_scored(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(6)
    stub_claude(score_everything_five)
    first = select.run(select_config, small_prompts, stage_dir, plans)
    calls = stub_claude([])
    second = select.run(select_config, small_prompts, stage_dir, plans)
    assert calls == [] and second == first


def test_a_resume_scores_only_the_documents_without_a_saved_row(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(6)
    stub_claude(score_everything_five)
    select.run(select_config, small_prompts, stage_dir, plans)
    path = stage_dir / select.SCORES_FILENAME
    kept = rows_of(stage_dir, select.SCORES_FILENAME)
    path.write_text("".join(json.dumps(r) + "\n" for r in kept[:4]), encoding="utf-8")
    calls = stub_claude(score_everything_five)
    result = select.run(select_config, small_prompts, stage_dir, plans)
    assert sorted(c["item_id"] for c in calls) == sorted(r["prompt_id"] for r in kept[4:])
    assert len(rows_of(stage_dir, select.SCORES_FILENAME)) == 6 and len(rows_of(stage_dir)) == 6
    assert all("considerations_spec" in p for p in result)


def test_scores_are_checkpointed_as_they_arrive(select_config, small_prompts, stage_dir, stub_claude):
    select_config["workers"] = 1
    seen = []

    def dispatcher(user_message, **kw):
        seen.append(len(rows_of(stage_dir, select.SCORES_FILENAME)))
        return score_everything_five(user_message)
    stub_claude(dispatcher)
    select.run(select_config, small_prompts, stage_dir, make_plans(4))
    assert seen == [0, 1, 2, 3]


def test_one_unusable_document_stops_the_stage_before_assigning_and_is_rescored_on_resume(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(5)
    bad_id = "matrix_000002"

    def dispatcher(user_message, **kw):
        return "not json" if kw["item_id"] == bad_id else score_everything_five(user_message)
    calls = stub_claude(dispatcher)
    with pytest.raises(SystemExit, match="1 of 5 scoring calls failed"):
        select.run(select_config, small_prompts, stage_dir, plans)
    assert len([c for c in calls if c["item_id"] == bad_id]) == 3                    # max_scoring_attempts
    failures = rows_of(stage_dir, select.FAILURES_FILENAME)
    assert [f["prompt_id"] for f in failures] == [bad_id] and "unusable answer" in failures[0]["problem"]
    assert bad_id not in {r["prompt_id"] for r in rows_of(stage_dir, select.SCORES_FILENAME)}
    assert not (stage_dir / select.SELECTIONS_FILENAME).exists()
    assert not (stage_dir / select.COVERAGE_FILENAME).exists()
    calls = stub_claude(score_everything_five)
    result = select.run(select_config, small_prompts, stage_dir, plans)
    assert [c["item_id"] for c in calls] == [bad_id]
    assert all(p["considerations_spec"] for p in result)


def test_a_failure_on_every_call_exits_as_systemic(select_config, small_prompts, stage_dir, stub_claude):
    def raising(user_message, **kw):
        raise RuntimeError("backend down")
    stub_claude(raising)
    with pytest.raises(SystemExit, match="every pending scoring call failed"):
        select.run(select_config, small_prompts, stage_dir, make_plans(3))


def test_model_falls_back_from_select_to_plan_to_global(select_config, small_prompts, stage_dir, stub_claude, tmp_path):
    plans = make_plans(2)
    calls = stub_claude(score_everything_five)
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


# --- switches and the snapshot ---

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
    stub_claude(score_everything_five)
    uninterrupted = select.run(select_config, small_prompts, stage_dir, plans)
    calls = stub_claude([])
    reattached = select.attach_selections(select_config, small_prompts, stage_dir, plans)
    assert calls == []
    assert reattached == uninterrupted


def test_attach_selections_ignores_documents_the_screen_removed_later(select_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(10)
    stub_claude(score_everything_five)
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
    stub_claude(score_everything_five)
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
    stub_claude(score_everything_five)
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
    # The dealt choices sit in this stretch too, and the dealt failure mode can read "none in particular".
    stretch_after_description = call["user_message"].lower().split("</document_description>")[1].split("Output a fragment")[0]
    assert "none" not in stretch_after_description.replace("none in particular", "")


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
