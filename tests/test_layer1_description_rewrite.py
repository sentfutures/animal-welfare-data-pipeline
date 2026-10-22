"""Behavior tests for the layer-1 description rewrite, driven through
layer1_description_rewrite.run() and rewrite_description() with a stubbed
shared.api.call_claude (no test here may reach the API; see conftest). The real
prompt template, real variables and real deal rules stay in play.

Money paths: a decline passes the original untouched, a rewrite replaces
``description`` and keeps ``original_description``, a dealt-value conflict drops
the plan, a malformed answer is not checkpointed, and resume repeats nothing.
"""

import json

import pytest

from evals import audit_sdf
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer1_description_rewrite as rewrite
from shared import utils


def answer(rewrite_needed=False, description="", checks=("plausibility_organisation",), conflict=False,
           conflict_reason="", **overrides):
    changes = [{"check": c, "reason": f"Fixed under {c}."} for c in checks] if rewrite_needed else []
    return json.dumps({
        "reasoning": "Read it against the dealt values.", "rewrite_needed": rewrite_needed,
        "rewritten_description": description, "changes": changes,
        "dealt_value_conflict": conflict, "conflict_reason": conflict_reason, **overrides})


def description_in(user_message):
    return user_message.split("<description>\n")[1].split("\n</description>")[0]


def make_plans(prompts_sdf, count=6):
    template = (prompts_sdf / "layer1.txt").read_text(encoding="utf-8")
    values, weights, value_ids = cp.load_variables(prompts_sdf / "variables.txt")
    rules = cp.load_deal_rules(prompts_sdf / "deal_constraints.txt", value_ids)
    hands = cp.hand_records(cp.deal_hands(template, values, weights, count, 0, rules))
    return [
        {"prompt_id": h["prompt_id"], "variables": h["variables"], "plan": "…",
         "description": f"Original description {i}.", "incoherent": False}
        for i, h in enumerate(hands)
    ]


@pytest.fixture
def stage_dir(tmp_path):
    directory = tmp_path / "layer1"
    directory.mkdir()
    return directory


@pytest.fixture
def plans(prompts_sdf):
    return make_plans(prompts_sdf)


def by_id(records):
    return {r["prompt_id"]: r for r in records}


# --- parsing and the public function ---

CHECKS = ["plausibility_organisation", "organisation_naming"]


@pytest.mark.parametrize("raw", [
    "not json",
    json.dumps({"reasoning": "r"}),                                              # missing fields
    answer(rewrite_needed=True, description=""),                                 # rewrite without text
    answer(rewrite_needed=True, description="New.", checks=()),                  # rewrite without a change
    answer(rewrite_needed=True, description="New.", checks=("made_up_check",)),  # unknown check
    answer(rewrite_needed=False, description="Stray text."),                     # decline carrying a rewrite
    answer(rewrite_needed=True, description="New.", conflict=True, conflict_reason="x"),  # rewrite and conflict
    answer(conflict=True, conflict_reason="  "),                                 # conflict without a reason
    answer(rewrite_needed="yes"),                                                # wrong type
])
def test_parse_rejects_unusable_or_self_contradictory_answers(raw):
    with pytest.raises(rewrite.RewriteAnswerError):
        rewrite.parse_rewrite_answer(raw, CHECKS)


def test_parse_the_three_outcomes():
    declined = rewrite.parse_rewrite_answer(answer(), CHECKS)
    assert not declined.rewrite_needed and declined.rewritten_description == "" and declined.changes == ()
    rewritten = rewrite.parse_rewrite_answer(answer(True, " New text. ", ("organisation_naming",)), CHECKS)
    assert rewritten.rewrite_needed and rewritten.rewritten_description == "New text."
    assert rewritten.changes == ({"check": "organisation_naming", "reason": "Fixed under organisation_naming."},)
    conflict = rewrite.parse_rewrite_answer(answer(conflict=True, conflict_reason="Scale is dealt."), CHECKS)
    assert conflict.dealt_value_conflict and not conflict.rewrite_needed and conflict.conflict_reason == "Scale is dealt."


def test_rewrite_description_is_schema_constrained_and_shows_variables_and_description(stub_claude, prompts_sdf):
    calls = stub_claude(lambda user_message, **kw: answer())
    prompts = rewrite.load_rewrite_prompts(prompts_sdf)
    result = rewrite.rewrite_description("Type: LINES-X", "A description.", prompts=prompts, model="m", item_id="p1")
    assert not result.rewrite_needed
    call = calls[0]
    assert call["stage"] == "layer1_description_rewrite" and call["model"] == "m" and call["item_id"] == "p1"
    assert "Type: LINES-X" in call["user_message"] and description_in(call["user_message"]) == "A description."
    assert call["json_schema"]["required"][:2] == ["reasoning", "rewrite_needed"]
    assert call["temperature"] is None and call["system_prompt"]


def test_truncated_answer_is_rejected(stub_claude, prompts_sdf):
    stub_claude(lambda user_message, **kw: (answer(), "max_tokens"))
    with pytest.raises(rewrite.RewriteAnswerError, match="truncated"):
        rewrite.rewrite_description("l", "d", prompts=rewrite.load_rewrite_prompts(prompts_sdf))


def test_prompt_covers_the_checks_and_says_most_plans_need_no_change(prompts_sdf):
    prompts = rewrite.load_rewrite_prompts(prompts_sdf)
    system, user = rewrite.build_rewrite_prompt(prompts, "LINES", "DESC")
    text = system + user
    assert "Most plans need no change" in system
    for phrase in ("route that exists", "simulated connectome", "dealt scale", "company that provides", "dealt_value_conflict",
                   "every organisation be fictional"):
        assert phrase.lower() in text.lower()
    assert "{" not in user
    source = open(rewrite.__file__, encoding="utf-8").read()
    assert "Most plans need no change" not in source  # model-facing prose lives in the .txt file


def test_the_answer_schema_names_every_check_the_report_counts(prompts_sdf):
    prompts = rewrite.load_rewrite_prompts(prompts_sdf)
    checks = rewrite.check_names(prompts)
    assert {"plausibility_organisation", "plausibility_species_or_custom", "plausibility_simulated_connectome",
            "plausibility_ai_information", "plausibility_figures_and_scale", "conformance_to_dealt_values",
            "organisation_naming"} == set(checks)


# --- the stage ---

def test_a_decline_passes_the_original_untouched(tiny_config, prompts_sdf, stage_dir, stub_claude, plans):
    calls = stub_claude(lambda user_message, **kw: answer())
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert len(calls) == len(plans)
    assert kept == plans and all(a is b for a, b in zip(kept, plans))  # the very same records
    assert all("original_description" not in p and "changes" not in p for p in kept)


def test_a_rewrite_replaces_description_and_keeps_the_original_and_changes(
    tiny_config, prompts_sdf, stage_dir, stub_claude, plans
):
    def dispatcher(user_message, **kw):
        if kw["item_id"] == plans[2]["prompt_id"]:
            return answer(True, "Rewritten description.", ("plausibility_figures_and_scale", "organisation_naming"))
        return answer()

    stub_claude(dispatcher)
    kept = by_id(rewrite.run(tiny_config, prompts_sdf, stage_dir, plans))
    changed = kept[plans[2]["prompt_id"]]
    assert changed["description"] == "Rewritten description."
    assert changed["original_description"] == "Original description 2."
    assert [c["check"] for c in changed["changes"]] == ["plausibility_figures_and_scale", "organisation_naming"]
    assert changed["variables"] == plans[2]["variables"]
    assert kept[plans[0]["prompt_id"]]["description"] == "Original description 0."
    # plans.jsonl is the planner's output and is never touched; the rewrite is stored beside it
    stored = by_id(utils.load_jsonl(stage_dir / "description_rewrites.jsonl"))
    assert stored[plans[2]["prompt_id"]]["original_description"] == "Original description 2."
    assert len(stored) == len(plans)


def test_the_model_sees_the_planners_variable_lines_and_the_description(
    tiny_config, prompts_sdf, stage_dir, stub_claude, plans
):
    calls = stub_claude(lambda user_message, **kw: answer())
    rewrite.run(tiny_config, prompts_sdf, stage_dir, plans[:2])
    render = cp.make_variable_line_renderer(prompts_sdf, 137)
    for call, plan in zip(sorted(calls, key=lambda c: c["item_id"]), plans[:2]):
        assert render(plan) in call["user_message"]
        assert description_in(call["user_message"]) == plan["description"]


def test_a_dealt_value_conflict_drops_the_plan_and_logs_it(tiny_config, prompts_sdf, stage_dir, stub_claude, plans):
    doomed = plans[1]["prompt_id"]
    stub_claude(lambda user_message, **kw: answer(conflict=True, conflict_reason="The dealt scale cannot occur.")
                if kw["item_id"] == doomed else answer())
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert doomed not in {p["prompt_id"] for p in kept} and len(kept) == len(plans) - 1
    rejections = utils.load_jsonl(stage_dir / "description_rewrite_rejections.jsonl")
    assert [(r["prompt_id"], r["conflict_reason"]) for r in rejections] == [(doomed, "The dealt scale cannot occur.")]
    assert rejections[0]["description"] == "Original description 1." and rejections[0]["variables"]


def test_incoherent_plans_are_not_sent_and_pass_through(tiny_config, prompts_sdf, stage_dir, stub_claude, plans):
    plans[3].update(description=None, incoherent=True)
    calls = stub_claude(lambda user_message, **kw: answer())
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert len(calls) == len(plans) - 1 and kept == plans


def test_a_malformed_answer_is_not_checkpointed_and_resume_retries_exactly_that_plan(
    tiny_config, prompts_sdf, stage_dir, stub_claude, plans
):
    broken = plans[4]["prompt_id"]
    stub_claude(lambda user_message, **kw: "not json" if kw["item_id"] == broken else answer())
    rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    checkpoint = utils.Checkpoint(stage_dir / "_description_rewrite_checkpoint.json")
    assert checkpoint.done_count == len(plans) - 1 and not checkpoint.is_done(broken)
    assert broken not in by_id(utils.load_jsonl(stage_dir / "description_rewrites.jsonl"))
    resumed = stub_claude(lambda user_message, **kw: answer(True, "Fixed on retry.", ("organisation_naming",)))
    kept = by_id(rewrite.run(tiny_config, prompts_sdf, stage_dir, plans))
    assert [c["item_id"] for c in resumed] == [broken]
    assert kept[broken]["description"] == "Fixed on retry."


def test_every_call_failing_is_systemic(tiny_config, prompts_sdf, stage_dir, stub_claude, plans):
    stub_claude(lambda user_message, **kw: "not json")
    with pytest.raises(SystemExit, match="every pending call failed"):
        rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)


def test_resume_after_a_completed_stage_makes_zero_calls_and_reapplies_results(
    tiny_config, prompts_sdf, stage_dir, stub_claude, plans
):
    stub_claude(lambda user_message, **kw: answer(True, "Rewritten.", ("organisation_naming",))
                if kw["item_id"] == plans[0]["prompt_id"] else answer())
    first = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    calls = stub_claude([])  # any call would exhaust the empty queue and fail
    assert rewrite.run(tiny_config, prompts_sdf, stage_dir, plans) == first
    assert rewrite.load_rewritten_plans(tiny_config, stage_dir, plans) == first
    assert calls == []


def test_model_falls_back_from_rewrite_to_plan_to_global(tiny_config, prompts_sdf, tmp_path, stub_claude, plans):
    calls = stub_claude(lambda user_message, **kw: answer())
    for name, knobs, expected in (
        ("first", {}, None),
        ("second", {"plan_model": "plan-model-x"}, "plan-model-x"),
        ("third", {"plan_model": "plan-model-x", "description_rewrite_model": "rewrite-model-x"}, "rewrite-model-x"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        tiny_config["sdf"].update(knobs)
        rewrite.run(tiny_config, prompts_sdf, directory, plans[:1])
        assert calls[-1]["model"] == expected


def test_disabled_stage_makes_no_calls_and_writes_nothing(tiny_config, prompts_sdf, stage_dir, stub_claude, plans):
    tiny_config["sdf"]["description_rewrite_enabled"] = False
    calls = stub_claude([])
    assert rewrite.run(tiny_config, prompts_sdf, stage_dir, plans) == plans
    assert rewrite.load_rewritten_plans(tiny_config, stage_dir, plans) == plans
    assert calls == [] and list(stage_dir.iterdir()) == []


def test_snapshot_without_the_template_skips_the_stage(tiny_config, tmp_path, stage_dir, stub_claude, plans):
    calls = stub_claude([])
    old_snapshot = tmp_path / "old_prompts"
    old_snapshot.mkdir()
    assert rewrite.run(tiny_config, old_snapshot, stage_dir, plans) == plans and calls == []


# --- reporting ---

def test_rewrite_shares_per_check_and_per_value_and_the_audit_lines(prompts_sdf, tmp_path, capsys):
    plans = make_plans(prompts_sdf, 6)
    stage_dir = tmp_path / "layer1"
    stage_dir.mkdir()
    utils.save_jsonl(plans, stage_dir / "plans.jsonl")

    def stored(index, **fields):
        return {"prompt_id": plans[index]["prompt_id"], "reasoning": "r", "rewrite_needed": False,
                "original_description": "o", "rewritten_description": "", "changes": [],
                "dealt_value_conflict": False, "conflict_reason": "", **fields}

    two_checks = [{"check": "organisation_naming", "reason": "r"}, {"check": "plausibility_figures_and_scale", "reason": "r"}]
    utils.save_jsonl([
        stored(0, rewrite_needed=True, rewritten_description="n", changes=two_checks),
        stored(1, rewrite_needed=True, rewritten_description="n", changes=two_checks[:1]),
        stored(2, dealt_value_conflict=True, conflict_reason="scale"),
        stored(3), stored(4), stored(5),
    ], stage_dir / "description_rewrites.jsonl")
    outputs = rewrite.load_description_rewrite_outputs(tmp_path)
    shares = rewrite.rewrite_share_rows(plans, outputs)
    assert (shares["checked"], shares["rewritten"], shares["declined"], shares["dropped_for_conflict"]) == (6, 2, 3, 1)
    assert shares["per_check"] == {"organisation_naming": 2, "plausibility_figures_and_scale": 1}
    domain_rows = [r for r in shares["per_value"] if r["variable"] == "domain"]
    assert sum(r["checked"] for r in domain_rows) == 6 and sum(r["rewritten"] for r in domain_rows) == 2
    assert sum(r["dropped"] for r in domain_rows) == 1

    report = {}
    audit_sdf.audit_description_rewrite(tmp_path, report)
    output = capsys.readouterr().out
    assert "2 of 6 checked" in output and "organisation_naming: 2" in output
    assert "dropped for a dealt-value conflict" in output
    assert report["description_rewrite"]["rewritten"] == 2


def test_audit_skips_a_run_without_description_rewrite_files(tmp_path, capsys):
    (tmp_path / "layer1").mkdir()
    report = {}
    audit_sdf.audit_description_rewrite(tmp_path, report)
    assert "no description rewrite files" in capsys.readouterr().out and "description_rewrite" not in report
