"""The critique, revise and combined review-and-revise steps of layer 3.

Interpretation of each reply shape (accepted, no review tags, unusable, thrown call), the request
read from the draft record, resume (zero calls for finished work; a failed revise after an accepted
critique resumes without a new critique call) and the events recorded. ``layer3_rewrite.run`` keeps
its signature and behaviour (scripts import it); test_sdf_layers.py pins the rest of that.
"""

import pytest

from sdf_pipeline import layer3_rewrite
from sdf_pipeline.document_event_log import read_document_events
from sdf_pipeline.layer3_rewrite import (
    CombinedReviewAndReviseStep, CritiqueStep, ReviseStep, RewrittenText)
from sdf_pipeline.update_steps import Accepted, Failed, FailureKind
from sdf_step_and_command_line_test_helpers import reply
from shared import utils
from test_sdf_layers import make_draft, single_call_config, split_config

GOOD_REVISE = "<improved_document>Better.</improved_document>"
GOOD_CRITIQUE = "<review>Problem: the voice is flat.</review>"


@pytest.fixture
def critique_step(tiny_config, prompts_sdf):
    return CritiqueStep(split_config(tiny_config), prompts_sdf)


@pytest.fixture
def revise_step(tiny_config, prompts_sdf):
    return ReviseStep(split_config(tiny_config), prompts_sdf, {"matrix_000000": "Problem."})


@pytest.fixture
def combined_step(tiny_config, prompts_sdf):
    return CombinedReviewAndReviseStep(single_call_config(tiny_config), prompts_sdf)


def dispatch(revise=GOOD_REVISE, critique=GOOD_CRITIQUE):
    return lambda user_message, **kw: critique if kw.get("stage") == "layer3_critique" else revise


# ---------------------------------------------------------------- interpret: critique

def test_critique_accepted_with_review_tags(critique_step):
    assert critique_step.interpret(make_draft(), reply("pre <review> Problem. </review> post")) == Accepted("Problem.")


def test_critique_with_no_review_tags_is_accepted_whole(critique_step):
    assert critique_step.interpret(make_draft(), reply("  Problem: the arc is bolted on.  ")) == Accepted(
        "Problem: the arc is bolted on.")


@pytest.mark.parametrize("text, stop, reason", [
    ("<review>half", "max_tokens", "critique truncated (stop_reason=max_tokens)"),
    ("<review>   </review>", "end_turn", "critique empty"),
    ("   ", "end_turn", "critique empty"),
])
def test_critique_unusable(critique_step, text, stop, reason):
    assert critique_step.interpret(make_draft(), reply(text, stop)) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_critique_thrown_call_is_recorded_as_thrown_and_prints_the_old_line(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    def raise_always(user_message, **kw):
        raise RuntimeError("down")
    stub_claude(raise_always)
    with pytest.raises(SystemExit, match="systemic"):
        layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])
    assert "  matrix_000000: no rewrite (error: RuntimeError: down) — will retry on resume" in capsys.readouterr().out
    (event,) = read_document_events(stage_dir.parent / "document_events.jsonl")
    assert (event["step"], event["outcome"], event["failure_kind"]) == ("critique", "failed", "thrown_call")


# ---------------------------------------------------------------- interpret: revise and combined

def test_revise_accepted_pairs_the_text_with_the_critique_that_produced_it(revise_step):
    outcome = revise_step.interpret(make_draft(), reply("noise " + GOOD_REVISE))
    assert outcome == Accepted(RewrittenText("Better.", "Problem."))


@pytest.mark.parametrize("text, stop, reason", [
    ("<improved_document>cut", "max_tokens", "truncated rewrite (stop_reason=max_tokens)"),
    ("no tags at all", "end_turn", "no <improved_document> tags"),
    ("<improved_document>  </improved_document>", "end_turn", "no <improved_document> tags"),
])
def test_revise_and_combined_unusable(revise_step, combined_step, text, stop, reason):
    for step in (revise_step, combined_step):
        assert step.interpret(make_draft(), reply(text, stop)) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_combined_takes_the_text_before_the_tags_as_the_review(combined_step):
    outcome = combined_step.interpret(make_draft(), reply("Problems: none.\n" + GOOD_REVISE))
    assert outcome == Accepted(RewrittenText("Better.", "Problems: none."))


def test_revise_and_combined_thrown_call_is_recorded_as_thrown(tiny_config, prompts_sdf, stub_claude, tmp_path):
    def raise_on_revise(user_message, **kw):
        if kw.get("stage") == "layer3_critique":
            return GOOD_CRITIQUE
        raise RuntimeError("down")
    stub_claude(raise_on_revise)
    stage = tmp_path / "layer3"
    with pytest.raises(SystemExit, match="systemic"):
        layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage, [make_draft()])
    steps_and_kinds = [(e["step"], e["outcome"], e.get("failure_kind"))
                       for e in read_document_events(tmp_path / "document_events.jsonl")]
    assert steps_and_kinds == [("critique", "accepted", None), ("revise", "failed", "thrown_call")]

    stub_claude(raise_on_revise)
    single_stage = tmp_path / "single" / "layer3"
    with pytest.raises(SystemExit, match="systemic"):
        layer3_rewrite.run(single_call_config(tiny_config), prompts_sdf, single_stage, [make_draft()])
    (event,) = read_document_events(tmp_path / "single" / "document_events.jsonl")
    assert (event["step"], event["failure_kind"]) == ("combined_review_and_revise", "thrown_call")


# ---------------------------------------------------------------- requests

def test_requests_read_description_variables_and_considerations_from_the_draft_not_a_later_plan(
        tiny_config, prompts_sdf, critique_step, revise_step, combined_step):
    draft = {**make_draft(), "description": "The text's own spec.", "considerations_spec": "Weigh the mice."}
    for step in (critique_step, revise_step, combined_step):
        request = step.build_request(draft)
        assert "The text's own spec." in request.user_message and "Weigh the mice." in request.user_message
    without = {**make_draft(), "description": "Spec without notes."}
    assert "Weigh the mice." not in revise_step.build_request(without).user_message


def test_each_step_sends_its_stage_tag_model_and_token_budget(tiny_config, prompts_sdf):
    config = split_config(tiny_config)
    config["sdf"]["rewrite_model"] = "rewrite-model"
    config["sdf"]["critique_model"] = "critique-model"
    draft = make_draft()
    critique = CritiqueStep(config, prompts_sdf).build_request(draft)
    revise = ReviseStep(config, prompts_sdf, {"matrix_000000": "P"}).build_request(draft)
    combined = CombinedReviewAndReviseStep(single_call_config(tiny_config), prompts_sdf).build_request(draft)
    assert (critique.stage, critique.model, critique.max_tokens, critique.cache_system) == (
        "layer3_critique", "critique-model", 8000, True)
    assert (revise.stage, revise.model, revise.max_tokens, revise.cache_system) == (
        "layer3_revise", "rewrite-model", 8000, True)
    assert (combined.stage, combined.max_tokens, combined.cache_system) == ("layer3_rewrite", 8000, True)
    assert "{review}" not in revise.user_message and "P" in revise.user_message


def test_voice_and_length_directives_ride_the_revise_and_combined_requests_only(tiny_config, prompts_sdf):
    draft = {**make_draft(), "revision": {"cards": {"voice_pace": "clipped", "length": "about 900 words"}, "personas": []}}
    two_call = split_config(tiny_config, style_mode="axes", use_length=True)
    one_call = single_call_config(tiny_config, style_mode="axes", use_length=True)
    requests = {
        "critique": CritiqueStep(two_call, prompts_sdf).build_request(draft),
        "revise": ReviseStep(two_call, prompts_sdf, {"matrix_000000": "P"}).build_request(draft),
        "combined": CombinedReviewAndReviseStep(one_call, prompts_sdf).build_request(draft),
    }
    for name in ("revise", "combined"):
        assert "more clipped than a typical" in requests[name].user_message
        assert "about 900 words" in requests[name].user_message
    assert "more clipped" not in requests["critique"].user_message
    assert "about 900 words" not in requests["critique"].user_message
    # with the arm off, the same deal changes nothing
    off = ReviseStep(split_config(tiny_config), prompts_sdf, {"matrix_000000": "P"}).build_request(draft)
    assert "clipped" not in off.user_message and "900 words" not in off.user_message


def test_the_two_call_templates_snapshot_check_keeps_its_message(tiny_config, prompts_sdf, tmp_path):
    import shutil
    snapshot = tmp_path / "old_prompts"
    shutil.copytree(prompts_sdf, snapshot)
    (snapshot / "layer3b_revise.txt").unlink()
    for build in (lambda: CritiqueStep(split_config(tiny_config), snapshot),
                  lambda: ReviseStep(split_config(tiny_config), snapshot, {})):
        with pytest.raises(SystemExit, match="layer3b_revise.txt is missing .* sdf.revision.split: false"):
            build()


# ---------------------------------------------------------------- resume

def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    config = split_config(tiny_config)
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(dispatch())
    layer3_rewrite.run(config, prompts_sdf, stage_dir, drafts)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    events_before = (stage_dir.parent / "document_events.jsonl").read_bytes()
    calls = stub_claude([])
    records = layer3_rewrite.run(config, prompts_sdf, stage_dir, drafts)
    assert calls == [] and [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before
    assert (stage_dir.parent / "document_events.jsonl").read_bytes() == events_before


def test_a_failed_revise_after_an_accepted_critique_resumes_without_a_new_critique_call(
        tiny_config, prompts_sdf, stage_dir, stub_claude):
    config = split_config(tiny_config)
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(dispatch(revise="no tags"))
    assert layer3_rewrite.run(config, prompts_sdf, stage_dir, drafts) == []
    assert [r["doc_id"] for r in utils.load_jsonl(stage_dir / "critiques.jsonl")] == ["matrix_000000", "matrix_000001"]
    assert not (stage_dir / "rewrites.jsonl").exists()

    calls = stub_claude(dispatch())
    records = layer3_rewrite.run(config, prompts_sdf, stage_dir, drafts)
    assert sorted((c["stage"], c["item_id"]) for c in calls) == [
        ("layer3_revise", "matrix_000000"), ("layer3_revise", "matrix_000001")]
    assert [r["review"] for r in records] == ["Problem: the voice is flat."] * 2
    assert len(utils.load_jsonl(stage_dir / "critiques.jsonl")) == 2
    steps = [(e["document_id"], e["step"], e["outcome"])
             for e in read_document_events(stage_dir.parent / "document_events.jsonl")]
    assert steps.count(("matrix_000000", "critique", "accepted")) == 1
    assert ("matrix_000000", "revise", "failed") in steps and ("matrix_000000", "revise", "accepted") in steps


def test_a_rewrite_already_on_disk_is_kept_and_costs_no_critique(tiny_config, prompts_sdf, stage_dir, stub_claude):
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(lambda user_message, **kw: "r\n" + GOOD_REVISE)
    layer3_rewrite.run(single_call_config(tiny_config), prompts_sdf, stage_dir, drafts[:1])
    # a single-call run resumed as split: the finished draft has no critique on disk and needs none
    calls = stub_claude(dispatch())
    records = layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, drafts)
    assert [c["item_id"] for c in calls] == ["matrix_000001", "matrix_000001"]
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]


def test_an_accepted_critique_does_not_finish_the_draft(tiny_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(dispatch(revise="no tags"))
    layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])
    assert not utils.Checkpoint(stage_dir / "_checkpoint.json").is_done("matrix_000000")
    assert utils.load_jsonl(stage_dir / "critiques.jsonl") == [
        {"doc_id": "matrix_000000", "review": "Problem: the voice is flat."}]
