"""The critique and revise steps of layer 4.

Reading of each reply shape (accepted, no review tags, unusable, thrown call), the request
read from the draft record, resume (zero calls for finished work; a failed revise after an accepted
critique resumes without a new critique call). ``draft_critique_and_revise_layer.run`` is the layer's
entry point (scripts import it); test_sdf_layers.py pins the rest of that.
"""

import pytest

from sdf_pipeline.layer4_draft_critique_and_revise import critique, revise
from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import RewrittenText, rewrite_record
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind
from sdf_step_and_command_line_test_helpers import outcome_of, reply, run_of
from shared import utils
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE
from sdf_plan_draft_rewrite_and_score_test_builders import config_with_revision_settings, make_draft

GOOD_REVISE = "<improved_document>Better.</improved_document>"
GOOD_CRITIQUE = "<review>Problem: the voice is flat.</review>"


@pytest.fixture
def critique_run(tiny_config, prompts_sdf, stage_dir):
    return run_of(config_with_revision_settings(tiny_config), prompts_sdf, CRITIQUE_AND_REVISE, stage_dir)


@pytest.fixture
def revise_run(tiny_config, prompts_sdf, stage_dir):
    utils.save_jsonl([{"doc_id": "matrix_000000", "review": "Problem."}], stage_dir / "critiques.jsonl")
    return run_of(config_with_revision_settings(tiny_config), prompts_sdf, CRITIQUE_AND_REVISE, stage_dir)


def dispatch(revise=GOOD_REVISE, critique=GOOD_CRITIQUE):
    return lambda user_message, **kw: critique if kw.get("stage") == "layer4_critique" else revise


# ---------------------------------------------------------------- reading the reply: critique

def test_critique_accepted_with_review_tags(critique_run):
    assert outcome_of(critique.STEP, make_draft(), reply("pre <review> Problem. </review> post"), critique_run) == Accepted("Problem.")


@pytest.mark.parametrize("text, stop, reason", [
    ("<review>half", "max_tokens", "critique truncated (stop_reason=max_tokens)"),
    ("<review>   </review>", "end_turn", "critique empty"),
    ("   ", "end_turn", "critique empty"),
])
def test_critique_unusable(critique_run, text, stop, reason):
    assert outcome_of(critique.STEP, make_draft(), reply(text, stop), critique_run) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_critique_thrown_call_is_recorded_as_thrown_and_prints_the_old_line(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    def raise_always(user_message, **kw):
        raise RuntimeError("down")
    stub_claude(raise_always)
    with pytest.raises(SystemExit, match="systemic"):
        draft_critique_and_revise_layer.run_critique_and_revise(config_with_revision_settings(tiny_config), prompts_sdf, stage_dir, [make_draft()])
    assert "  matrix_000000: no rewrite (error: RuntimeError: down) — will retry on resume" in capsys.readouterr().out


# ---------------------------------------------------------------- reading the reply: revise

def test_revise_accepted_pairs_the_text_with_the_critique_that_produced_it(revise_run):
    outcome = outcome_of(revise.STEP, make_draft(), reply("noise " + GOOD_REVISE), revise_run)
    assert outcome == Accepted(RewrittenText("Better.", "Problem."))


@pytest.mark.parametrize("text, stop, reason", [
    ("<improved_document>cut", "max_tokens", "truncated rewrite (stop_reason=max_tokens)"),
    ("no tags at all", "end_turn", "no <improved_document> tags"),
    ("<improved_document>  </improved_document>", "end_turn", "no <improved_document> tags"),
])
def test_revise_unusable(revise_run, text, stop, reason):
    assert outcome_of(revise.STEP, make_draft(), reply(text, stop), revise_run) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_revise_thrown_call_is_recorded_as_thrown(tiny_config, prompts_sdf, stub_claude, tmp_path, capsys):
    def raise_on_revise(user_message, **kw):
        if kw.get("stage") == "layer4_critique":
            return GOOD_CRITIQUE
        raise RuntimeError("down")
    stub_claude(raise_on_revise)
    stage = tmp_path / "layer3"
    with pytest.raises(SystemExit, match="systemic"):
        draft_critique_and_revise_layer.run_critique_and_revise(config_with_revision_settings(tiny_config), prompts_sdf, stage, [make_draft()])
    assert [r["doc_id"] for r in utils.load_jsonl(stage / "critiques.jsonl")] == ["matrix_000000"]
    assert not (stage / "rewrites.jsonl").exists()
    assert "  matrix_000000: no rewrite (error: RuntimeError: down) — will retry on resume" in capsys.readouterr().out


# ---------------------------------------------------------------- requests

def test_each_step_sends_its_stage_tag_model_and_token_budget(tiny_config, prompts_sdf, stage_dir):
    config = config_with_revision_settings(tiny_config)
    config["sdf"]["rewrite_model"] = "rewrite-model"
    config["sdf"]["critique_model"] = "critique-model"
    utils.save_jsonl([{"doc_id": "matrix_000000", "review": "P"}], stage_dir / "critiques.jsonl")
    draft = make_draft()
    two_call = run_of(config, prompts_sdf, CRITIQUE_AND_REVISE, stage_dir)
    critique_request = critique.STEP.build_request([draft], two_call)
    revise_request = revise.STEP.build_request([draft], two_call)
    assert (critique_request.stage, critique_request.model, critique_request.max_tokens, critique_request.cache_system) == (
        "layer4_critique", "critique-model", 8000, True)
    assert (revise_request.stage, revise_request.model, revise_request.max_tokens, revise_request.cache_system) == (
        "layer4_revise", "rewrite-model", 8000, True)
    assert "{review}" not in revise_request.user_message and "P" in revise_request.user_message


def test_the_two_call_templates_snapshot_check_keeps_its_message(tiny_config, prompts_sdf, tmp_path, stage_dir):
    import shutil
    snapshot = tmp_path / "old_prompts"
    shutil.copytree(prompts_sdf, snapshot)
    (snapshot / "layer3b_revise.txt").unlink()
    utils.save_jsonl([{"doc_id": "matrix_000000", "review": "P"}], stage_dir / "critiques.jsonl")
    for step in (critique.STEP, revise.STEP):
        run = run_of(config_with_revision_settings(tiny_config), snapshot, CRITIQUE_AND_REVISE, stage_dir)
        with pytest.raises(SystemExit, match="layer3b_revise.txt is missing"):
            step.build_request([make_draft()], run)


# ---------------------------------------------------------------- resume

def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    config = config_with_revision_settings(tiny_config)
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(dispatch())
    draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, drafts)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    records = draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, drafts)
    assert calls == [] and [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before


def test_a_rewrite_already_on_disk_is_kept_and_costs_no_critique(tiny_config, prompts_sdf, stage_dir, stub_claude):
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    utils.save_jsonl([rewrite_record(drafts[0], RewrittenText("Better.", "r"))], stage_dir / "rewrites.jsonl")
    # the finished draft has no critique on disk and needs none
    calls = stub_claude(dispatch())
    records = draft_critique_and_revise_layer.run_critique_and_revise(config_with_revision_settings(tiny_config), prompts_sdf, stage_dir, drafts)
    assert [c["item_id"] for c in calls] == ["matrix_000001", "matrix_000001"]
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]


def test_resume_sends_nothing_for_a_draft_with_a_rewrite_but_no_critique_and_returns_its_rewrite(
        tiny_config, prompts_sdf, stage_dir, stub_claude):
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    utils.save_jsonl([rewrite_record(drafts[0], RewrittenText("Better.", "r"))], stage_dir / "rewrites.jsonl")
    rewrite_on_disk = utils.load_jsonl(stage_dir / "rewrites.jsonl")[0]
    # the resume critiques and revises only the second draft, and returns both rewrites in input order
    calls = stub_claude(dispatch())
    records = draft_critique_and_revise_layer.run_critique_and_revise(config_with_revision_settings(tiny_config), prompts_sdf, stage_dir, drafts)
    assert {c["item_id"] for c in calls} == {"matrix_000001"}
    assert records[0] == rewrite_on_disk
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
