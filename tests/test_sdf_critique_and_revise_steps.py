"""The critique, revise and one-call critique-and-revise steps.

Reading of each reply shape (accepted, no review tags, unusable, thrown call), the request
read from the draft record, resume (zero calls for finished work; a failed revise after an accepted
critique resumes without a new critique call). ``rewrite.run`` is the layer's
entry point (scripts import it); test_sdf_layers.py pins the rest of that.
"""

import pytest

from sdf_pipeline import layer3_rewrite as rewrite
from sdf_pipeline.layer3_rewrite import RewrittenText
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind
from sdf_step_and_command_line_test_helpers import outcome_of, reply, run_of
from shared import utils
from sdf_pipeline.shared_step_machinery.step_record import LAYER3
from sdf_plan_draft_rewrite_and_score_test_builders import make_draft, single_call_config, split_config

GOOD_REVISE = "<improved_document>Better.</improved_document>"
GOOD_CRITIQUE = "<review>Problem: the voice is flat.</review>"


@pytest.fixture
def critique_run(tiny_config, prompts_sdf, stage_dir):
    return run_of(split_config(tiny_config), prompts_sdf, LAYER3, stage_dir)


@pytest.fixture
def revise_run(tiny_config, prompts_sdf, stage_dir):
    utils.save_jsonl([{"doc_id": "matrix_000000", "review": "Problem."}], stage_dir / "critiques.jsonl")
    return run_of(split_config(tiny_config), prompts_sdf, LAYER3, stage_dir)


@pytest.fixture
def combined_run(tiny_config, prompts_sdf, stage_dir):
    return run_of(single_call_config(tiny_config), prompts_sdf, LAYER3, stage_dir)


def dispatch(revise=GOOD_REVISE, critique=GOOD_CRITIQUE):
    return lambda user_message, **kw: critique if kw.get("stage") == "layer3_critique" else revise


# ---------------------------------------------------------------- reading the reply: critique

def test_critique_accepted_with_review_tags(critique_run):
    assert outcome_of(rewrite.CRITIQUE_STEP, make_draft(), reply("pre <review> Problem. </review> post"), critique_run) == Accepted("Problem.")


@pytest.mark.parametrize("text, stop, reason", [
    ("<review>half", "max_tokens", "critique truncated (stop_reason=max_tokens)"),
    ("<review>   </review>", "end_turn", "critique empty"),
    ("   ", "end_turn", "critique empty"),
])
def test_critique_unusable(critique_run, text, stop, reason):
    assert outcome_of(rewrite.CRITIQUE_STEP, make_draft(), reply(text, stop), critique_run) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_critique_thrown_call_is_recorded_as_thrown_and_prints_the_old_line(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    def raise_always(user_message, **kw):
        raise RuntimeError("down")
    stub_claude(raise_always)
    with pytest.raises(SystemExit, match="systemic"):
        rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])
    assert "  matrix_000000: no rewrite (error: RuntimeError: down) — will retry on resume" in capsys.readouterr().out


# ---------------------------------------------------------------- reading the reply: revise and combined

def test_revise_accepted_pairs_the_text_with_the_critique_that_produced_it(revise_run):
    outcome = outcome_of(rewrite.REVISE_STEP, make_draft(), reply("noise " + GOOD_REVISE), revise_run)
    assert outcome == Accepted(RewrittenText("Better.", "Problem."))


@pytest.mark.parametrize("text, stop, reason", [
    ("<improved_document>cut", "max_tokens", "truncated rewrite (stop_reason=max_tokens)"),
    ("no tags at all", "end_turn", "no <improved_document> tags"),
    ("<improved_document>  </improved_document>", "end_turn", "no <improved_document> tags"),
])
def test_revise_and_combined_unusable(revise_run, combined_run, text, stop, reason):
    for step, run in ((rewrite.REVISE_STEP, revise_run), (rewrite.ONE_CALL_STEP, combined_run)):
        assert outcome_of(step, make_draft(), reply(text, stop), run) == Failed(FailureKind.UNUSABLE_ANSWER, reason)


def test_combined_takes_the_text_before_the_tags_as_the_review(combined_run):
    outcome = outcome_of(rewrite.ONE_CALL_STEP, make_draft(), reply("Problems: none.\n" + GOOD_REVISE), combined_run)
    assert outcome == Accepted(RewrittenText("Better.", "Problems: none."))


def test_revise_and_combined_thrown_call_is_recorded_as_thrown(tiny_config, prompts_sdf, stub_claude, tmp_path, capsys):
    def raise_on_revise(user_message, **kw):
        if kw.get("stage") == "layer3_critique":
            return GOOD_CRITIQUE
        raise RuntimeError("down")
    stub_claude(raise_on_revise)
    stage = tmp_path / "layer3"
    with pytest.raises(SystemExit, match="systemic"):
        rewrite.run(split_config(tiny_config), prompts_sdf, stage, [make_draft()])
    assert [r["doc_id"] for r in utils.load_jsonl(stage / "critiques.jsonl")] == ["matrix_000000"]
    assert not (stage / "rewrites.jsonl").exists()
    assert "  matrix_000000: no rewrite (error: RuntimeError: down) — will retry on resume" in capsys.readouterr().out

    stub_claude(raise_on_revise)
    single_stage = tmp_path / "single" / "layer3"
    with pytest.raises(SystemExit, match="systemic"):
        rewrite.run(single_call_config(tiny_config), prompts_sdf, single_stage, [make_draft()])


# ---------------------------------------------------------------- requests

def test_each_step_sends_its_stage_tag_model_and_token_budget(tiny_config, prompts_sdf, stage_dir):
    config = split_config(tiny_config)
    config["sdf"]["rewrite_model"] = "rewrite-model"
    config["sdf"]["critique_model"] = "critique-model"
    utils.save_jsonl([{"doc_id": "matrix_000000", "review": "P"}], stage_dir / "critiques.jsonl")
    draft = make_draft()
    two_call = run_of(config, prompts_sdf, LAYER3, stage_dir)
    critique_request = rewrite.CRITIQUE_STEP.build_request([draft], two_call)
    revise_request = rewrite.REVISE_STEP.build_request([draft], two_call)
    combined = rewrite.ONE_CALL_STEP.build_request(
        [draft], run_of(single_call_config(tiny_config), prompts_sdf, LAYER3, stage_dir))
    assert (critique_request.stage, critique_request.model, critique_request.max_tokens, critique_request.cache_system) == (
        "layer3_critique", "critique-model", 8000, True)
    assert (revise_request.stage, revise_request.model, revise_request.max_tokens, revise_request.cache_system) == (
        "layer3_revise", "rewrite-model", 8000, True)
    assert (combined.stage, combined.max_tokens, combined.cache_system) == ("layer3_rewrite", 8000, True)
    assert "{review}" not in revise_request.user_message and "P" in revise_request.user_message


def test_the_two_call_templates_snapshot_check_keeps_its_message(tiny_config, prompts_sdf, tmp_path, stage_dir):
    import shutil
    snapshot = tmp_path / "old_prompts"
    shutil.copytree(prompts_sdf, snapshot)
    (snapshot / "layer3b_revise.txt").unlink()
    with pytest.raises(SystemExit, match="layer3_rewrite: layer3b_revise.txt is missing .* sdf.revision.split: false"):
        rewrite.run(split_config(tiny_config), snapshot, stage_dir, [make_draft()])


# ---------------------------------------------------------------- resume

def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    config = split_config(tiny_config)
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(dispatch())
    rewrite.run(config, prompts_sdf, stage_dir, drafts)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    records = rewrite.run(config, prompts_sdf, stage_dir, drafts)
    assert calls == [] and [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before


def test_a_rewrite_already_on_disk_is_kept_and_costs_no_critique(tiny_config, prompts_sdf, stage_dir, stub_claude):
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(lambda user_message, **kw: "r\n" + GOOD_REVISE)
    rewrite.run(single_call_config(tiny_config), prompts_sdf, stage_dir, drafts[:1])
    # a single-call run resumed as split: the finished draft has no critique on disk and needs none
    calls = stub_claude(dispatch())
    records = rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, drafts)
    assert [c["item_id"] for c in calls] == ["matrix_000001", "matrix_000001"]
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]


def test_resume_sends_nothing_for_a_draft_with_a_rewrite_but_no_critique_and_returns_its_rewrite(
        tiny_config, prompts_sdf, stage_dir, stub_claude):
    drafts = [make_draft("matrix_000000"), make_draft("matrix_000001")]
    stub_claude(lambda user_message, **kw: "r\n" + GOOD_REVISE)
    rewrite.run(single_call_config(tiny_config), prompts_sdf, stage_dir, drafts[:1])
    assert not (stage_dir / "critiques.jsonl").exists()
    rewrite_on_disk = utils.load_jsonl(stage_dir / "rewrites.jsonl")[0]
    # the split resume critiques and revises only the second draft, and returns both rewrites in input order
    calls = stub_claude(dispatch())
    records = rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, drafts)
    assert {c["item_id"] for c in calls} == {"matrix_000001"}
    assert records[0] == rewrite_on_disk
    assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
