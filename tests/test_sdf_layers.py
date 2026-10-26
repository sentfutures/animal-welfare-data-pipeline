"""Behavior tests for the SDF matrix pipeline stages, driven through each
stage's run() with a stubbed shared.api.call_claude. No test here may reach
the API (see conftest). Real templates, variables, and constitution files stay
in play — that's what catches template/pipeline drift.

Money paths per stage (protects paid work at $-per-run scale):
- parse happy path
- malformed-response handling (retry-on-resume or recorded fallback)
- checkpoint/resume with zero API calls for completed work
"""

import json
import shutil

import pytest

from sdf_pipeline import (
    compose_prompts as cp,
    layer1_plan,
    layer2_draft,
    layer3_rewrite,
    layer4_score,
)
from shared import utils


@pytest.fixture
def stage_dir(tmp_path):
    d = tmp_path / "stage"
    d.mkdir()
    return d


@pytest.fixture
def final_dir(tmp_path):
    d = tmp_path / "final"
    d.mkdir()
    return d


def plan_reply(description="A fully specified document plan.", notes="notes"):
    return (
        f"<document_planning>{notes}</document_planning>\n"
        f"<document_description>\n{description}\n</document_description>"
    )


def make_plan(pid="matrix_000000", description="Spec text.", incoherent=False):
    return {
        "prompt_id": pid,
        "variables": {
            "document_type": "a news article",
            "culture": "France, written in French, with French idioms and references",
            "tone": "neutral or journalistic",
            "reasoning_featured": "individualism: the notion that welfare is a property of individuals, not systems",
        },
        "plan": "…",
        "description": None if incoherent else description,
        "incoherent": incoherent,
    }


def make_draft(pid="matrix_000000", content="Un document."):
    p = make_plan(pid)
    return {"doc_id": pid, "variables": p["variables"], "description": "Spec text.", "content": content}


def make_rewrite(pid="matrix_000000", content="Un document réécrit."):
    d = make_draft(pid, content)
    return {**d, "review": "review notes"}


SCORE_OK = json.dumps({"alignment": 9, "realism": 8, "spec_conformance": 9, "notes": "fine"})


class TestLayer12Plan:
    def test_composes_once_then_plans(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(lambda user_message, **kw: plan_reply(f"Plan for {kw['item_id']}."))
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)

        prompts = utils.load_jsonl(stage_dir / "prompts.jsonl")
        assert len(prompts) == tiny_config["sdf"]["n_prompts"] == len(records) == len(calls)
        # the composed system section (preamble) travels on each call
        assert all(c["system_prompt"] for c in calls)
        assert all(c["stage"] == "layer1_plan" for c in calls)
        for r in records:
            assert r["description"] == f"Plan for {r['prompt_id']}."
            assert not r["incoherent"]

    def test_composition_is_reused_and_resume_makes_no_calls(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        stub_claude(lambda user_message, **kw: plan_reply())
        layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        prompts_before = utils.load_jsonl(stage_dir / "prompts.jsonl")

        calls = stub_claude([])
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert calls == []
        assert len(records) == tiny_config["sdf"]["n_prompts"]
        assert utils.load_jsonl(stage_dir / "prompts.jsonl") == prompts_before

    def test_incoherent_is_checkpointed_as_deliberate_rejection(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        stub_claude(lambda user_message, **kw:
                    "<document_description>INCOHERENT: no such document.</document_description>")
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert all(r["incoherent"] and r["description"] is None for r in records)

        calls = stub_claude([])
        layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert calls == []  # rejected combos are done, not retried

    def test_malformed_plan_is_retried_on_resume(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: "chatter with no description tags")
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert records == []

        calls = stub_claude(lambda user_message, **kw: plan_reply())
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert len(calls) == tiny_config["sdf"]["n_prompts"]  # every failure retried
        assert len(records) == tiny_config["sdf"]["n_prompts"]

    def test_truncated_plan_is_retried_on_resume(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: (plan_reply(), "max_tokens"))
        assert layer1_plan.run(tiny_config, prompts_sdf, stage_dir) == []

        calls = stub_claude(lambda user_message, **kw: plan_reply())
        assert len(layer1_plan.run(tiny_config, prompts_sdf, stage_dir)) == 2
        assert len(calls) == 2

    def test_plan_model_override_used(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = {**tiny_config, "sdf": {**tiny_config["sdf"], "plan_model": "claude-haiku-4-5"}}
        calls = stub_claude(lambda user_message, **kw: plan_reply())
        layer1_plan.run(config, prompts_sdf, stage_dir)
        assert all(c["model"] == "claude-haiku-4-5" for c in calls)

    def test_one_poison_call_does_not_kill_the_layer(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        # One prompt's API call fails outright (the claude_code backend's
        # usage-policy false positive seen live): its siblings must still be
        # planned and checkpointed, and the failed prompt retried on resume.
        def dispatch(user_message, **kw):
            if kw["item_id"] == "matrix_000001":
                raise RuntimeError("API Error: usage policy")
            return plan_reply(f"Plan for {kw['item_id']}.")

        stub_claude(dispatch)
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert [r["prompt_id"] for r in records] == ["matrix_000000"]

        calls = stub_claude(lambda user_message, **kw: plan_reply("Recovered."))
        records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
        assert len(calls) == 1  # only the failed prompt is retried
        assert sorted(r["prompt_id"] for r in records) == ["matrix_000000", "matrix_000001"]

    def test_all_plan_calls_failing_is_a_systemic_error(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        def dispatch(user_message, **kw):
            raise RuntimeError("auth failure")

        stub_claude(dispatch)
        with pytest.raises(SystemExit, match="systemic"):
            layer1_plan.run(tiny_config, prompts_sdf, stage_dir)


class TestLayer3Draft:
    def test_drafts_from_descriptions(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(lambda user_message, **kw:
                            f"thinking\n<document>Doc for {kw['item_id']}.</document>")
        plans = [make_plan("matrix_000000"), make_plan("matrix_000001")]
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)

        assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000001"]
        assert records[0]["content"] == "Doc for matrix_000000."
        # constitution rides the system prompt; spec rides the user message
        assert all("Spec text." in c["user_message"] for c in calls)
        assert all(c["system_prompt"] for c in calls)
        assert all(c["max_tokens"] == 6000 for c in calls)
        assert all(c["cache_system"] for c in calls)  # static constitution system prompt is cached
        assert utils.load_jsonl(stage_dir / "drafts.jsonl") == records

    def test_incoherent_plans_are_skipped(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
        plans = [make_plan("matrix_000000", incoherent=True), make_plan("matrix_000001")]
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
        assert [r["doc_id"] for r in records] == ["matrix_000001"]
        assert len(calls) == 1

    def test_missing_tags_retried_on_resume(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: "no tags at all")
        plans = [make_plan()]
        assert layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans) == []

        calls = stub_claude(lambda user_message, **kw: "<document>Recovered.</document>")
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
        assert len(calls) == 1
        assert records[0]["content"] == "Recovered."

    def test_resume_makes_no_calls(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: "<document>D.</document>")
        plans = [make_plan()]
        layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)

        calls = stub_claude([])
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
        assert calls == []
        assert len(records) == 1

    def test_one_poison_call_does_not_kill_the_layer(self, tiny_config, prompts_sdf,
                                                     stage_dir, stub_claude):
        # One doc's API call fails outright (the claude_code backend's
        # usage-policy false positive seen live): its siblings must still be
        # drafted and checkpointed, and the failed doc retried on resume.
        def dispatch(user_message, **kw):
            if kw["item_id"] == "matrix_000001":
                raise RuntimeError("API Error: usage policy")
            return f"<document>Doc for {kw['item_id']}.</document>"

        stub_claude(dispatch)
        plans = [make_plan("matrix_000000"), make_plan("matrix_000001"),
                 make_plan("matrix_000002")]
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
        assert [r["doc_id"] for r in records] == ["matrix_000000", "matrix_000002"]

        calls = stub_claude(lambda user_message, **kw:
                            "<document>Recovered.</document>")
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, plans)
        assert len(calls) == 1  # only the failed doc is retried
        assert sorted(r["doc_id"] for r in records) == [
            "matrix_000000", "matrix_000001", "matrix_000002"]

    def test_all_calls_failing_is_a_systemic_error(self, tiny_config, prompts_sdf,
                                                   stage_dir, stub_claude):
        def dispatch(user_message, **kw):
            raise RuntimeError("auth failure")

        stub_claude(dispatch)
        with pytest.raises(SystemExit, match="systemic"):
            layer2_draft.run(tiny_config, prompts_sdf, stage_dir,
                             [make_plan("matrix_000000"), make_plan("matrix_000001")])


class TestLayer4Rewrite:
    def test_rewrites_and_keeps_review(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        """The single-call shape: one reply carries the review and the rewrite."""
        calls = stub_claude(lambda user_message, **kw:
                            "Problems: none.\n<improved_document>Better doc.</improved_document>")
        drafts = [make_draft()]
        records = layer3_rewrite.run(
            single_call_config(tiny_config), prompts_sdf, stage_dir, drafts)

        assert records[0]["content"] == "Better doc."
        assert records[0]["review"] == "Problems: none."
        # the draft and its generating spec both reach the rewriter
        assert "Un document." in calls[0]["user_message"]
        assert "Spec text." in calls[0]["user_message"]
        assert calls[0]["max_tokens"] == 8000
        assert calls[0]["cache_system"]  # static constitution system prompt is cached

    def test_rewrite_model_override_used(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = {**tiny_config, "sdf": {**tiny_config["sdf"], "rewrite_model": "claude-fable-5"}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        assert calls[0]["model"] == "claude-fable-5"

    def test_missing_tags_retried_on_resume(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: "review only, forgot the tags")
        drafts = [make_draft()]
        assert layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, drafts) == []

        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>Recovered.</improved_document>")
        records = layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, drafts)
        assert len(calls) == 1
        assert records[0]["content"] == "Recovered."

    def test_resume_makes_no_calls(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: "r\n<improved_document>B.</improved_document>")
        drafts = [make_draft()]
        layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, drafts)

        calls = stub_claude([])
        records = layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, drafts)
        assert calls == []
        assert len(records) == 1


# Final lines that carry no terminal punctuation but end a complete document.
# These are the shapes evals/audit_sdf.py's ends_mid_sentence once counted as
# truncations (shared/textstats.py); a stage making the same misjudgement would
# discard a paid document rather than mis-report a number.
SIGNOFF_ENDINGS = pytest.mark.parametrize("ending", [
    "Bien à vous,\n\n— Michelle",
    "Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board",
    "See also: Machine ethics; Digital minds; Precautionary reasoning under uncertainty",
    "关联条目 · 动物福利 · 感受性 · 生态系统服务 · 人工智能伦理准则",
], ids=["signoff", "letterhead", "see-also", "cjk-label-row"])


class TestCompleteDocumentsAreNotRejectedAsTruncated:
    """The false-positive direction, at the stages where it would cost a document.

    These stages decide truncation from `stop_reason` alone and never inspect the
    text, so a document's final line cannot affect whether it survives. That is
    the property under test: a draft or rewrite closing on a sign-off, letterhead
    or See-also row must be checkpointed verbatim.

    The truncated counterparts sit in the same class on purpose. A test that only
    proves nothing gets rejected would pass equally well with the guard deleted,
    so each stage is pinned in both directions.
    """

    @SIGNOFF_ENDINGS
    def test_layer2_keeps_a_draft_ending_on_a_signoff(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude, ending
    ):
        body = f"Le comité a examiné les normes de bien-être animal.\n\n{ending}"
        stub_claude(lambda user_message, **kw: f"notes\n<document>{body}</document>")
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, [make_plan()])

        assert len(records) == 1
        assert records[0]["content"] == body  # sign-off intact, nothing trimmed
        assert utils.load_jsonl(stage_dir / "drafts.jsonl") == records

    def test_layer2_rejects_a_truncated_draft_and_retries_on_resume(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        stub_claude(lambda user_message, **kw:
                    ("notes\n<document>The inspector noted that the stocking den",
                     "max_tokens"))
        assert layer2_draft.run(tiny_config, prompts_sdf, stage_dir, [make_plan()]) == []
        assert utils.load_jsonl(stage_dir / "drafts.jsonl") == []  # not checkpointed

        calls = stub_claude(lambda user_message, **kw: "n\n<document>Recovered.</document>")
        records = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, [make_plan()])
        assert len(calls) == 1
        assert records[0]["content"] == "Recovered."

    @SIGNOFF_ENDINGS
    def test_layer3_keeps_a_rewrite_ending_on_a_signoff(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude, ending
    ):
        body = f"Le comité a revu les normes, avec des corrections.\n\n{ending}"
        stub_claude(lambda user_message, **kw:
                    f"Problems: none.\n<improved_document>{body}</improved_document>")
        records = layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, [make_draft()])

        assert len(records) == 1
        assert records[0]["content"] == body
        assert utils.load_jsonl(stage_dir / "rewrites.jsonl") == records

    def test_layer3_rejects_a_truncated_rewrite_and_retries_on_resume(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        config = single_call_config(tiny_config)
        stub_claude(lambda user_message, **kw:
                    ("Problems: none.\n<improved_document>Cut mid-sen", "max_tokens"))
        assert layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()]) == []
        assert utils.load_jsonl(stage_dir / "rewrites.jsonl") == []

        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>Recovered.</improved_document>")
        records = layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        assert len(calls) == 1
        assert records[0]["content"] == "Recovered."


class TestLayer5Score:
    def test_scores_gate_and_corpus_fields(self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
        def dispatch(user_message, **kw):
            if kw["item_id"] == "matrix_000000":
                return SCORE_OK
            return json.dumps({"alignment": 4, "realism": 9, "spec_conformance": 9, "notes": "misaligned"})

        calls = stub_claude(dispatch)
        rewrites = [make_rewrite("matrix_000000"), make_rewrite("matrix_000001", "Autre document.")]
        corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, rewrites)

        assert [r["doc_id"] for r in corpus] == ["matrix_000000"]  # A4 gated out
        rec = corpus[0]
        # eval-facing fields derived from the matrix variables
        assert rec["type_name"] == "a news article"
        assert rec["language"] == "French"
        assert rec["register"] == "neutral or journalistic"
        assert rec["scores"]["spec_conformance"] == 9
        assert all(c["cache_system"] for c in calls)  # static constitution system prompt is cached
        assert utils.load_jsonl(final_dir / "sdf_corpus.jsonl") == corpus

    def test_low_spec_conformance_does_not_gate(self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
        stub_claude(lambda user_message, **kw:
                    json.dumps({"alignment": 9, "realism": 9, "spec_conformance": 2, "notes": "drifted"}))
        corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
        assert len(corpus) == 1  # advisory only

    def test_parse_error_records_fallback_and_is_not_retried(
        self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude
    ):
        stub_claude(lambda user_message, **kw: "utter nonsense, no json")
        corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
        assert corpus == []  # 5/5 falls below threshold 7
        scores = utils.load_jsonl(stage_dir / "scores.jsonl")
        assert scores[0]["scores"]["notes"] == "Parse error."

        calls = stub_claude([])
        layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
        assert calls == []  # checkpointed: re-scoring would re-bill for nothing

    def test_wrong_shaped_judge_reply_falls_back_like_a_parse_error(
        self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude
    ):
        # A list-shaped reply is valid JSON but not a score object; extract_json_object
        # rejects it into the parse-error default rather than crashing on .get()
        # (regression guard for the extract_json -> extract_json_object fix merged from main).
        stub_claude(lambda user_message, **kw: "[8, 7, 9]")
        corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, [make_rewrite()])
        assert corpus == []  # 5/5 default falls below threshold 7
        scores = utils.load_jsonl(stage_dir / "scores.jsonl")
        assert scores[0]["scores"]["notes"] == "Parse error."

    def test_near_dup_cull(self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
        config = {**tiny_config, "sdf": {**tiny_config["sdf"], "near_dup_threshold": 0.90}}
        stub_claude(lambda user_message, **kw: SCORE_OK)
        same = ("Le même document mot pour mot, assez long pour produire des shingles "
                "stables et un cosinus de similarité fiable entre les deux copies.")
        rewrites = [make_rewrite("matrix_000000", same), make_rewrite("matrix_000001", same)]
        corpus = layer4_score.run(config, prompts_sdf, stage_dir, final_dir, rewrites)
        assert [r["doc_id"] for r in corpus] == ["matrix_000000"]  # keep-first
        dropped = utils.load_jsonl(stage_dir / "near_dup_dropped.jsonl")
        assert dropped[0]["doc_id"] == "matrix_000001"
        assert dropped[0]["kept_doc_id"] == "matrix_000000"

    def test_resume_makes_no_calls(self, tiny_config, prompts_sdf, stage_dir, final_dir, stub_claude):
        stub_claude(lambda user_message, **kw: SCORE_OK)
        rewrites = [make_rewrite()]
        layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, rewrites)

        calls = stub_claude([])
        corpus = layer4_score.run(tiny_config, prompts_sdf, stage_dir, final_dir, rewrites)
        assert calls == []
        assert len(corpus) == 1


def test_derive_language():
    assert cp.derive_language("Japan, written in Japanese, with Japanese idioms") == "Japanese"
    assert cp.derive_language("the United States, written in English, with American idioms") == "English"
    assert cp.derive_language("no language clause") == "English"


def split_config(tiny_config, **revision):
    """tiny_config with the layer-3 split (and any other revision knob) on."""
    return {**tiny_config,
            "sdf": {**tiny_config["sdf"], "revision": {"split": True, **revision}}}


def single_call_config(tiny_config, **revision):
    """tiny_config pinned to the single-call layer3.txt, which still ships."""
    return {**tiny_config,
            "sdf": {**tiny_config["sdf"], "revision": {"split": False, **revision}}}


def revise_call(calls: list[dict]) -> dict:
    """The call the directives ride on, under either layer-3 shape."""
    return next(c for c in calls if c["stage"] in ("layer3_revise", "layer3_rewrite"))


class TestLayer3SplitRewrite:
    """Experiment 3: critique and revise as two calls instead of one.

    The money property is that the two calls are checkpointed SEPARATELY. A
    revise call that fails after its critique succeeded must leave the paid
    critique on disk, or --resume re-bills work that already landed.
    """

    @staticmethod
    def _dispatch(revise="<improved_document>Better.</improved_document>",
                  critique="<review>Problem: the voice is flat.</review>"):
        def fake(user_message, **kw):
            return critique if kw.get("stage") == "layer3_critique" else revise
        return fake

    def test_two_calls_with_distinct_stage_tags(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(self._dispatch())
        records = layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])

        assert [c["stage"] for c in calls] == ["layer3_critique", "layer3_revise"]
        assert records[0]["content"] == "Better."
        # the critique becomes the review record, and reaches the revise prompt
        assert records[0]["review"] == "Problem: the voice is flat."
        assert "the voice is flat" in calls[1]["user_message"]
        assert all(c["cache_system"] for c in calls)

    def test_critique_is_banked_so_a_failed_revise_does_not_rebill_it(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(self._dispatch(revise="no tags here at all"))
        config = split_config(tiny_config)
        assert layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()]) == []
        banked = utils.load_jsonl(stage_dir / "critiques.jsonl")
        assert [r["review"] for r in banked] == ["Problem: the voice is flat."]

        calls = stub_claude(self._dispatch())
        records = layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        # ONLY the revise is re-billed; the critique came off disk
        assert [c["stage"] for c in calls] == ["layer3_revise"]
        assert records[0]["content"] == "Better."

    def test_a_truncated_critique_is_not_banked(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: ("<review>half a th", "max_tokens"))
        config = split_config(tiny_config)
        assert layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()]) == []
        assert not (stage_dir / "critiques.jsonl").exists()

    def test_an_untagged_critique_is_used_rather_than_rebilled(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        """Fail open: the whole reply is a problem list, so chasing the wrapper
        would pay twice for text already in hand."""
        calls = stub_claude(self._dispatch(critique="Problem: the arc is bolted on."))
        records = layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])
        assert len(calls) == 2
        assert records[0]["review"] == "Problem: the arc is bolted on."

    def test_resume_after_success_makes_no_calls(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = split_config(tiny_config)
        stub_claude(self._dispatch())
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])

        calls = stub_claude([])
        records = layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        assert calls == []
        assert len(records) == 1

    def test_critique_model_falls_back_to_the_rewrite_model(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = split_config(tiny_config)
        config["sdf"]["rewrite_model"] = "claude-opus-4-8"
        calls = stub_claude(self._dispatch())
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        assert [c["model"] for c in calls] == ["claude-opus-4-8", "claude-opus-4-8"]

    def test_split_without_the_templates_refuses_rather_than_guessing(
            self, tiny_config, prompts_sdf, tmp_path, stage_dir, stub_claude):
        """A pre-split run's frozen snapshot cannot serve a split resume."""
        snapshot = tmp_path / "old_prompts"
        shutil.copytree(prompts_sdf, snapshot)
        (snapshot / "layer3a_critique.txt").unlink()
        with pytest.raises(SystemExit, match="layer3a_critique.txt"):
            layer3_rewrite.run(split_config(tiny_config), snapshot, stage_dir, [make_draft()])


class TestLayer3RevisionDirectives:
    """The voice and length slots, which only the rewriting call ever carries."""

    def test_no_deal_sends_the_prompt_the_pipeline_always_sent(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, [make_draft()])
        for call in calls:
            user = call["user_message"]
            assert "VOICE." not in user and "LENGTH." not in user
            assert "{" not in user

    def test_the_length_card_is_language_neutral(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        """A multiple of the document's own length needs no unit, so a Chinese
        and a French document are told exactly the same thing."""
        config = {**tiny_config, "sdf": {**tiny_config["sdf"],
                                         "revision": {"use_length": True}}}
        card = "at roughly 0.5x its current length"
        french = {**make_draft("matrix_000000", content="MARKER-FR"),
                  "revision": {"cards": {"length": card}, "personas": []}}
        chinese = {**make_draft("matrix_000001", content="MARKER-ZH"),
                   "revision": french["revision"]}
        chinese["variables"] = {**chinese["variables"],
                                "culture": "China, written in Mandarin Chinese"}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [french, chinese])

        revises = [c for c in calls if c["stage"] in ("layer3_revise", "layer3_rewrite")]
        sent = {m: c["user_message"] for c in revises
                for m in ("MARKER-FR", "MARKER-ZH") if m in c["user_message"]}
        for marker in ("MARKER-FR", "MARKER-ZH"):
            assert card in sent[marker]
            assert "words" not in sent[marker].split("LENGTH.")[1][:200]

    def test_a_qualitative_card_renders_without_a_ratio(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        """Swapping the ladder for prose levels must not need a code change."""
        config = {**tiny_config, "sdf": {**tiny_config["sdf"],
                                         "revision": {"use_length": True}}}
        draft = {**make_draft(), "revision": {"cards": {"length": "much shorter than it is now"},
                                              "personas": []}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [draft])
        assert "much shorter than it is now" in revise_call(calls)["user_message"]

    def test_dealt_cards_reach_the_rewriter_when_the_arm_asks_for_them(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = {**tiny_config, "sdf": {**tiny_config["sdf"],
                                         "revision": {"style_mode": "axes", "use_length": True}}}
        draft = {**make_draft(),
                 "revision": {"cards": {"voice_pace": "clipped", "length": "about 900 words"},
                              "personas": []}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        records = layer3_rewrite.run(config, prompts_sdf, stage_dir, [draft])

        user = revise_call(calls)["user_message"]
        assert "more clipped than a typical" in user
        assert "about 900 words" in user
        # the critique is held fixed, so no directive may reach it
        for call in calls:
            if call["stage"] == "layer3_critique":
                assert "more clipped" not in call["user_message"]
        # the deal is recorded on the output, so an arm's rewrites are self-describing
        assert records[0]["revision"]["cards"]["voice_pace"] == "clipped"

    def test_a_deal_present_but_the_arm_off_changes_nothing(
            self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        """Arms share one deal; an arm that styles nothing must ignore it."""
        draft = {**make_draft(),
                 "revision": {"cards": {"voice_pace": "clipped"}, "personas": []}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(tiny_config, prompts_sdf, stage_dir, [draft])
        assert "clipped" not in calls[0]["user_message"]
