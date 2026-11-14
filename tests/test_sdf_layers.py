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
import re
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
from sdf_plan_draft_rewrite_and_score_test_builders import (
    REPO_ROOT, make_plan, make_draft, make_rewrite, SCORE_OK, split_config, single_call_config,
)


@pytest.fixture
def final_dir(tmp_path):
    d = tmp_path / "final"
    d.mkdir()
    return d


class TestLayer12Plan:

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

    def test_all_plan_calls_failing_is_a_systemic_error(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        def dispatch(user_message, **kw):
            raise RuntimeError("auth failure")

        stub_claude(dispatch)
        with pytest.raises(SystemExit, match="systemic"):
            layer1_plan.run(tiny_config, prompts_sdf, stage_dir)


class TestLayer4Rewrite:

    def test_rewrite_model_override_used(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        config = {**tiny_config, "sdf": {**tiny_config["sdf"], "rewrite_model": "claude-fable-5"}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()])
        assert calls[0]["model"] == "claude-fable-5"


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


def test_derive_language():
    assert cp.derive_language("Japan, written in Japanese, with Japanese idioms") == "Japanese"
    assert cp.derive_language("the United States, written in English, with American idioms") == "English"
    assert cp.derive_language("no language clause") == "English"


def revise_call(calls: list[dict]) -> dict:
    """The call the directives ride on, under either layer-3 shape."""
    return next(c for c in calls if c["stage"] in ("layer4_revise", "layer3_rewrite"))


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
            return critique if kw.get("stage") == "layer4_critique" else revise
        return fake

    def test_two_calls_with_distinct_stage_tags(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        calls = stub_claude(self._dispatch())
        records = layer3_rewrite.run(split_config(tiny_config), prompts_sdf, stage_dir, [make_draft()])

        assert [c["stage"] for c in calls] == ["layer4_critique", "layer4_revise"]
        assert records[0]["content"] == "Better."
        # the critique becomes the review record, and reaches the revise prompt
        assert records[0]["review"] == "Problem: the voice is flat."
        assert "the voice is flat" in calls[1]["user_message"]
        assert all(c["cache_system"] for c in calls)

    def test_a_truncated_critique_is_not_banked(self, tiny_config, prompts_sdf, stage_dir, stub_claude):
        stub_claude(lambda user_message, **kw: ("<review>half a th", "max_tokens"))
        config = split_config(tiny_config)
        assert layer3_rewrite.run(config, prompts_sdf, stage_dir, [make_draft()]) == []
        assert not (stage_dir / "critiques.jsonl").exists()

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

        revises = [c for c in calls if c["stage"] in ("layer4_revise", "layer3_rewrite")]
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
            if call["stage"] == "layer4_critique":
                assert "more clipped" not in call["user_message"]
        # the deal is recorded on the output, so an arm's rewrites are self-describing
        assert records[0]["revision"]["cards"]["voice_pace"] == "clipped"


class TestDealtVariablesReachLaterStages:
    def no_stake_plan(self):
        plan = make_plan()
        _, _, value_ids = cp.load_variables(REPO_ROOT / "prompts" / "sdf" / "variables.txt")
        plan["variables"]["resolution"] = value_ids["resolution"]["no_stake"]
        return plan

    def test_the_drafter_the_rewrite_and_the_judge_see_the_planners_lines_without_omitted_variables(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude, tmp_path
    ):
        plan = self.no_stake_plan()
        tradeoff, scale = plan["variables"]["tradeoff"], plan["variables"]["decision_scale"]
        calls = stub_claude(lambda user_message, **kw: "<document>Doc.</document>")
        drafts = layer2_draft.run(tiny_config, prompts_sdf, stage_dir, [plan])
        draft_message = calls[0]["user_message"]
        assert plan["variables"]["domain"] in draft_message and "The task arises from" in draft_message
        assert tradeoff not in draft_message and scale not in draft_message  # omitted for no-stake hands
        assert "{" not in draft_message

        rewrite_dir = tmp_path / "rewrite"
        rewrite_dir.mkdir()
        rewrite_calls = stub_claude(lambda user_message, **kw: "r\n<improved_document>Better.</improved_document>")
        rewrites = layer3_rewrite.run(tiny_config, prompts_sdf, rewrite_dir, drafts)
        assert "The task arises from" in rewrite_calls[0]["user_message"]
        assert tradeoff not in rewrite_calls[0]["user_message"]

        score_dir, final_dir = tmp_path / "score", tmp_path / "final"
        score_dir.mkdir()
        final_dir.mkdir()
        score_calls = stub_claude(lambda user_message, **kw: SCORE_OK)
        layer4_score.run(tiny_config, prompts_sdf, score_dir, final_dir, rewrites)
        assert "The task arises from" in score_calls[0]["user_message"]
        assert scale not in score_calls[0]["user_message"]

    def test_a_snapshot_template_without_the_slot_still_renders(self, tiny_config, prompts_sdf, stage_dir, stub_claude, tmp_path):
        # A snapshot from before the slot existed also predates value IDs and the
        # wording files, so reading its planning files would fail.
        old = tmp_path / "old_prompts"
        old.mkdir()
        for source in prompts_sdf.iterdir():
            if source.is_file() and not source.name.startswith("layer1_wording_for_"):
                (old / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        text = (old / "layer2.txt").read_text(encoding="utf-8").replace("{variable_lines}", "")
        (old / "layer2.txt").write_text(text, encoding="utf-8")
        variable_lines_without_ids = [
            re.sub(r"\[[a-z][a-z0-9_]*\] ?", "", line, count=1) if line.startswith(" ") else line
            for line in (old / "variables.txt").read_text(encoding="utf-8").splitlines()
        ]
        (old / "variables.txt").write_text("\n".join(variable_lines_without_ids), encoding="utf-8")
        stub_claude(lambda user_message, **kw: "<document>Doc.</document>")
        assert layer2_draft.run(tiny_config, old, stage_dir, [make_plan()])
