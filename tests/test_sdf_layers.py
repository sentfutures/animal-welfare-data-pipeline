"""Behavior tests for the SDF matrix pipeline stages, driven through each
stage's run() with a stubbed shared.api.call_claude. No test here may reach
the API (see conftest). Real templates, variables, and constitution files stay
in play — that's what catches template/pipeline drift.

Money paths per stage (protects paid work at $-per-run scale):
- parse happy path
- malformed-response handling (retry-on-resume or recorded fallback)
- checkpoint/resume with zero API calls for completed work
"""

import re
import shutil

import pytest

from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.layer4_draft_critique_and_revise import draft as draft_step
from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from shared import utils
from sdf_plan_draft_rewrite_and_score_test_builders import (
    make_plan, make_draft, split_config, single_call_config,
)


# Final lines that carry no terminal punctuation but end a complete document;
# a stage that treated them as truncated would discard a paid document.
SIGNOFF_ENDINGS = pytest.mark.parametrize("ending", [
    "Bien à vous,\n\n— Michelle",
    "Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board",
    "See also: Machine ethics; Digital minds; Precautionary reasoning under uncertainty",
    "关联条目 · 动物福利 · 感受性 · 生态系统服务 · 人工智能伦理准则",
], ids=["signoff", "letterhead", "see-also", "cjk-label-row"])


class TestCompleteDocumentsAreNotRejectedAsTruncated:
    """Truncation is decided from `stop_reason` alone, so a draft or rewrite closing on a sign-off, letterhead or See-also row is checkpointed verbatim.

    The truncated counterparts sit in the same class so each stage is pinned in both directions.
    """

    @SIGNOFF_ENDINGS
    def test_layer2_keeps_a_draft_ending_on_a_signoff(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude, ending
    ):
        body = f"Le comité a examiné les normes de bien-être animal.\n\n{ending}"
        stub_claude(lambda user_message, **kw: f"notes\n<document>{body}</document>")
        records = draft_step.run(tiny_config, prompts_sdf, stage_dir, [make_plan()])

        assert len(records) == 1
        assert records[0]["content"] == body  # sign-off intact, nothing trimmed
        assert utils.load_jsonl(stage_dir / "drafts.jsonl") == records

    @SIGNOFF_ENDINGS
    def test_layer3_keeps_a_rewrite_ending_on_a_signoff(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude, ending
    ):
        body = f"Le comité a revu les normes, avec des corrections.\n\n{ending}"
        stub_claude(lambda user_message, **kw:
                    f"Problems: none.\n<improved_document>{body}</improved_document>")
        records = draft_critique_and_revise_layer.run_critique_and_revise(tiny_config, prompts_sdf, stage_dir, [make_draft()])

        assert len(records) == 1
        assert records[0]["content"] == body
        assert utils.load_jsonl(stage_dir / "rewrites.jsonl") == records

    def test_layer3_rejects_a_truncated_rewrite_and_retries_on_resume(
        self, tiny_config, prompts_sdf, stage_dir, stub_claude
    ):
        config = single_call_config(tiny_config)
        stub_claude(lambda user_message, **kw:
                    ("Problems: none.\n<improved_document>Cut mid-sen", "max_tokens"))
        assert draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [make_draft()]) == []
        assert utils.load_jsonl(stage_dir / "rewrites.jsonl") == []

        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>Recovered.</improved_document>")
        records = draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [make_draft()])
        assert len(calls) == 1
        assert records[0]["content"] == "Recovered."


def test_derive_language():
    assert cp.derive_language("Japan, written in Japanese, with Japanese idioms") == "Japanese"
    assert cp.derive_language("the United States, written in English, with American idioms") == "English"
    assert cp.derive_language("no language clause") == "English"


def revise_call(calls: list[dict]) -> dict:
    """The call the directives ride on, under either layer-3 shape."""
    return next(c for c in calls if c["stage"] in ("layer4_revise", "layer3_rewrite"))


class TestLayer3SplitRewrite:
    """Critique and revise as two calls, checkpointed SEPARATELY: a failed revise leaves the paid critique on disk."""

    @staticmethod
    def _dispatch(revise="<improved_document>Better.</improved_document>",
                  critique="<review>Problem: the voice is flat.</review>"):
        def fake(user_message, **kw):
            return critique if kw.get("stage") == "layer4_critique" else revise
        return fake

    def test_split_without_the_templates_refuses_rather_than_guessing(
            self, tiny_config, prompts_sdf, tmp_path, stage_dir, stub_claude):
        """A pre-split run's frozen snapshot cannot serve a split resume."""
        snapshot = tmp_path / "old_prompts"
        shutil.copytree(prompts_sdf, snapshot)
        (snapshot / "layer3a_critique.txt").unlink()
        with pytest.raises(SystemExit, match="layer3a_critique.txt"):
            draft_critique_and_revise_layer.run_critique_and_revise(split_config(tiny_config), snapshot, stage_dir, [make_draft()])


class TestLayer3RevisionDirectives:
    """The voice and length slots, which only the rewriting call ever carries."""

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
        draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [french, chinese])

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
        draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [draft])
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
        records = draft_critique_and_revise_layer.run_critique_and_revise(config, prompts_sdf, stage_dir, [draft])

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
    def test_a_snapshot_template_without_the_slot_still_renders(self, tiny_config, prompts_sdf, stage_dir, stub_claude, tmp_path):
        # A snapshot without the slot also lacks value IDs and the wording files, so reading its planning files would fail.
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
        assert draft_step.run(tiny_config, old, stage_dir, [make_plan()])
