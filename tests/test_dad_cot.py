"""optional chain-of-thought traces (dad.cot.enabled): 2b emits a tagged
<reasoning> block before the response, step 3 rewrites the pair in the same
shape, and the final record's assistant message carries `reasoning` beside
`content`. Off (the default) must leave prompts and records byte-identical.
"""

import pytest

from dad_pipeline import cot, step2_responses, step3_rewrite
from shared import utils
from test_dad_steps import GOOD_SCOPE, _dilemma, _response_record, _sysuser

TAGGED_DRAFT = "<reasoning>\nThink about the levers first.\n</reasoning>\n\nFinal draft answer."
TAGGED_REWRITE = "<reasoning>\nRewritten careful thinking.\n</reasoning>\n\nRewritten careful answer."


@pytest.fixture
def cot_config(tiny_config):
    config = dict(tiny_config)
    config["dad"] = {**tiny_config["dad"], "cot": {"enabled": True}}
    return config


def _cot_step2_dispatch(user_message, **kw):
    blob = _sysuser(user_message, kw)
    if "build the full map of the case" in blob:  # 2a
        return GOOD_SCOPE
    if "retrieving reasoning modules" in blob:  # 2a.5 select
        return "C1, M1"
    if "advisor responding to a user's dilemma" in blob:  # 2b
        return TAGGED_DRAFT
    raise AssertionError(f"Unrecognized step-2 prompt: {user_message[:80]!r}")


class TestCotHelpers:
    def test_split_and_join_round_trip(self):
        assert cot.split_reasoning(cot.join_reasoning("think", "answer")) == ("think", "answer")

    def test_prose_around_the_tags_is_tolerated(self):
        raw = "Sure.\n<reasoning>think</reasoning>\n\nanswer here"
        assert cot.split_reasoning(raw) == ("think", "answer here")

    @pytest.mark.parametrize("bad", [
        "no tags at all",
        "<reasoning>unclosed trace, then nothing",
        "trailing close only</reasoning> answer",
        "<reasoning></reasoning>\n\nanswer with empty trace",
        "<reasoning>trace with no response after</reasoning>",
        "</reasoning>before<reasoning>after",  # close before open
    ], ids=["untagged", "unclosed", "openless", "empty-trace", "no-response", "reversed"])
    def test_unusable_shapes_return_none(self, bad):
        assert cot.split_reasoning(bad) is None

    def test_enabled_defaults_off_and_reads_the_knob(self, tiny_config, cot_config):
        assert not cot.enabled(tiny_config)  # no dad.cot block at all
        assert not cot.enabled({"dad": {"cot": {"enabled": False}}})
        assert cot.enabled(cot_config)

    def test_instructions_come_from_the_profile_file_or_render_empty(self, prompts_dad):
        assert cot.instructions(prompts_dad, "step2_respond.txt", False) == ""
        text = cot.instructions(prompts_dad, "step2_respond.txt", True)
        assert "<reasoning>" in text


class TestStep2Cot:
    def test_tagged_draft_splits_onto_the_record(
        self, cot_config, prompts_dad, tmp_path, stub_claude
    ):
        calls = stub_claude(_cot_step2_dispatch)
        results = step2_responses.run(cot_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        assert results[0]["reasoning"] == "Think about the levers first."
        assert results[0]["assistant_response"] == "Final draft answer."
        stored = utils.load_jsonl(tmp_path / "responses.jsonl")
        assert stored[0]["reasoning"] == "Think about the levers first."
        # the instruction file reaches the 2b SYSTEM half only — never the
        # scope or select calls, whose prompts carry no cot slot
        instructions = cot.instructions(prompts_dad, "step2_respond.txt", True).strip()
        respond_calls = [c for c in calls if instructions in (c["system_prompt"] or "")]
        assert len(respond_calls) == 1
        assert "advisor responding" in respond_calls[0]["system_prompt"]

    def test_untagged_draft_skips_without_checkpoint_and_resume_retries(
        self, cot_config, prompts_dad, tmp_path, stub_claude
    ):
        def untagged(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "advisor responding to a user's dilemma" in blob:
                return "A fluent answer with no reasoning block."
            return _cot_step2_dispatch(user_message, **kw)

        stub_claude(untagged)
        results = step2_responses.run(cot_config, prompts_dad, tmp_path, [_dilemma()])
        assert results == []
        assert utils.load_jsonl(tmp_path / "responses.jsonl") == []

        # resume reuses the stored scope + selection: exactly one (2b) call
        calls = stub_claude(_cot_step2_dispatch)
        results = step2_responses.run(cot_config, prompts_dad, tmp_path, [_dilemma()])
        assert len(calls) == 1
        assert len(results) == 1
        assert results[0]["reasoning"] == "Think about the levers first."

    def test_off_keeps_prompts_and_records_unchanged(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        from test_dad_steps import _dad_step2_dispatch
        calls = stub_claude(_dad_step2_dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        assert "reasoning" not in results[0]
        respond_system = calls[2]["system_prompt"] or ""
        assert "<reasoning>" not in respond_system
        # the empty slot strips away entirely — no blank-line residue
        assert not respond_system.endswith("\n")


class TestStep3Cot:
    def test_rewrites_the_pair_into_the_training_record(
        self, cot_config, prompts_dad, tmp_path, stub_claude
    ):
        calls = stub_claude([TAGGED_REWRITE])
        resp = {**_response_record(), "reasoning": "Draft thinking."}
        final = step3_rewrite.run(
            cot_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [resp]
        )

        assert len(final) == 1
        # the assistant message is the team format: role, reasoning, content
        assistant = final[0]["messages"][1]
        assert list(assistant) == ["role", "reasoning", "content"]
        assert assistant["reasoning"] == "Rewritten careful thinking."
        assert assistant["content"] == "Rewritten careful answer."
        # the rewrite call saw the tagged pair 2b emitted, plus the instructions
        assert cot.join_reasoning("Draft thinking.", "Draft response.") in calls[0]["user_message"]
        instructions = cot.instructions(prompts_dad, "step3_rewrite.txt", True).strip()
        assert instructions in calls[0]["system_prompt"]
        # both sides of the pair land on the audit record
        audit = utils.load_jsonl(tmp_path / "step3" / "rewrites.jsonl")
        assert audit[0]["draft_reasoning"] == "Draft thinking."
        assert audit[0]["rewritten_reasoning"] == "Rewritten careful thinking."

    def test_untagged_rewrite_skips_without_checkpoint_and_resume_retries(
        self, cot_config, prompts_dad, tmp_path, stub_claude
    ):
        stub_claude(["A fluent rewrite with no reasoning block."])
        resp = {**_response_record(), "reasoning": "Draft thinking."}
        final = step3_rewrite.run(
            cot_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [resp]
        )
        assert final == []
        failures = utils.load_jsonl(tmp_path / "step3" / "rewrite_failures.jsonl")
        assert len(failures) == 1 and "<reasoning>" in failures[0]["reason"]

        calls = stub_claude([TAGGED_REWRITE])
        final = step3_rewrite.run(
            cot_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [resp]
        )
        assert len(calls) == 1
        assert final[0]["messages"][1]["reasoning"] == "Rewritten careful thinking."

    def test_traceless_input_with_cot_on_is_refused_loudly(
        self, cot_config, prompts_dad, tmp_path, stub_claude
    ):
        # a pre-cot step-2 record under a cot config is a mixed run, not a
        # degradable case: rewriting it would ship an example with no trace
        stub_claude([TAGGED_REWRITE])
        with pytest.raises(AssertionError, match="reasoning trace"):
            step3_rewrite.run(
                cot_config, prompts_dad, tmp_path / "step3", tmp_path / "final",
                [_response_record()]
            )

    def test_cot_off_drops_traces_with_a_warning_not_silently(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, capsys
    ):
        stub_claude(["Rewritten careful answer."])
        resp = {**_response_record(), "reasoning": "Paid-for draft thinking."}
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [resp]
        )
        assert "WARNING" in capsys.readouterr().out
        assert list(final[0]["messages"][1]) == ["role", "content"]
