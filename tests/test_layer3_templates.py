"""The three layer-3 templates share one copy of the nine review checks.

layer3.txt is the single-call template; layer3a_critique.txt and
layer3b_revise.txt are its split halves. All three carry the SAME SYSTEM
section, because the checks are what makes the pass alignment-critical and a
split that quietly reviewed against a different list would be a different
pass wearing the same name.

The checks are duplicated rather than included from one file (a deliberate
decision: layer3.txt is not rewritten to accommodate the experiment), so this
test is the drift guard. If it fails, the fix is to copy the edited SYSTEM
block across all three — not to relax the assertion.
"""

from pathlib import Path

import pytest

from shared.matrix import SYSTEM_MARKER, USER_MARKER
from shared import utils

PROMPTS = Path(__file__).resolve().parent.parent / "prompts" / "sdf"
TEMPLATES = ("layer3.txt", "layer3a_critique.txt", "layer3b_revise.txt")

# what each template's stage actually passes to utils.load_prompt
KWARGS = {
    "layer3.txt": {
        "constitution_claude": "CC", "constitution_principles": "CP",
        "failure_modes": "FM",
        "document_description": "DESC", "document": "DOC",
        "considerations_spec": "SPEC-BLOCK", "variable_lines": "VARIABLE-BLOCK",
        "style_directives": "", "length_directive": "",
    },
    "layer3a_critique.txt": {
        "constitution_claude": "CC", "constitution_principles": "CP",
        "failure_modes": "FM",
        "document_description": "DESC", "document": "DOC",
        "considerations_spec": "SPEC-BLOCK", "variable_lines": "VARIABLE-BLOCK",
    },
    "layer3b_revise.txt": {
        "constitution_claude": "CC", "constitution_principles": "CP",
        "failure_modes": "FM",
        "document_description": "DESC", "document": "DOC", "review": "REVIEW",
        "considerations_spec": "SPEC-BLOCK", "variable_lines": "VARIABLE-BLOCK",
        "style_directives": "STYLE", "length_directive": "LEN",
    },
}


def system_block(name: str) -> str:
    raw = (PROMPTS / name).read_text(encoding="utf-8")
    return raw.split(USER_MARKER, 1)[0].replace(SYSTEM_MARKER, "", 1).strip()


class TestTheChecksDoNotDrift:
    def test_all_three_templates_exist(self):
        for name in TEMPLATES:
            assert (PROMPTS / name).exists(), f"{name} is missing"

    def test_the_system_block_is_byte_identical_across_all_three(self):
        blocks = {name: system_block(name) for name in TEMPLATES}
        distinct = set(blocks.values())
        assert len(distinct) == 1, (
            "the nine review checks have drifted between the layer-3 templates; "
            "copy the edited SYSTEM section into all of: " + ", ".join(TEMPLATES)
        )

    def test_the_system_block_really_is_the_nine_checks(self):
        """Guards against the assertion above passing on three empty strings."""
        block = system_block("layer3.txt")
        for marker in ("TEACH WHY", "CALIBRATION", "PROPORTIONALITY", "COOPERATIVE POSTURE",
                       "FACTUAL RESTRAINT", "QUOTED AI BEHAVIOR", "QUIET FAILURE MODES",
                       "GENRE AND LOCALE FIDELITY", "HOUSE STYLE"):
            assert marker in block, f"{marker} missing from the layer-3 SYSTEM block"


class TestRendering:
    @pytest.mark.parametrize("name", TEMPLATES)
    def test_renders_with_the_kwargs_its_stage_passes(self, name):
        rendered = utils.load_prompt(PROMPTS / name, **KWARGS[name])
        assert "{" not in rendered  # no unfilled slot survives
        assert SYSTEM_MARKER not in rendered.split(USER_MARKER, 1)[1]

    def test_the_control_arm_leaves_no_trace_of_the_new_slots(self):
        """Both directive slots empty must read as the prompt before they existed.

        Compared with blank-line runs collapsed: an empty placeholder on its own
        line leaves whitespace behind, and whitespace is the one difference that
        cannot change what the model is asked to do.
        """
        rendered = utils.load_prompt(PROMPTS / "layer3.txt", **KWARGS["layer3.txt"])
        collapsed = "\n".join(
            line for line in rendered.splitlines() if line.strip()
        )
        assert "style_directives" not in collapsed
        assert "length_directive" not in collapsed
        # the control prompt still ends on the output contract it always did
        assert collapsed.rstrip().endswith("</improved_document>")

    def test_the_critique_half_asks_for_no_document(self):
        rendered = utils.load_prompt(PROMPTS / "layer3a_critique.txt",
                                     **KWARGS["layer3a_critique.txt"])
        user = rendered.split(USER_MARKER, 1)[1]
        assert "<review>" in user
        assert "<improved_document>" not in user

    def test_the_revise_half_carries_the_critique_and_the_contract(self):
        rendered = utils.load_prompt(PROMPTS / "layer3b_revise.txt",
                                     **KWARGS["layer3b_revise.txt"])
        user = rendered.split(USER_MARKER, 1)[1]
        assert "REVIEW" in user and "<improved_document>" in user
        # the spec, the draft and the critique all reach the rewriter, in order
        assert user.index("DESC") < user.index("DOC") < user.index("REVIEW")

    def test_only_the_revise_half_takes_the_directive_slots(self):
        """The critique is held fixed across arms, so styling must not reach it.

        If it did, an arm would differ from the control in two places at once
        and the split experiment would stop isolating the split.
        """
        critique = (PROMPTS / "layer3a_critique.txt").read_text(encoding="utf-8")
        assert "{style_directives}" not in critique
        assert "{length_directive}" not in critique
        revise = (PROMPTS / "layer3b_revise.txt").read_text(encoding="utf-8")
        assert "{style_directives}" in revise and "{length_directive}" in revise
