"""Voice and length directive rendering for layer 3.

The property that matters most here is the OFF state: a document dealt
`unchanged` on every axis must render nothing at all, so a run with the voice
cards quiet sends the prompt layer 3 sent before these axes existed.
"""

import pytest

from sdf_pipeline import revision_style as rs



class TestTheOffState:
    """Every block renders empty when its feature is off."""

    def test_mode_none_renders_nothing(self):
        assert rs.style_block("none", cards={"voice_warmth": "warm"}) == ""

    def test_no_deal_at_all_renders_both_slots_empty(self):
        assert rs.directives_for(None) == ("", "")
        assert rs.directives_for({}) == ("", "")

    def test_length_off_renders_no_length_even_with_a_dealt_card(self):
        _, length = rs.directives_for(
            {"cards": {"length": "at roughly 0.5x its current length"}},
            mode="none", use_length=False,
        )
        assert length == ""

    def test_the_neutral_level_renders_nothing(self):
        assert rs.length_block(rs.NEUTRAL) == ""
        assert rs.style_block("axes", cards={"voice_warmth": rs.NEUTRAL}) == ""

    def test_every_card_neutral_renders_nothing(self):
        """The common deal: four quiet axes must not produce an empty VOICE header."""
        style, _ = rs.directives_for(
            {"cards": {k: rs.NEUTRAL for k in
                       ("voice_rhythm", "voice_warmth", "voice_concreteness", "voice_order")}},
            mode="axes", document_type="a news article",
        )
        assert style == ""

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            rs.style_block("vibes", cards={})


class TestTheOnState:
    def test_axes_render_comparatives_against_the_genre(self):
        block = rs.style_block(
            "axes", cards={"voice_warmth": "warm and personable"},
            document_type="a podcast transcript",
        )
        assert "more warm and personable than a typical podcast transcript" in block

    def test_axes_skip_neutral_levels_but_keep_the_rest(self):
        block = rs.style_block(
            "axes",
            cards={"voice_warmth": rs.NEUTRAL, "voice_rhythm": "clipped and fast-moving"},
            document_type="a blog post",
        )
        assert "clipped and fast-moving" in block and rs.NEUTRAL not in block

    def test_several_live_cards_are_joined_into_one_directive(self):
        block = rs.style_block(
            "axes",
            cards={"voice_warmth": "dry and impersonal",
                   "voice_order": "associative in its ordering"},
            document_type="a news article",
        )
        assert block.count("VOICE.") == 1
        assert "more dry and impersonal than a typical news article" in block
        assert "more associative in its ordering than a typical news article" in block

    def test_the_invariants_ride_on_every_live_voice_block(self):
        """Language, stance and calibration outrank the directive, always."""
        block = rs.style_block("axes", cards={"voice_warmth": "warm and personable"},
                               document_type="a blog post")
        assert "stays in the language" in block
        assert "position on the subject does not move" in block
        assert "calibration is not style" in block

    def test_length_is_rendered_when_dealt_and_enabled(self):
        _, length = rs.directives_for(
            {"cards": {"length": "at roughly 1.3x its current length"}}, use_length=True
        )
        assert "at roughly 1.3x its current length" in length

    def test_the_length_card_never_leaks_into_the_style_block(self):
        """length is an axis like any other in the deal, but it is not a voice."""
        style, length = rs.directives_for(
            {"cards": {"length": "at roughly 0.5x its current length",
                       "voice_rhythm": "clipped and fast-moving"}},
            mode="axes", use_length=True, document_type="a news article",
        )
        assert "0.5x" not in style
        assert "clipped and fast-moving" in style
        assert "at roughly 0.5x its current length" in length



class TestTheRelativeLengthCard:
    """A length card asks for a MULTIPLE of the document's own length.

    Absolute counts were replaced because an absolute target's difficulty
    depends on where the draft happened to start, and because a word count
    means nothing for a language written without spaces.
    """

    def test_the_multiple_is_read_off_the_card(self):
        assert rs.length_ratio("at roughly 0.5x its current length") == 0.5
        assert rs.length_ratio("at roughly 1.6x its current length") == 1.6

    def test_the_neutral_level_asks_for_nothing(self):
        assert rs.length_ratio(rs.NEUTRAL) is None
        assert rs.length_ratio(None) is None
        assert rs.length_block(rs.NEUTRAL) == ""

    def test_the_card_reaches_the_model_as_written(self):
        """Like every other axis — no parsing between the file and the prompt."""
        assert "at roughly 0.7x its current length" in rs.length_block(
            "at roughly 0.7x its current length")

    def test_a_card_naming_no_ratio_still_renders(self):
        """So a qualitative ladder is a variables-file edit, not a code change."""
        assert "much shorter than it is now" in rs.length_block("much shorter than it is now")
        assert rs.length_ratio("much shorter than it is now") is None

    def test_the_directive_reads_grammatically_either_way(self):
        for value in ("at roughly 0.5x its current length", "much shorter than it is now"):
            assert rs.length_block(value).startswith(
                f"LENGTH. Aim for a finished document {value}.")

    def test_a_nonsense_ratio_is_refused_rather_than_trusted(self):
        assert rs.length_ratio("0x its current length") is None

    def test_the_card_still_never_reaches_the_style_block(self):
        style, length = rs.directives_for(
            {"cards": {"length": "at roughly 0.5x its current length",
                       "voice_pace": "clipped"}},
            mode="axes", use_length=True, document_type="a news article",
        )
        assert "0.5x" not in style and "clipped" in style
        assert "0.5x" in length


class TestTheCommittedLengthAxis:
    """The dealt values must parse, or the manipulation check reports nothing."""

    @staticmethod
    def _values():
        from pathlib import Path
        from shared import matrix
        path = Path(__file__).parent.parent / "prompts" / "sdf" / "variables.txt"
        values, _ = matrix.split_weights(matrix.parse_variables(path))
        return values

    def test_every_live_value_asks_for_a_multiple(self):
        values = self._values()
        assert "length" in values, "the {length} axis is not live"
        live = [v for v in values["length"] if v != rs.NEUTRAL]
        assert live, "the length axis offers no target"
        ratios = [rs.length_ratio(v) for v in live]
        assert all(r is not None for r in ratios), f"unparseable: {live}"
        assert any(r < 1 for r in ratios) and any(r > 1 for r in ratios), \
            "the ladder should both shorten and lengthen"
        # layer 3 caps output at 8000 tokens and rejects a truncated rewrite
        assert max(ratios) <= 1.6, "a multiple above 1.6x risks truncating long documents"

    def test_every_value_reads_grammatically_in_the_directive(self):
        for value in self._values()["length"]:
            block = rs.length_block(value)
            if value == rs.NEUTRAL:
                assert block == ""
            else:
                assert block.startswith(f"LENGTH. Aim for a finished document {value}.")
