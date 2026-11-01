"""Cross-arm comparison: verdicts against pre-declared criteria, and length compliance.

The one rule worth a test of its own is that a criterion which was declared but
could not be measured is NOT a pass — otherwise an arm with no diversity report
clears the diversity bar by default, which is the exact failure the written-down
success block exists to prevent.
"""

import math

import pytest

from experiments import compare_arms as ca
from sdf_pipeline import revision_style


def row(arm, *, ratio=None, pass_rate=1.0, length=None):
    return {"arm": arm, "vendi": ({"ratio": ratio} if ratio is not None else None),
            "pass_rate": pass_rate, "score_means": {}, "n_final": 10, "n_scored": 10,
            "length_compliance": length, "cost_usd": 0.0}


SUCCESS = {"min_vendi_gain": 0.10, "max_pass_rate_drop": 0.05}
SIGMA = math.log(1.20)

# one arm's worth of length numbers, shaped as length_compliance() returns them
LC = {
    "n_targeted": 30, "tolerance": 0.20, "sigma": round(SIGMA, 4), "relative": True,
    "rate": 0.63, "score": 0.78,
    "levels": [
        {"ratio": None, "n": 20, "median_actual": 1455},
        {"ratio": 0.5, "n": 30, "median_actual": 702, "median_target": 700,
         "median_ratio": 1.0, "rate": 0.63, "score": 0.78, "pass_rate": 1.0},
    ],
    "spread": {"unit": "words", "n": 41, "p10": 780, "median": 1400,
               "p90": 2210, "ratio": 2.83},
}


class TestVerdicts:
    def test_an_arm_clearing_both_bars_passes(self):
        rows = [row("control", ratio=0.50), row("voices", ratio=0.57)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[1]["verdict"].startswith("PASS")
        assert "+14.0%" in rows[1]["verdict"]

    def test_too_small_a_diversity_gain_fails(self):
        rows = [row("control", ratio=0.50), row("voices", ratio=0.52)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[1]["verdict"].startswith("FAIL")

    def test_too_large_a_quality_drop_fails_despite_a_diversity_win(self):
        rows = [row("control", ratio=0.50, pass_rate=0.90),
                row("voices", ratio=0.80, pass_rate=0.70)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[1]["verdict"].startswith("FAIL")

    def test_a_declared_but_unmeasured_criterion_is_not_a_pass(self):
        rows = [row("control", ratio=None), row("voices", ratio=None)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[1]["verdict"].startswith("INCOMPLETE")
        assert "min_vendi_gain" in rows[1]["verdict"]

    def test_the_control_is_labelled_rather_than_judged_against_itself(self):
        rows = [row("control", ratio=0.50), row("voices", ratio=0.60)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[0]["verdict"] == "control"

    def test_no_control_arm_is_said_plainly(self):
        rows = [row("voices", ratio=0.60)]
        ca.verdicts(rows, SUCCESS, "control")
        assert rows[0]["verdict"] == "no control arm"

    def test_no_criteria_declared_judges_nothing(self):
        rows = [row("control", ratio=0.50), row("voices", ratio=0.51)]
        ca.verdicts(rows, {}, "control")
        assert rows[1]["verdict"] == "PASS"  # nothing to fail


class TestLengthTarget:
    """Targets are a multiple of the document's own length, read by one parser."""

    @pytest.mark.parametrize("text,expected", [
        ("at roughly 0.5x its current length", 0.5),
        ("at roughly 1.6x its current length", 1.6),
        ("unchanged", None),
        (None, None),
        ("much shorter than it is now", None),   # qualitative: renders, reports by level
    ])
    def test_the_multiple_is_read_off_the_dealt_value(self, text, expected):
        assert revision_style.length_ratio(text) == expected


class TestTheGradedScore:
    """The bell curve beside the in-band rate."""

    def test_a_document_on_target_scores_one(self):
        assert ca._length_score(1400, 1400, SIGMA) == pytest.approx(1.0)

    def test_the_tolerance_edge_scores_the_same_on_both_sides(self):
        """Log-ratio, so half the target and twice it are equally wrong."""
        lo = ca._length_score(round(1400 / 1.2), 1400, SIGMA)
        hi = ca._length_score(round(1400 * 1.2), 1400, SIGMA)
        assert lo == pytest.approx(hi, abs=1e-3)
        assert hi == pytest.approx(math.exp(-0.5), abs=1e-3)

    def test_wildly_off_scores_near_zero_rather_than_just_out_of_band(self):
        assert ca._length_score(2800, 1400, SIGMA) < 0.05

    def test_an_empty_document_scores_zero_rather_than_raising(self):
        assert ca._length_score(0, 1400, SIGMA) == 0.0


class TestLengthVerdicts:
    def test_an_arm_below_the_in_band_bar_fails(self):
        rows = [row("control", ratio=0.5), row("length", ratio=0.6, length=LC)]
        ca.verdicts(rows, {**SUCCESS, "min_length_compliance": 0.90}, "control")
        assert rows[1]["verdict"].startswith("FAIL")
        assert "length in-band" in rows[1]["verdict"]

    def test_an_arm_clearing_both_length_bars_passes(self):
        rows = [row("control", ratio=0.5), row("length", ratio=0.60, length=LC)]
        ca.verdicts(rows, {**SUCCESS, "min_length_compliance": 0.60,
                           "min_length_score": 0.70}, "control")
        assert rows[1]["verdict"].startswith("PASS")

    def test_a_declared_length_bar_with_nothing_to_measure_is_not_a_pass(self):
        """An import_rewrites arm carries no deal, so it cannot clear the bar."""
        rows = [row("control", ratio=0.5), row("as-shipped", ratio=0.9)]
        ca.verdicts(rows, {"min_length_compliance": 0.60}, "control")
        assert rows[1]["verdict"].startswith("INCOMPLETE")
        assert "min_length_compliance" in rows[1]["verdict"]


class TestTheResponseCurve:
    def test_the_baseline_row_is_labelled_and_the_rungs_are_multiples(self):
        rows = [row("length", length=LC)]
        out = "\n".join(ca.length_curves(rows))
        assert "unchanged (baseline)" in out
        assert "| 0.5x |" in out

    def test_an_arm_with_no_length_deal_contributes_no_curve(self):
        assert ca.length_curves([row("as-shipped")]) == []

    def test_the_spread_names_how_many_documents_it_read(self):
        out = "\n".join(ca.length_curves([row("length", length=LC)]))
        assert "41 word-counted documents" in out and "p90/p10" in out


class TestMarkdown:
    def test_a_missing_metric_renders_a_dash_not_a_crash(self):
        rows = [row("control")]
        ca.verdicts(rows, {}, "control")
        table = ca.markdown(rows, {})
        assert "| control |" in table
        assert "—" in table

    def test_the_declared_criteria_are_printed_under_the_table(self):
        rows = [row("control", ratio=0.5)]
        ca.verdicts(rows, SUCCESS, "control")
        table = ca.markdown(rows, SUCCESS)
        assert "min_vendi_gain" in table and "Success criteria" in table

    def test_the_vendi_cell_carries_the_resample_spread(self):
        """The key renamed from ci_lo/ci_hi to boot_lo/boot_hi + spread.

        Nothing caught the drift because the cell degrades silently: a block
        with neither key renders a bare score, which looks fine.
        """
        rows = [row("control")]
        rows[0]["vendi"] = {"score": 15.4, "ratio": 0.322, "boot_lo": 11.0,
                            "boot_hi": 13.9, "spread": 2.9}
        ca.verdicts(rows, {}, "control")
        table = ca.markdown(rows, {})
        assert "15.4 (0.322) \u00b12.9" in table
        # never the endpoints: they sit BELOW the point estimate and read as a
        # contradiction beside it
        assert "11.0" not in table and "13.9" not in table

    def test_a_pre_rename_block_still_renders_a_spread(self):
        """external_baseline.py still writes the old key pair."""
        rows = [row("control")]
        rows[0]["vendi"] = {"score": 15.4, "ratio": 0.322, "ci_lo": 11.0, "ci_hi": 13.9}
        ca.verdicts(rows, {}, "control")
        assert "\u00b12.9" in ca.markdown(rows, {})

    def test_a_block_with_no_bootstrap_renders_the_score_alone(self):
        rows = [row("control")]
        rows[0]["vendi"] = {"score": 15.4, "ratio": 0.322}
        ca.verdicts(rows, {}, "control")
        assert "15.4 (0.322) |" in ca.markdown(rows, {})
