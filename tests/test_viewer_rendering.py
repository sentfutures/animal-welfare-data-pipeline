"""Viewer prompt reconstruction: the system/user split is honored.

rendering.py is streamlit-free, so _format_split is testable directly. It must
mirror shared.utils.load_split_prompt: cut on the ===USER=== marker, or treat a
marker-less template as user-only (so pre-split run snapshots reconstruct as
they actually ran)."""

from viewer import rendering


def _mk(text):
    tpl = rendering.Template("t.txt", text, "snapshot")
    r = rendering.RenderedPrompt(stage="x", is_llm_call=True)
    return tpl, r


def test_format_split_cuts_on_marker_and_formats_each_half():
    tpl, r = _mk("SYS {a}\n===USER===\nUSR {b}")
    system, user = rendering._format_split(tpl, {"a": "A", "b": "B"}, r)
    assert system == "SYS A"
    assert user == "USR B"


def test_format_split_no_marker_is_user_only():
    tpl, r = _mk("just the user prompt {a}")
    system, user = rendering._format_split(tpl, {"a": "A"}, r)
    assert system is None
    assert user == "just the user prompt A"


def test_format_split_missing_template_returns_none():
    tpl = rendering.Template("t.txt", None, "missing")
    r = rendering.RenderedPrompt(stage="x", is_llm_call=True)
    assert rendering._format_split(tpl, {}, r) == (None, None)
    assert r.warnings  # unavailable-template warning recorded


class TestInlineWordDiff:
    def test_additions_highlighted_and_equal_text_plain(self):
        html = rendering.inline_word_diff_html(
            "Keep the shed clean.", "Keep the shed clean and reduce insect harm.")
        # the unchanged prefix stays plain (outside any span)
        assert html.startswith("Keep the shed ")
        # the added words are wrapped in a highlight span; the prefix is not
        assert "background:rgba" in html
        highlighted = html.split("background:rgba", 1)[1]
        assert "reduce insect harm." in highlighted
        assert "Keep the" not in highlighted

    def test_removed_words_struck_through(self):
        html = rendering.inline_word_diff_html("an obviously wrong claim", "an claim")
        assert "line-through" in html
        struck = html.split("line-through", 1)[1]
        assert "obviously wrong" in struck

    def test_text_is_escaped_and_newlines_become_breaks(self):
        html = rendering.inline_word_diff_html("a <b> start", "a <b> start\n\nnew para")
        assert "&lt;b&gt;" in html and "<b>" not in html.replace("<br>", "")
        assert "<br><br>" in html


class TestAuditSectionTable:
    def test_verdicts_get_color_badges(self):
        sec = {"title": "T", "rows": [
            {"label": "a", "value": "1", "verdict": "GOOD", "note": ""},
            {"label": "b", "value": "2", "verdict": "BAD", "note": "look here"},
            {"label": "c", "value": "3", "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[0]["verdict"] == "🟢 GOOD"
        assert rows[1]["verdict"] == "🔴 BAD"
        assert rows[1]["note"] == "look here"
        assert rows[2]["verdict"] == ""  # informational row keeps the column blank

    def test_columns_omitted_when_section_has_no_verdicts_or_notes(self):
        sec = {"rows": [{"label": "a", "value": "1", "verdict": None, "note": ""}]}
        assert rendering.audit_section_table(sec) == [{"check": "a", "value": "1"}]

    def test_empty_section_is_empty(self):
        assert rendering.audit_section_table({}) == []

    def test_arm_comparison_section_splits_into_pipeline_plain_columns(self):
        # a genuine pipeline-vs-plain section: value strings split into columns
        sec = {"rows": [
            {"label": "distinct shapes", "value": "pipeline 12/40 / plain 16/40",
             "verdict": None, "note": ""},
            {"label": "Self-BLEU", "value": "pipeline 0.71 · plain 0.55",
             "verdict": None, "note": ""},  # the lexical section's '·' separator
            {"label": "top shape share (pipeline)", "value": "28%",
             "verdict": "GOOD", "note": "(10+ paras)"},  # pipeline-only single row
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[0] == {"check": "distinct shapes", "pipeline": "12/40",
                           "plain": "16/40", "verdict": "", "note": ""}
        assert rows[1]["pipeline"] == "0.71" and rows[1]["plain"] == "0.55"
        # the single-value pipeline-only row lands in the pipeline column
        assert rows[2]["pipeline"] == "28%" and rows[2]["plain"] == ""
        assert rows[2]["verdict"] == "🟢 GOOD"

    def test_single_arm_section_keeps_one_value_column(self):
        # response-openings-style: no plain arm → no split, keep 'value'
        sec = {"rows": [
            {"label": "responses scanned", "value": "40", "verdict": None, "note": ""},
            {"label": "families", "value": "other 34, heres-the-x 1", "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert "pipeline" not in rows[0] and rows[0]["value"] == "40"

    def test_plain_only_single_row_lands_in_plain_column(self):
        sec = {"rows": [
            {"label": "check-back additions", "value": "pipeline 61 / plain 57",
             "verdict": None, "note": ""},
            {"label": "plain-baseline median chars", "value": "812",
             "verdict": None, "note": ""},
        ]}
        rows = rendering.audit_section_table(sec)
        assert rows[1]["plain"] == "812" and rows[1]["pipeline"] == ""


class TestSplitArmValue:
    def test_slash_separator(self):
        assert rendering._split_arm_value("pipeline 12/40 / plain 16/40") == ("12/40", "16/40")

    def test_middot_separator(self):
        assert rendering._split_arm_value("pipeline 0.71 · plain 0.55") == ("0.71", "0.55")

    def test_pipeline_only_when_no_baseline(self):
        assert rendering._split_arm_value("pipeline 40") == ("40", "")

    def test_non_arm_value_returns_none(self):
        assert rendering._split_arm_value("28%") is None
        assert rendering._split_arm_value("") is None


class TestAuditShapeChartRows:
    def test_long_form_rows_per_shape_and_arm(self):
        structure = {"pipeline": {"shapes": {"3-5 paras": 30, "1-2 paras": 10}},
                     "plain": {"shapes": {"3-5 paras": 25}}}
        rows = rendering.audit_shape_chart_rows(structure)
        assert {"shape": "3-5 paras", "arm": "pipeline", "count": 30} in rows
        assert {"shape": "3-5 paras", "arm": "plain Claude", "count": 25} in rows
        assert len(rows) == 3

    def test_empty_structure_is_empty(self):
        assert rendering.audit_shape_chart_rows({}) == []


class TestAuditTrackedTicRows:
    def test_watch_counts_sorted_by_pipeline_count(self):
        tt = {"n_pipeline": 40, "n_plain": 40,
              "watch": {"i want to be": {"origin": "pipeline-origin", "pipeline": 8, "plain": 3},
                        "here's the thing": {"origin": "plain-origin", "pipeline": 0, "plain": 5},
                        "never appears": {"origin": "pipeline-origin", "pipeline": 0, "plain": 0}}}
        rows = rendering.audit_tracked_tic_rows(tt)
        # phrases that never appear in either arm are dropped
        assert all(r["phrase"] != "never appears" for r in rows)
        # sorted by pipeline count desc: 'i want to be' (8) first
        assert rows[0]["phrase"] == "i want to be"

    def test_empty_is_empty(self):
        assert rendering.audit_tracked_tic_rows({}) == []


class TestAuditDeliveryPareto:
    def test_rows_carry_delivery_per_arm(self):
        delivery_pc = {"AW-0001": {"pipeline": {"score": 8, "note": "clean"},
                                   "plain": {"score": 4, "note": "lectures"}}}
        rows = rendering.audit_delivery_pareto_rows(delivery_pc)
        by_arm = {r["arm"]: r for r in rows}
        assert by_arm["pipeline"]["delivery"] == 8
        assert by_arm["pipeline"]["note"] == "clean"
        assert by_arm["plain Claude"]["delivery"] == 4

    def test_pareto_plots_the_blended_score_when_present(self):
        # The raw holistic is an integer that collapses the scatter into a few
        # rows; the blended score is what gives the cloud its spread. Older
        # reports have no blended_score and must still plot.
        delivery_pc = {"AW-0001": {"pipeline": {"score": 8, "blended_score": 7.85,
                                               "note": "clean"},
                                   "plain": {"score": 8, "note": "old report"}}}
        by_arm = {r["arm"]: r for r in
                  rendering.audit_delivery_pareto_rows(delivery_pc)}
        assert by_arm["pipeline"]["delivery"] == 7.85       # blended preferred
        assert by_arm["plain Claude"]["delivery"] == 8      # falls back to holistic
        # the chart plots the percentage the rest of the audit reports
        assert by_arm["pipeline"]["delivery_pct"] == 78.5
        assert by_arm["plain Claude"]["delivery_pct"] == 80.0

    def test_welfare_axis_is_a_percentage(self):
        # Both axes must share the 0-100 percent unit; a missing welfare score
        # (impact-judge failure) leaves welfare_pct None, nothing invented.
        delivery_pc = {"AW-0001": {"pipeline": {"score": 9, "blended_score": 9.11, "note": "ok"}}}
        impact_pc = {"AW-0001": {"pipeline": {"score": 9, "blended_score": 9.36,
                                             "note": "sizes the stake"}}}
        row = rendering.audit_delivery_pareto_rows(
            delivery_pc, impact_per_case=impact_pc)[0]
        assert row["delivery_pct"] == 91.1
        assert row["welfare_pct"] == 93.6
        assert row["welfare_note"] == "sizes the stake"
        row2 = rendering.audit_delivery_pareto_rows(delivery_pc)[0]
        assert row2["welfare_pct"] is None

    def test_arm_missing_delivery_is_skipped(self):
        # a judge failure leaves no delivery score -> the arm can't sit on the x axis
        delivery_pc = {"AW-0001": {"pipeline": {"note": "no score"}}}
        assert rendering.audit_delivery_pareto_rows(delivery_pc) == []


class TestAuditChartRows:
    def test_length_rows_wide_form_keeps_missing_plain_as_none(self):
        per_case = {"AW-0002": {"pipeline": 500, "plain": 200},
                    "AW-0001": {"pipeline": 300, "plain": None}}
        rows = rendering.audit_length_chart_rows(per_case)
        # sorted by record; one row per record, one column per arm (colors are
        # pinned by column order — AUDIT_ARM_COLUMNS/AUDIT_ARM_COLORS)
        assert rows == [
            {"record": "AW-0001", "plain Claude": None, "pipeline": 300},
            {"record": "AW-0002", "plain Claude": 200, "pipeline": 500},
        ]

    def test_arm_columns_and_colors_stay_paired(self):
        assert len(rendering.AUDIT_ARM_COLUMNS) == len(rendering.AUDIT_ARM_COLORS)
        assert rendering.AUDIT_ARM_COLUMNS[0] == "plain Claude"

    def test_labels_map_records_to_example_gids_with_fallback(self):
        # per_case stays keyed by prompt_id; labels swap in the stable example
        # gid for display, and unmapped ids (pre-gid runs) pass through
        labels = {"AW-0001": "E-0042"}
        per_case = {"AW-0001": {"pipeline": 300, "plain": None},
                    "AW-0002": {"pipeline": 500, "plain": 200}}
        rows = rendering.audit_length_chart_rows(per_case, labels)
        assert [r["record"] for r in rows] == ["E-0042", "AW-0002"]
        delivery_case = {"AW-0001": {"pipeline": {"score": 7, "note": ""}}}
        assert rendering.audit_delivery_pareto_rows(
            delivery_case, labels)[0]["record"] == "E-0042"
        assert rendering.audit_record_label("AW-0009", labels) == "AW-0009"
        assert rendering.audit_record_label("AW-0009", None) == "AW-0009"


class TestAuditSectionMeta:
    # Every section title the current audit emits must resolve through the
    # title fallback (old reports carry no group/gloss fields).
    CURRENT_TITLES = [
        "Response lengths (vs plain baseline)", "Stock phrases (responses)",
    ]

    def test_field_wins_over_title_fallback(self):
        sec = {"title": "Response lengths (vs plain baseline)",
               "group": "paid", "gloss": "custom"}
        assert rendering.audit_section_group(sec) == "paid"
        assert rendering.audit_section_gloss(sec) == "custom"

    def test_title_fallback_covers_every_current_section(self):
        for title in self.CURRENT_TITLES:
            sec = {"title": title}
            assert rendering.audit_section_group(sec) in rendering.AUDIT_GROUP_ORDER[:-1], title
            assert rendering.audit_section_gloss(sec), title

    def test_unknown_sections_degrade_to_other(self):
        sec = {"title": "Some future check"}
        assert rendering.audit_section_group(sec) == "other"
        assert rendering.audit_section_gloss(sec) == ""


class TestAuditVerdictSummary:
    def test_worst_verdict_counts_and_report_order(self):
        report = {"sections": [
            {"title": "A", "group": "prompt", "rows": [
                {"verdict": "GOOD"}, {"verdict": "OK"}, {"verdict": None}]},
            {"title": "B", "group": "response", "rows": [
                {"verdict": "GOOD"}, {"verdict": "BAD"}, {"verdict": "OK"}]},
            {"title": "C", "group": "prompt", "rows": [{"verdict": None}]},
        ]}
        rows = rendering.audit_verdict_summary(report)
        assert [r["section"] for r in rows] == ["A", "B", "C"]
        assert rows[0]["worst"] == "OK" and rows[0]["counts"] == {"GOOD": 1, "OK": 1, "BAD": 0}
        assert rows[1]["worst"] == "BAD"
        assert rows[2]["worst"] is None  # purely informational section

    def test_skipped_sections_are_flagged(self):
        report = {"sections": [{"title": "A", "rows": [{"verdict": None}]}],
                  "skipped_sections": [{"section": "A", "reason": "bare-file input"}]}
        rows = rendering.audit_verdict_summary(report)
        assert rows[0]["skipped"] is True

    def test_empty_report_gives_empty_summary(self):
        assert rendering.audit_verdict_summary({}) == []


class TestAuditBatchTotals:
    def test_totals_with_absolute_and_percent_deltas(self):
        report = {
            "response_lengths": {"per_case": {
                "AW-0001": {"pipeline": 400, "plain": 200},
                "AW-0002": {"pipeline": 600, "plain": 300},
                "AW-0003": {"pipeline": 999, "plain": None},  # unpaired: excluded
            }},
        }
        rows = rendering.audit_batch_totals(report)
        assert rows == [
            {"metric": "total characters", "plain Claude": "500", "pipeline": "1,000",
             "Δ absolute": "+500", "Δ %": "+100.0%"},
        ]

    def test_empty_report_gives_no_rows(self):
        assert rendering.audit_batch_totals({}) == []


class TestComposedGateRefineRendering:
    """Composed 1c gate + 1d refine runs: BOTH stages must be renderable —
    the gate must never short-circuit the refine view (the stage split's whole
    point is that both calls are real, paid, and reviewable)."""

    @staticmethod
    def _composed_run(tmp_path):
        import random
        import shutil

        from dad_pipeline import compose_scenarios as cs

        run = tmp_path / "run"
        (run / "inputs" / "prompts").mkdir(parents=True)
        for name in ("step1c_gate.txt", "step1d_refine.txt"):
            shutil.copy(rendering.REPO_ROOT / "prompts" / "dad" / name
                        if hasattr(rendering, "REPO_ROOT")
                        else f"prompts/dad/{name}",
                        run / "inputs" / "prompts" / name)
        scenario = cs.deal_scenarios(1, random.Random(3))[0]
        scenario["scenario_description"] = "A designed situation."
        lineage = {
            "scenario": scenario,
            "gate": {"passed": True, "failures": [], "attempt": 1},
            "dilemma": {
                "scenario_id": scenario["scenario_id"],
                "user_message": "Refined final text.",
                "draft_user_message": "Judged draft text.",
                "annotation": {"visibility": "explicit", "leverage": "their personal choices"},
            },
        }
        manifest = {"manifest_version": 2, "git_commit": None}
        return run, manifest, lineage

    def test_gate_stage_renders_the_judged_draft(self, tmp_path):
        run, manifest, lineage = self._composed_run(tmp_path)
        r = rendering.render_prompt("dad", "step1_gate", run, manifest, lineage)
        assert r.is_llm_call
        # the gate judged the PRE-refine draft, not the shipped rewrite
        assert "Judged draft text." in (r.user or "")
        assert any("PASS" in w for w in r.warnings)

    def test_refine_stage_renders_even_when_the_gate_ran(self, tmp_path):
        # regression: the old single-stage view short-circuited to the gate
        # whenever gate.jsonl had a record, hiding the refine call entirely
        run, manifest, lineage = self._composed_run(tmp_path)
        r = rendering.render_prompt("dad", "step1_refine", run, manifest, lineage)
        assert r.is_llm_call
        assert "Judged draft text." in (r.user or "")     # the draft under review
        assert "<draft_prompt>" in (r.user or "")

    def test_gate_stage_marks_not_run_on_pre_gate_lineage(self, tmp_path):
        run, manifest, lineage = self._composed_run(tmp_path)
        lineage["gate"] = None
        r = rendering.render_prompt("dad", "step1_gate", run, manifest, lineage)
        assert not r.is_llm_call
        assert any("did not use the 1c gate" in w for w in r.warnings)


class TestSdfStageTemplates:
    """The SDF draft/rewrite/score branches must fetch the template each run
    ACTUALLY ran with. The layer renumber moved old 3/4/5 -> new 2/3/4 and the
    two schemes overlap (old layer3.txt is the draft template, new layer3.txt
    the rewrite one), so a hardcoded name renders the wrong prompt for half the
    runs in outputs/. These were the branches no test covered."""

    STAGES = ("draft", "rewrite", "score")

    def _snapshot(self, tmp_path, names):
        run = tmp_path / "run"
        prompts = run / "inputs" / "prompts"
        prompts.mkdir(parents=True)
        for n in names:
            (prompts / n).write_text(f"BODY OF {n}", encoding="utf-8")
        return run

    def test_pre_renumber_snapshot_keeps_the_old_names(self, tmp_path):
        run = self._snapshot(tmp_path, ["layers1-2.txt", "layer3.txt",
                                        "layer4.txt", "layer5.txt"])
        got = [rendering.sdf_stage_template(run, None, s) for s in self.STAGES]
        assert got == ["layer3.txt", "layer4.txt", "layer5.txt"]

    def test_pre_matrix_snapshot_also_keeps_the_old_names(self, tmp_path):
        # layer1..layer5 with NO layers1-2.txt — the layout utils' own marker
        # cannot see, and the reason this helper keys off layer5.txt instead.
        run = self._snapshot(tmp_path, [f"layer{i}.txt" for i in range(1, 6)])
        got = [rendering.sdf_stage_template(run, None, s) for s in self.STAGES]
        assert got == ["layer3.txt", "layer4.txt", "layer5.txt"]

    def test_post_renumber_snapshot_takes_the_new_names(self, tmp_path):
        run = self._snapshot(tmp_path, [f"layer{i}.txt" for i in range(1, 5)])
        got = [rendering.sdf_stage_template(run, None, s) for s in self.STAGES]
        assert got == ["layer2.txt", "layer3.txt", "layer4.txt"]

    def test_every_stage_maps_to_a_distinct_template_per_scheme(self):
        for scheme in (0, 1):
            names = [rendering._SDF_STAGE_TEMPLATES[s][scheme] for s in self.STAGES]
            assert len(set(names)) == len(names)

    def _lineage(self):
        return {"subtype": {"type_name": "T", "description": "D", "tone": "warm"},
                "rewrite": {"original": "ORIGINAL DOC", "rewritten": "SHIPPED DOC"}}

    def test_render_prompt_reads_the_run_s_own_draft_template(self, tmp_path):
        """End to end through the public entry point, both schemes: the body
        that reaches r.user must come from the file that run really used."""
        for i, (names, expected) in enumerate((
            (["layers1-2.txt", "layer3.txt", "layer4.txt", "layer5.txt"],
             {"draft": "layer3.txt", "rewrite": "layer4.txt", "score": "layer5.txt"}),
            ([f"layer{n}.txt" for n in range(1, 5)],
             {"draft": "layer2.txt", "rewrite": "layer3.txt", "score": "layer4.txt"}),
        )):
            run = self._snapshot(tmp_path / f"scheme{i}", names)
            manifest = {"manifest_version": 3, "config": {}}
            for stage, want in expected.items():
                r = rendering.render_prompt("sdf", stage, run, manifest, self._lineage())
                assert f"BODY OF {want}" in (r.user or ""), (stage, want, names)
                assert want in [t.name for t in r.template_sources]
