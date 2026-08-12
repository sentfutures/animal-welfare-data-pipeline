"""Tests for the mechanical checks in evals/audit_sdf.py (fully offline).

Only the pure-ish audit functions are tested (records in, report dict out);
the LLM pattern pass goes through shared.api.call_claude and is guarded by
the same conftest safety net as everything else.
"""

import json

from evals import audit_sdf


def _recs(*contents):
    return [{"doc_id": f"d{i}", "content": c, "language": "en"} for i, c in enumerate(contents)]


class TestMarkdownMetric:
    def test_counts_each_markdown_class(self):
        report = {}
        audit_sdf.audit_markdown(_recs(
            "# Heading\nplain prose follows here.",
            "Prose with **bold emphasis** inside.",
            "- a markdown bullet\n- another",
            "| a | b |\n| c | d |",
            "Entirely plain prose, no markup at all.",
        ), report)
        md = report["markdown"]
        assert md["# headings"] == 0.2
        assert md["**bold**"] == 0.2
        assert md["markdown bullets"] == 0.2
        assert md["tables"] == 0.2
        assert md["any_frac"] == 0.8

    def test_clean_corpus_scores_zero(self):
        report = {}
        audit_sdf.audit_markdown(_recs(
            "Plain running prose. Nothing fancy here at all.",
            "1. A plain-text enumeration is not markdown.\n2. Second item.",
        ), report)
        assert report["markdown"]["any_frac"] == 0.0

    def test_mid_text_dashes_are_not_bullets(self):
        report = {}
        audit_sdf.audit_markdown(_recs("A clause — set off with dashes — is fine.\nSo is a hy-phen."), report)
        assert report["markdown"]["markdown bullets"] == 0.0


class TestParseJsonBlock:
    def test_plain_and_fenced(self):
        assert audit_sdf._parse_json_block('[1, 2]') == [1, 2]
        assert audit_sdf._parse_json_block('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_first_block_from_trailing_prose(self):
        # the "Extra data" failure seen live: valid JSON followed by more text
        raw = '[{"pattern": "x"}]\n\nAdditionally, I noticed...'
        assert audit_sdf._parse_json_block(raw) == [{"pattern": "x"}]

    def test_recovers_json_after_leading_prose(self):
        raw = 'Here are the patterns:\n[{"pattern": "y"}]'
        assert audit_sdf._parse_json_block(raw) == [{"pattern": "y"}]


class TestTruncationAudit:
    def test_separates_mid_sentence_from_trailing_separator(self):
        report = {}
        audit_sdf.audit_length_truncation(_recs(
            "Complete sentence.",
            # A truncation tell is a full line of prose stopping dead; a short
            # unpunctuated fragment no longer counts (see ends_mid_sentence).
            "The inspector noted that the stocking density exceeded the "
            "recommended threshold by a substantial",
            "Ends fine but with a rule.\n\n---",
        ), report)
        assert report["length"]["truncated"] == 1
        assert report["length"]["trailing_separator"] == 1

    def test_signoff_and_see_also_endings_are_not_truncation(self):
        # These shapes made up all 225 flags on the committed corpora.
        report = {}
        audit_sdf.audit_length_truncation(_recs(
            "The follow-up visit is booked.\n\n— Michelle",
            "Coverage remains partial.\n\nSee also\nMachine ethics",
            "请通过合法渠道解决。\n\n清河区市场监督管理局\n消费者权益保护科",
        ), report)
        assert report["length"]["truncated"] == 0

    def test_long_signoff_and_see_also_rows_are_not_truncation(self):
        # The shapes the length-only rule still got wrong: a sign-off or a
        # cross-reference row over 12 words / 80 characters. Six of the nine
        # documents the pinned SDF run flagged were these, so the row reported
        # 1.9% against a corpus with no truncation in it at all.
        report = {}
        audit_sdf.audit_length_truncation(_recs(
            "Thank you for the referral.\n\nDr. Amara Okonkwo | Senior Veterinary "
            "Officer | National Animal Welfare Board",
            "Coverage of the incident remains partial.\n\nSee also: Digital minds and "
            "moral status; AI welfare protocols; Duty cycle standards in "
            "software-based labor; Precautionary reasoning under moral uncertainty",
            "Marking this resolved per the template. This one is about the ducks. "
            "— user:mod_greenhollow, 27 May",
        ), report)
        assert report["length"]["truncated"] == 0

    def test_a_real_truncation_is_still_counted(self):
        # The other half: the exemptions must not empty the row out. This tail
        # is cut mid-word and carries an em dash and commas, the punctuation the
        # sign-off exemptions key on.
        report = {}
        audit_sdf.audit_length_truncation(_recs(
            "Complete sentence.",
            "Exhibit lighting 0600 to 0100 is a 19-hour day with no dark period, "
            "handling blocks every ninety minutes with no scheduled rest, in "
            "enclosures at the template minimum — which means most of four to six thous",
        ), report)
        assert report["length"]["truncated"] == 1


class TestRegisterMetric:
    def test_english_full_name_docs_are_scored(self):
        # Regression: audit_register matched only language == "en", but final
        # corpora label English docs "English" (derive_language), so the proxy
        # silently skipped every production doc and never populated the report.
        report = {}
        audit_sdf.audit_register(
            [{"doc_id": "d0", "language": "English",
              "content": "I'm sure I've said it before, but I can't help it — it's my thing."}],
            report)
        assert "register" in report
        assert report["register"]["reads_personal_frac"] == 1.0

    def test_en_code_docs_still_scored(self):
        report = {}
        audit_sdf.audit_register(
            [{"doc_id": "d0", "language": "en",
              "content": "The committee reviewed the report and issued its findings."}],
            report)
        assert "register" in report  # "en" code still accepted (legacy/test records)


class TestCompositionMetric:
    @staticmethod
    def _rec(doc_id, tone, domain, type_name):
        return {"doc_id": doc_id, "content": "x", "language": "English",
                "type_name": type_name,
                "variables": {"tone": tone, "domain": domain,
                              "centrality": "the central subject of the document"}}

    def test_reads_axes_from_matrix_variables_not_unknown(self):
        # Regression: composition read role/tone via a layer-1 type map the matrix
        # pipeline never writes, so every axis printed 100% "unknown". It must read
        # tone/domain/centrality from each record's `variables`, and the document
        # type from the top-level type_name.
        report = {}
        audit_sdf.audit_composition([
            self._rec("d0", "neutral or journalistic", "agriculture and food production", "a news article"),
            self._rec("d1", "skeptical or displeased", "agriculture and food production", "a blog post"),
        ], report)
        comp = report["composition"]
        assert comp["tone"] == {"neutral or journalistic": 1, "skeptical or displeased": 1}
        assert comp["domain"] == {"agriculture and food production": 2}
        assert "unknown" not in comp["tone"] and "unknown" not in comp["domain"]
        assert comp["n_types"] == 2  # counted from type_name, not a stale type map


class TestPrincipleCoverage:
    """Coverage judge: per-doc calls through call_claude (parallel_map fan-out,
    so tests use the callable-dispatcher stub form), principle ids derived from
    load_principles() — never hardcoded (the CSV renumbers under editing)."""

    def _numbers(self):
        from shared import constitution_loader
        return [int(p["number"]) for p in constitution_loader.load_principles()]

    def test_happy_path_counts_and_report(self, stub_claude):
        numbers = self._numbers()
        first, second = numbers[0], numbers[1]

        def dispatch(user_message, **kwargs):
            if "DOC-A" in user_message:
                return json.dumps([first, second])
            return json.dumps([first])

        calls = stub_claude(dispatch)
        report = {}
        audit_sdf.audit_principle_coverage(
            _recs("DOC-A text.", "DOC-B text."), {"workers": 2}, report, sample=10)
        cov = report["principle_coverage"]
        assert cov["rated"] == 2
        assert cov["by_principle"][first] == 1.0
        assert cov["by_principle"][second] == 0.5
        # every unmentioned principle is reported (at zero) and flagged starved
        assert set(cov["by_principle"]) == set(numbers)
        assert set(cov["starved"]) == set(numbers) - {first, second}
        # the rendered principles block and the document reached the judge
        assert all(k["stage"] == "eval_audit_sdf" for k in calls)
        assert "DOC-A" in calls[0]["user_message"] or "DOC-A" in calls[1]["user_message"]

    def test_malformed_and_out_of_range_responses(self, stub_claude):
        numbers = self._numbers()
        bogus = max(numbers) + 50

        def dispatch(user_message, **kwargs):
            if "DOC-A" in user_message:
                return "no json here at all"          # unrated, not zero
            return json.dumps([numbers[0], bogus])    # bogus id dropped

        stub_claude(dispatch)
        report = {}
        audit_sdf.audit_principle_coverage(
            _recs("DOC-A text.", "DOC-B text."), {"workers": 1}, report, sample=10)
        cov = report["principle_coverage"]
        assert cov["rated"] == 1
        assert cov["by_principle"][numbers[0]] == 1.0
        assert bogus not in cov["by_principle"]

    def test_all_calls_failing_skips_cleanly(self, stub_claude):
        stub_claude(lambda user_message, **kwargs: "not json")
        report = {}
        audit_sdf.audit_principle_coverage(
            _recs("DOC-A text."), {"workers": 1}, report, sample=10)
        assert report["principle_coverage"] == {"rated": 0}
