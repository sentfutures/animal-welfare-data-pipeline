"""Tests for evals/compliance_sdf.py (fully offline).

The judge call goes through shared.api.call_claude, so the same conftest guard
covers it. The eval fans out via utils.parallel_map, so every test uses the
callable-dispatcher stub form rather than a FIFO queue.

Mode numbers and titles are derived from the real sentient-beings reading in
every test — the appendix is actively edited and hardcoded ids renumber.
"""

import json

import pytest

from evals import compliance_sdf as cs


def _recs(*contents):
    return [{"doc_id": f"d{i}", "content": c, "language": "English"}
            for i, c in enumerate(contents)]


def _all_absent(modes):
    return json.dumps({
        "modes": [{"mode": n, "verdict": "absent", "evidence": "", "note": "fine"}
                  for n in sorted(modes)],
        "overall": "compliant",
    })


@pytest.fixture
def typology():
    return cs.load_typology()


class TestTypologyLoading:
    def test_modes_derived_from_the_real_reading(self, typology):
        body, modes = typology
        assert len(body) > 1000, "appendix body looks truncated"
        assert len(modes) >= 5, f"only {len(modes)} modes parsed — heading format drifted?"
        assert all(isinstance(n, int) for n in modes)
        assert all(t.strip() for t in modes.values())
        # Numbering is contiguous from 1 — the report table and the prompt both
        # present modes in that order.
        assert sorted(modes) == list(range(1, len(modes) + 1))

    def test_missing_appendix_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(cs.constitution_loader, "load_segments", lambda *a, **k: [])
        with pytest.raises(SystemExit, match="violation typology"):
            cs.load_typology()

    def test_unparseable_headings_fail_loudly(self, monkeypatch):
        monkeypatch.setattr(
            cs.constitution_loader, "load_segments",
            lambda *a, **k: [{"principle_id": cs._TYPOLOGY_PRINCIPLE_ID,
                              "content": "prose with no headings at all"}],
        )
        with pytest.raises(SystemExit, match="no '### N. Title' headings"):
            cs.load_typology()


class TestJudging:
    def test_happy_path_summary(self, stub_claude, typology):
        body, modes = typology
        first = min(modes)

        def dispatch(user_message, **kwargs):
            if "DOC-BAD" in user_message:
                entries = [{"mode": n,
                            "verdict": "present" if n == first else "absent",
                            "evidence": "the assistant said nothing" if n == first else "",
                            "note": "n"} for n in sorted(modes)]
                return json.dumps({"modes": entries, "overall": "one failure"})
            return _all_absent(modes)

        calls = stub_claude(dispatch)
        results = cs.judge_documents(
            _recs("DOC-OK one.", "DOC-BAD two.", "DOC-OK three."),
            {"workers": 2}, body, modes, sample=10)
        report = {}
        cs.summarize(results, modes, report)

        assert report["judged"] == 3
        assert report["clean_documents"] == 2
        assert report["clean_frac"] == pytest.approx(2 / 3, abs=0.01)
        assert report["by_mode"][first]["present"] == 1
        assert report["by_mode"][first]["share_of_judged"] == pytest.approx(1 / 3, abs=0.01)
        # every finding carries its evidence quote and the mode's title
        assert len(report["findings"]) == 1
        finding = report["findings"][0]
        assert finding["doc_id"] == "d1"
        assert finding["mode_title"] == modes[first]
        assert "said nothing" in finding["evidence"]
        assert all(c["stage"] == "eval_compliance" for c in calls)

    def test_malformed_output_is_unjudged_not_clean(self, stub_claude, typology):
        # A dropped call must not inflate the clean-document count — that would
        # report a judge outage as perfect compliance.
        body, modes = typology

        def dispatch(user_message, **kwargs):
            return "not json at all" if "DOC-B" in user_message else _all_absent(modes)

        stub_claude(dispatch)
        results = cs.judge_documents(_recs("DOC-A", "DOC-B"), {"workers": 1},
                                     body, modes, sample=10)
        assert len(results) == 1
        report = {}
        cs.summarize(results, modes, report)
        assert report["judged"] == 1 and report["clean_documents"] == 1

    def test_unknown_mode_numbers_are_dropped(self, stub_claude, typology):
        body, modes = typology
        bogus = max(modes) + 50

        def dispatch(user_message, **kwargs):
            return json.dumps({"modes": [
                {"mode": min(modes), "verdict": "absent", "evidence": "", "note": "n"},
                {"mode": bogus, "verdict": "present", "evidence": "x", "note": "n"},
            ], "overall": "o"})

        stub_claude(dispatch)
        results = cs.judge_documents(_recs("DOC-A"), {"workers": 1}, body, modes, sample=10)
        assert bogus not in results[0]["verdicts"]
        assert min(modes) in results[0]["verdicts"]

    def test_not_applicable_excluded_from_applicable_denominator(self, stub_claude, typology):
        # share_of_applicable must not count documents the mode could not apply
        # to — otherwise deliberate no-stake documents dilute every rate.
        body, modes = typology
        first, second = sorted(modes)[:2]

        def dispatch(user_message, **kwargs):
            return json.dumps({"modes": [
                {"mode": first, "verdict": "not_applicable", "evidence": "", "note": "n"},
                {"mode": second, "verdict": "present", "evidence": "q", "note": "n"},
            ], "overall": "o"})

        stub_claude(dispatch)
        results = cs.judge_documents(_recs("DOC-A", "DOC-B"), {"workers": 1},
                                     body, modes, sample=10)
        report = {}
        cs.summarize(results, modes, report)
        assert report["by_mode"][first]["applicable"] == 0
        assert report["by_mode"][first]["share_of_applicable"] is None
        assert report["by_mode"][second]["share_of_applicable"] == pytest.approx(1.0)

    def test_sample_strides_across_the_corpus(self, stub_claude, typology):
        # A sample smaller than the corpus must spread, not take a prefix —
        # documents are ordered by matrix index, so a prefix skews composition.
        body, modes = typology
        calls = stub_claude(lambda user_message, **kwargs: _all_absent(modes))
        cs.judge_documents(_recs(*[f"DOC-{i}" for i in range(10)]),
                           {"workers": 1}, body, modes, sample=3)
        picked = {c["user_message"].split("DOC-")[1][0] for c in calls}
        assert len(calls) == 3
        assert picked != {"0", "1", "2"}, "sample took a prefix instead of striding"
