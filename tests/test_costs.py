"""Reading a cost log back: what a run spent, and what full scale would cost.

The projection is the part worth testing. Layer 3 sends a ~48k-token cached
system prompt, so a 2-document smoke run spends most of its money writing a
cache it then amortises. Scaling that run's per-document average would
overstate a 150-document run by several times, which is exactly the number
someone would use to decide whether to run it.
"""

import json

import pytest

from shared import costs


def write_log(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def call(stage="layer3_rewrite", item="d0", usd=0.05, cache_write=0, cache_read=0,
         model="claude-sonnet-5", attempts=1):
    return {"model": model, "backend": "api", "stage": stage, "item_id": item,
            "input_tokens": 2500, "output_tokens": 2700,
            "cache_creation_tokens": cache_write, "cache_read_tokens": cache_read,
            "attempts": attempts, "duration_s": 30.0, "cost_usd": usd}


class TestSummarize:
    def test_a_missing_log_is_zero_not_an_error(self, tmp_path):
        s = costs.summarize(tmp_path / "nope.jsonl")
        assert s["totals"]["calls"] == 0 and s["totals"]["usd"] == 0.0

    def test_totals_stages_and_models(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [
            call(stage="layer3_rewrite", item="d0", usd=0.20),
            call(stage="layer3_rewrite", item="d1", usd=0.10),
            call(stage="layer4_score", item="d0", usd=0.05, model="claude-opus-5"),
        ])
        s = costs.summarize(log)
        assert s["totals"]["calls"] == 3
        assert s["totals"]["usd"] == pytest.approx(0.35)
        assert s["by_stage"]["layer3_rewrite"]["usd"] == pytest.approx(0.30)
        assert s["by_stage"]["layer3_rewrite"]["items"] == 2
        # stages are ordered most expensive first
        assert list(s["by_stage"]) == ["layer3_rewrite", "layer4_score"]
        assert set(s["by_model"]) == {"claude-sonnet-5", "claude-opus-5"}

    def test_retries_are_counted_because_they_are_billed(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [call(attempts=3), call(attempts=1)])
        assert costs.summarize(log)["totals"]["retries"] == 2

    def test_a_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(json.dumps(call(usd=0.10)) + "\nnot json\n", encoding="utf-8")
        assert costs.summarize(path)["totals"]["usd"] == pytest.approx(0.10)

    def test_untagged_records_are_labelled_rather_than_dropped(self, tmp_path):
        rec = call()
        del rec["stage"]
        log = write_log(tmp_path / "c.jsonl", [rec])
        assert "(untagged)" in costs.summarize(log)["by_stage"]


class TestSinceMark:
    """The global cost log is shared; a tool must report only its own calls."""

    def test_mark_then_summarize_covers_only_the_new_records(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [call(usd=9.99), call(usd=9.99)])
        start = costs.mark(log)
        assert start == 2

        existing = log.read_text(encoding="utf-8")
        log.write_text(existing + json.dumps(call(usd=0.25)) + "\n", encoding="utf-8")
        s = costs.summarize(log, since=start)
        assert s["totals"]["calls"] == 1
        assert s["totals"]["usd"] == pytest.approx(0.25)

    def test_mark_on_a_missing_log_is_zero(self, tmp_path):
        assert costs.mark(tmp_path / "nope.jsonl") == 0


class TestProjection:
    def test_one_time_cache_writes_are_not_scaled(self, tmp_path):
        # 2 documents: the first writes a 47,600-token cache, the second reads it.
        log = write_log(tmp_path / "c.jsonl", [
            call(item="d0", usd=0.1510, cache_write=47600),
            call(item="d1", usd=0.0415, cache_read=47600),
        ])
        s = costs.summarize(log)
        cache_write = s["totals"]["cache_write_usd"]
        assert cache_write == pytest.approx(47600 / 1e6 * 2.0 * 1.25, rel=1e-6)

        projected = costs.project(s, n_done=2, n_target=150)
        naive = s["totals"]["usd"] / 2 * 150
        # the honest projection is far below the naive one, which is the point
        assert projected < naive / 2
        variable = s["totals"]["usd"] - cache_write
        assert projected == pytest.approx(cache_write + variable / 2 * 150)

    def test_a_log_with_no_caching_projects_linearly(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [call(item="d0", usd=0.10),
                                               call(item="d1", usd=0.10)])
        s = costs.summarize(log)
        assert costs.project(s, n_done=2, n_target=100) == pytest.approx(10.0)

    def test_nothing_to_project_from_returns_none(self, tmp_path):
        s = costs.summarize(tmp_path / "nope.jsonl")
        assert costs.project(s, n_done=0, n_target=10) is None
        assert costs.project(s, n_done=5, n_target=10) is None

    def test_an_unpriced_model_contributes_no_phantom_cache_cost(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl",
                        [call(model="some-future-model", usd=0.10, cache_write=50000)])
        assert costs.summarize(log)["totals"]["cache_write_usd"] == 0.0


class TestFormatting:
    def test_an_empty_log_says_nothing_was_billed(self, tmp_path):
        text = costs.format_summary(costs.summarize(tmp_path / "nope.jsonl"))
        assert "no API calls" in text

    def test_the_total_and_per_item_cost_are_both_shown(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [call(item="d0", usd=0.20),
                                               call(item="d1", usd=0.20)])
        text = costs.format_summary(costs.summarize(log), n_done=2)
        assert "$0.4000" in text and "$0.2000 each" in text

    def test_the_projection_is_labelled_as_not_a_simple_ratio(self, tmp_path):
        log = write_log(tmp_path / "c.jsonl", [call(item="d0", usd=0.25, cache_write=47600)])
        text = costs.format_summary(costs.summarize(log), n_done=1, project_to=150)
        assert "PROJECTED at 150" in text
        assert "ONE-TIME" in text
        assert "not simply the figure above times a ratio" in text

    def test_report_prints_and_returns(self, tmp_path, capsys):
        log = write_log(tmp_path / "c.jsonl", [call(usd=0.30)])
        s = costs.report(log, title="Smoke cost", n_done=1)
        assert s["totals"]["usd"] == pytest.approx(0.30)
        assert "SMOKE COST: $0.3000" in capsys.readouterr().out
