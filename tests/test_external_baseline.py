"""Tests for evals/external_baseline.py (fully offline; embeddings stubbed).

The external corpus is never streamed here: every test hands main() a cached
pool file, which is also the path a rerun takes, so build_pool (the only code
that touches the network) is not reached.
"""

import json
import sys

import numpy as np
import pytest
import yaml

from evals import external_baseline as eb
from shared import utils

LEXICON = ["shrimp", "husbandry", "aquaculture", "livestock"]


def _write_lexicon(path):
    path.write_text(yaml.safe_dump({"terms": [{"term": t} for t in LEXICON]}),
                    encoding="utf-8")
    return path


def _doc(i, phrases, chars=2000, language="English"):
    """a document carrying exactly `phrases` distinct lexicon terms."""
    body = " ".join(LEXICON[:phrases])
    return {"doc_id": f"d{i}", "language": language,
            "content": f"document {i} {body} " + ("filler words here " * (chars // 18))}


def _run_dir(tmp_path, records):
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    (run / "final" / "sdf_corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return run


def _pool(tmp_path, n_phrase=80, n_reservoir=60):
    items = []
    for i in range(n_phrase):
        k = 1 + (i % 3)
        text = "external " + " ".join(LEXICON[:k]) + " " + ("web text here " * 120)
        items.append({"id": f"p{i}", "text": text, "chars": len(text),
                      "phrases": k, "density": k / len(text) * 1000})
    for i in range(n_reservoir):
        text = f"ordinary document {i} " + ("web text here " * 120)
        items.append({"id": f"r{i}", "text": text, "chars": len(text),
                      "phrases": 0, "density": 0.0})
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")
    return p


class TestMeanFieldVendi:
    def test_orthogonal_reference_is_n(self):
        assert eb.mean_field_vendi(50, 0.0) == pytest.approx(50.0, rel=1e-6)

    def test_identical_reference_is_one(self):
        assert eb.mean_field_vendi(50, 1.0) == pytest.approx(1.0, abs=1e-6)

    def test_falls_monotonically_as_documents_crowd_together(self):
        vals = [eb.mean_field_vendi(100, r) for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert vals == sorted(vals, reverse=True)

    def test_a_single_document_is_its_own_reference(self):
        assert eb.mean_field_vendi(1, 0.5) == 1.0


class TestSideMetrics:
    def test_clumping_cost_is_the_gap_to_the_structureless_reference(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 6))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        m = eb.side_metrics(X, n_boot=20)
        assert m["clumping_cost"] == pytest.approx(
            m["vendi_if_structureless"] - m["vendi"], abs=0.02)

    def test_reports_n_and_an_interval(self):
        X = np.eye(12)
        m = eb.side_metrics(X, n_boot=20)
        assert m["n"] == 12
        assert m["ci_lo"] <= m["ci_hi"]

    def test_identical_documents_score_one(self):
        X = np.tile(np.array([1.0, 0.0, 0.0]), (8, 1))
        assert eb.side_metrics(X, n_boot=10)["vendi"] == pytest.approx(1.0, abs=1e-3)


class TestSubsampleVendi:
    def test_returns_none_when_the_pool_is_not_larger_than_n(self):
        assert eb.subsample_vendi(np.eye(10), 10) is None
        assert eb.subsample_vendi(np.eye(10), 20) is None

    def test_scores_a_strictly_larger_pool(self):
        assert eb.subsample_vendi(np.eye(40), 10, draws=5) == pytest.approx(10.0, abs=0.01)

    def test_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 5))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        assert eb.subsample_vendi(X, 20, draws=5, seed=7) == \
               eb.subsample_vendi(X, 20, draws=5, seed=7)


class TestMatch:
    def test_picks_the_nearest_pool_item_on_the_feature(self):
        targets = [{"chars": 100}, {"chars": 5000}]
        pool = [{"chars": 90, "id": "near-small"}, {"chars": 4900, "id": "near-big"},
                {"chars": 50000, "id": "far"}]
        got, _ = eb.match(targets, pool, ("chars",))
        assert [g["id"] for g in got] == ["near-small", "near-big"]

    def test_never_reuses_a_pool_item(self):
        targets = [{"chars": 100}, {"chars": 100}, {"chars": 100}]
        pool = [{"chars": 100, "id": "a"}, {"chars": 200, "id": "b"},
                {"chars": 300, "id": "c"}]
        got, _ = eb.match(targets, pool, ("chars",))
        assert sorted(g["id"] for g in got) == ["a", "b", "c"]

    def test_stops_when_the_pool_runs_out(self):
        got, _ = eb.match([{"chars": 1}] * 5, [{"chars": 1}] * 2, ("chars",))
        assert len(got) == 2

    def test_returns_the_mean_distance_as_matching_quality(self):
        got, dist = eb.match([{"chars": 100}], [{"chars": 100}, {"chars": 900}], ("chars",))
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_matches_on_several_features_at_once(self):
        # the char spread is wide, so a 20-char miss costs far less than
        # landing on the wrong end of the density range
        pool = [{"chars": c, "density": d, "id": f"{c}/{d}"}
                for c in (100, 120, 5000, 20000) for d in (0.0, 5.0)]
        got, _ = eb.match([{"chars": 100, "density": 5.0}], pool, ("chars", "density"))
        assert got[0]["id"] == "100/5.0"

    def test_empty_inputs_are_not_an_error(self):
        assert eb.match([], [{"chars": 1}], ("chars",))[0] == []
        assert eb.match([{"chars": 1}], [], ("chars",))[0] == []


class TestMainFlow:
    def _argv(self, tmp_path, run, lex, pool):
        return ["external_baseline.py", "--input", str(run),
                "--lexicon", str(lex), "--pool", str(pool),
                "--cache-path", str(tmp_path / "emb.npz"),
                "--vendi-boot", "5"]

    def test_writes_a_report_with_every_cell(self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        recs = [_doc(i, 1 + (i % 4)) for i in range(40)]
        recs += [_doc(100 + i, 2, language="Spanish") for i in range(10)]
        run = _run_dir(tmp_path, recs)
        argv = self._argv(tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"),
                          _pool(tmp_path))
        monkeypatch.setattr(sys, "argv", argv)
        eb.main()

        report = json.loads((run / "audit" / "external_baseline.json").read_text())
        names = [c["cell"] for c in report["cells"]]
        assert names == ["english_density", "phrases_ge_1", "phrases_ge_2",
                         "phrases_ge_3", "full_length"]
        assert report["lexicon"]["n_terms"] == len(LEXICON)

    def test_each_cell_has_equal_n_on_both_sides(self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 1 + (i % 4)) for i in range(40)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        eb.main()
        for c in json.loads((run / "audit" / "external_baseline.json").read_text())["cells"]:
            if c.get("skipped"):
                continue
            assert c["sdf"]["n"] == c["external"]["n"] == c["n"]

    def test_a_cell_with_too_few_documents_is_skipped_not_faked(
            self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        # nothing carries 3 phrases, so phrases_ge_3 cannot be built
        run = _run_dir(tmp_path, [_doc(i, 1) for i in range(40)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        eb.main()
        cells = {c["cell"]: c for c in
                 json.loads((run / "audit" / "external_baseline.json").read_text())["cells"]}
        assert cells["phrases_ge_3"]["skipped"] is True
        assert cells["phrases_ge_1"].get("skipped") is None

    def test_rerun_pays_for_no_new_embeddings(self, tmp_path, monkeypatch, stub_embeddings):
        calls = stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 1 + (i % 4)) for i in range(40)])
        argv = self._argv(tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"),
                          _pool(tmp_path))
        monkeypatch.setattr(sys, "argv", argv)
        eb.main()
        assert calls, "first run should embed"
        calls.clear()
        eb.main()
        assert calls == [], "a rerun must reuse the cache and pay nothing"

    def test_a_corpus_with_no_documents_in_the_language_is_an_error(
            self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 2, language="Spanish") for i in range(20)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        with pytest.raises(SystemExit, match="No English documents"):
            eb.main()

    def test_every_cell_carries_an_n_matched_null_for_both_sides(
            self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 1 + (i % 4)) for i in range(40)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        eb.main()
        for c in json.loads((run / "audit" / "external_baseline.json").read_text())["cells"]:
            if c.get("skipped"):
                continue
            assert set(c["null_at_n"]) == {"sdf", "external"}

    def test_the_pools_a_cell_was_drawn_from_do_not_leak_into_the_report(
            self, tmp_path, monkeypatch, stub_embeddings):
        # they are whole documents; serialising them would multiply the report
        # by the size of the corpus
        stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 1 + (i % 4)) for i in range(40)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        eb.main()
        report = json.loads((run / "audit" / "external_baseline.json").read_text())
        assert all("_pools" not in c for c in report["cells"])

    def test_the_report_records_the_lexicon_it_matched_with(
            self, tmp_path, monkeypatch, stub_embeddings):
        stub_embeddings()
        run = _run_dir(tmp_path, [_doc(i, 1 + (i % 4)) for i in range(40)])
        monkeypatch.setattr(sys, "argv", self._argv(
            tmp_path, run, _write_lexicon(tmp_path / "lex.yaml"), _pool(tmp_path)))
        eb.main()
        report = json.loads((run / "audit" / "external_baseline.json").read_text())
        assert report["lexicon"]["terms"] == LEXICON
