"""Tests for evals/diversity.py (fully offline; embeddings stubbed).

Metric functions are tested pure (matrix in, number out) against geometry with
known answers; the CLI money paths (happy path, empty-record fallback, cache
reuse = zero paid calls on rerun) run through main() with stub_embeddings.
"""

import json
import sys

import numpy as np
import pytest

from evals import diversity
from shared import utils


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestMetrics:
    def test_orthogonal_docs_are_maximally_diverse(self):
        X = np.eye(4, dtype=np.float32)
        assert diversity.mean_pairwise_cosine(X) == pytest.approx(0.0, abs=1e-6)
        assert diversity.vendi_score(X) == pytest.approx(4.0, abs=1e-4)
        sims, _ = diversity.nearest_neighbors(X)
        assert sims.max() == pytest.approx(0.0, abs=1e-6)

    def test_identical_docs_collapse_to_one(self):
        X = np.tile(_unit([1.0, 2.0, 2.0]), (3, 1))
        assert diversity.mean_pairwise_cosine(X) == pytest.approx(1.0, abs=1e-5)
        assert diversity.vendi_score(X) == pytest.approx(1.0, abs=1e-4)
        sims, idx = diversity.nearest_neighbors(X)
        assert np.allclose(sims, 1.0, atol=1e-5)
        assert all(int(idx[i]) != i for i in range(3))  # self-similarity masked

    def test_two_clusters_have_vendi_two(self):
        X = np.stack([_unit([1, 0, 0]), _unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 1, 0])])
        assert diversity.vendi_score(X) == pytest.approx(2.0, abs=1e-4)

    def test_mean_pairwise_matches_brute_force(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((10, 5)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        brute = np.mean([X[i] @ X[j] for i in range(10) for j in range(10) if i != j])
        assert diversity.mean_pairwise_cosine(X) == pytest.approx(float(brute), abs=1e-5)

    def test_nearest_neighbors_blockwise_matches_brute_force(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((5, 3)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sims, idx = diversity.nearest_neighbors(X, block=2)  # block < n exercises seams
        S = X @ X.T
        np.fill_diagonal(S, -2.0)
        assert np.allclose(sims, S.max(axis=1), atol=1e-5)
        assert list(idx) == list(S.argmax(axis=1))

    def test_degenerate_sizes(self):
        one = np.ones((1, 4), dtype=np.float32) / 2
        assert diversity.mean_pairwise_cosine(one) == 0.0
        assert diversity.vendi_score(one) == 1.0
        sims, _ = diversity.nearest_neighbors(one)
        assert list(sims) == [0.0]


class TestRecordAccess:
    def test_sdf_record_uses_content(self):
        assert diversity.record_text({"content": " a doc "}) == "a doc"

    def test_dad_record_joins_messages(self):
        rec = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        assert diversity.record_text(rec) == "q\n\na"

    def test_unknown_shape_is_empty(self):
        assert diversity.record_text({"other": "x"}) == ""
        assert diversity.record_text({"content": "   "}) == ""

    def test_record_id_prefers_doc_id_then_record_id(self):
        assert diversity.record_id({"doc_id": "d", "record_id": "r"}, 3) == "d"
        assert diversity.record_id({"record_id": "r"}, 3) == "r"
        assert diversity.record_id({}, 3) == "row3"


class TestStrideSample:
    def test_under_cap_returns_all(self):
        assert diversity.stride_sample([1, 2, 3], 5) == [1, 2, 3]

    def test_over_cap_is_deterministic_ordered_subset(self):
        items = list(range(100))
        a = diversity.stride_sample(items, 10)
        assert a == diversity.stride_sample(items, 10)
        assert len(a) == 10
        assert a == sorted(a)
        assert set(a) <= set(items)


class TestTopPairs:
    def test_pair_reported_once_with_snippets(self):
        sims = np.array([0.99, 0.99, 0.10], dtype=np.float32)
        idx = np.array([1, 0, 0])
        ids = ["d0", "d1", "d2"]
        texts = ["same   doc\ntext here", "same doc text here", "different"]
        pairs = diversity.top_pairs(sims, idx, ids, texts, limit=10, floor=0.8)
        assert len(pairs) == 1  # d0~d1 and d1~d0 deduplicate; d2 is below floor
        assert {pairs[0]["a"], pairs[0]["b"]} == {"d0", "d1"}
        assert pairs[0]["a_snippet"] == "same doc text here"  # whitespace collapsed


class TestGroups:
    def test_intra_and_centroid_cosine(self):
        recs = [{"type_id": 0, "type_name": "memos"}, {"type_id": 0, "type_name": "memos"},
                {"type_id": 1}, {"type_id": 1}]
        X = np.stack([_unit([1, 0, 0]), _unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 1, 0])])
        groups = diversity.group_breakdown(recs, X)
        assert [g["n"] for g in groups] == [2, 2]
        assert groups[0]["type_name"] == "memos"  # from the record's own type_name
        assert groups[1]["type_name"] == "1"      # fallback to str(type_id) when absent
        assert groups[0]["intra_mean_cosine"] == pytest.approx(1.0, abs=1e-4)
        assert diversity.centroid_mean_cosine(recs, X) == pytest.approx(0.0, abs=1e-6)

    def test_absent_type_ids_mean_no_breakdown(self):
        recs = [{"record_id": "r0"}, {"record_id": "r1"}]
        X = np.eye(2, dtype=np.float32)
        assert diversity.group_breakdown(recs, X) is None
        assert diversity.centroid_mean_cosine(recs, X) is None

    def test_degenerate_centroid_is_dropped_not_miscomputed(self):
        # type 0's two unit docs are antipodal -> a ~zero centroid. It must be
        # dropped (a zero row would break mean_pairwise_cosine's unit-norm
        # assumption and yield a spurious negative), leaving < 2 real centroids.
        recs = [{"type_id": 0}, {"type_id": 0}, {"type_id": 1}]
        X = np.stack([_unit([1, 0, 0]), _unit([-1, 0, 0]), _unit([0, 1, 0])])
        assert diversity.centroid_mean_cosine(recs, X) is None

    def test_degenerate_centroid_dropped_with_two_survivors(self):
        recs = [{"type_id": 0}, {"type_id": 0}, {"type_id": 1}, {"type_id": 2}]
        X = np.stack([_unit([1, 0, 0]), _unit([-1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
        # type 0 cancels to zero and is dropped; types 1 and 2 are orthogonal
        assert diversity.centroid_mean_cosine(recs, X) == pytest.approx(0.0, abs=1e-6)


class TestCache:
    def test_second_call_reuses_everything(self, stub_embeddings, tmp_path):
        cache_path = tmp_path / "cache.npz"
        texts = ["alpha", "beta", "gamma"]

        calls = stub_embeddings(dim=32)
        X1, stats1 = diversity.embed_with_cache(texts, "m", cache_path)
        assert stats1 == {"cached": 0, "embedded": 3}
        assert len(calls) == 1

        calls2 = stub_embeddings(dim=32)
        X2, stats2 = diversity.embed_with_cache(texts, "m", cache_path)
        assert stats2 == {"cached": 3, "embedded": 0}
        assert calls2 == []  # zero paid calls for completed work
        assert np.allclose(X1, X2)

    def test_cache_is_model_keyed(self, stub_embeddings, tmp_path):
        cache_path = tmp_path / "cache.npz"
        stub_embeddings(dim=32)
        diversity.embed_with_cache(["alpha"], "model-a", cache_path)
        calls = stub_embeddings(dim=32)
        _, stats = diversity.embed_with_cache(["alpha"], "model-b", cache_path)
        assert stats == {"cached": 0, "embedded": 1}
        assert len(calls) == 1

    def test_no_cache_path_embeds_every_time(self, stub_embeddings):
        calls = stub_embeddings(dim=32)
        diversity.embed_with_cache(["alpha"], "m", None)
        diversity.embed_with_cache(["alpha"], "m", None)
        assert len(calls) == 2


def _write_corpus(path, records):
    utils.save_jsonl(records, path)
    return path


def _sdf_records():
    dup = "The exact same document text, repeated verbatim across two records."
    # type_name travels on each record (matrix schema); the group breakdown reads
    # it from there, not from a layer-1 type map.
    return [
        {"doc_id": "d0", "type_id": 0, "type_name": "committee minutes", "content": dup},
        {"doc_id": "d1", "type_id": 0, "type_name": "committee minutes", "content": dup},
        {"doc_id": "d2", "type_id": 1, "type_name": "field notes", "content": "A completely different piece about municipal composting rules."},
        {"doc_id": "d3", "type_id": 1, "type_name": "field notes", "content": "Field notes from a shorebird count on the estuary flats."},
        {"doc_id": "d4", "type_id": 0, "type_name": "committee minutes", "content": "Minutes of the aquaculture standards committee, third session."},
        {"doc_id": "d5", "type_id": 1, "type_name": "field notes", "content": ""},  # empty: skipped and reported
    ]


def _run_main(monkeypatch, corpus_path, config_path, *extra):
    monkeypatch.setattr(sys, "argv", [
        "diversity.py", "--input", str(corpus_path), "--config", str(config_path), *extra,
    ])
    diversity.main()


class TestMainEndToEnd:
    def test_happy_path_writes_report_and_flags_duplicates(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch, capsys
    ):
        corpus = _write_corpus(tmp_path / "corpus.jsonl", _sdf_records())
        stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)

        report = json.loads((tmp_path / "audit" / "diversity_report.json").read_text())
        assert report["n_records"] == 6
        assert report["n_embedded"] == 5
        assert report["n_empty"] == 1 and report["empty_ids"] == ["d5"]
        # the verbatim duplicate pair is caught semantically
        assert report["nn"]["max"] == pytest.approx(1.0, abs=1e-4)
        assert {report["top_pairs"][0]["a"], report["top_pairs"][0]["b"]} == {"d0", "d1"}
        assert report["nn"]["over_0.90"] >= 2 / 5
        assert 1.0 < report["vendi"]["score"] < 5.0
        assert [g["n"] for g in report["groups"]] == [3, 2]
        assert (tmp_path / "audit" / "embeddings_cache.npz").exists()
        assert "REDUNDANCY" in capsys.readouterr().out

    def test_rerun_makes_zero_embedding_calls(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch
    ):
        corpus = _write_corpus(tmp_path / "corpus.jsonl", _sdf_records())
        stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)

        calls = stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)
        assert calls == []  # rerun rides the cache: paid work is never re-billed

    def test_dad_corpus_shape_works(self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch):
        records = [
            {"record_id": f"r{i}", "messages": [
                {"role": "user", "content": f"question number {i} about {topic}?"},
                {"role": "assistant", "content": f"an answer regarding {topic}."},
            ]}
            for i, topic in enumerate(["lobsters", "broilers", "shrimp"])
        ]
        corpus = _write_corpus(tmp_path / "dad_corpus.jsonl", records)
        stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)
        report = json.loads((tmp_path / "audit" / "diversity_report.json").read_text())
        assert report["n_embedded"] == 3
        assert report["groups"] is None  # DAD records carry no type_id

    def test_run_dir_input_resolves_corpus_and_type_names(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch
    ):
        run_dir = tmp_path / "runs" / "2026-01-01_00-00_dev"
        _write_corpus(run_dir / "final" / "sdf_corpus.jsonl", _sdf_records())
        stub_embeddings(dim=32)
        _run_main(monkeypatch, run_dir, tiny_config_file)
        report = json.loads((run_dir / "audit" / "diversity_report.json").read_text())
        assert report["corpus"] == "sdf_corpus.jsonl"
        # names come from each record's own type_name, not a layer-1 type map
        assert report["groups"][0]["type_name"] == "committee minutes"

    def test_max_docs_caps_what_gets_embedded(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch
    ):
        records = [{"doc_id": f"d{i}", "content": f"document body {i}"} for i in range(10)]
        corpus = _write_corpus(tmp_path / "corpus.jsonl", records)
        calls = stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file, "--max-docs", "4")
        report = json.loads((tmp_path / "audit" / "diversity_report.json").read_text())
        assert report["n_embedded"] == 4
        assert sum(len(c["texts"]) for c in calls) == 4

    def test_too_few_documents_exits(self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch):
        corpus = _write_corpus(tmp_path / "corpus.jsonl", [{"doc_id": "d0", "content": "only one"}])
        stub_embeddings(dim=32)
        with pytest.raises(SystemExit, match="at least 2"):
            _run_main(monkeypatch, corpus, tiny_config_file)

    def test_missing_input_exits(self, tmp_path, tiny_config_file, monkeypatch):
        with pytest.raises(SystemExit, match="Input not found"):
            _run_main(monkeypatch, tmp_path / "nope.jsonl", tiny_config_file)


class TestCompareReports:
    def _current(self):
        return {
            "embed_model": "text-embedding-3-small",
            "nn": {"mean": 0.5, "over_0.90": 0.10},
            "mean_pairwise_cosine": 0.30,
            "vendi": {"ratio": 0.40},
        }

    def test_prints_deltas(self, tmp_path, capsys):
        prev = tmp_path / "prev.json"
        prev.write_text(json.dumps({
            "embed_model": "text-embedding-3-small",
            "nn": {"mean": 0.4, "over_0.90": 0.05},
            "mean_pairwise_cosine": 0.25,
            "vendi": {"ratio": 0.50},
        }))
        diversity.compare_reports(self._current(), str(prev))
        out = capsys.readouterr().out
        assert "COMPARE" in out
        assert "+0.100" in out  # mean nn 0.4 -> 0.5

    def test_model_mismatch_skips_comparison(self, tmp_path, capsys):
        prev = tmp_path / "prev.json"
        prev.write_text(json.dumps({"embed_model": "text-embedding-3-large"}))
        diversity.compare_reports(self._current(), str(prev))
        assert "skipped" in capsys.readouterr().out


class TestKmeansEvenness:
    def test_even_clusters_score_one(self):
        X = np.array([[1, 0]] * 10 + [[0, 1]] * 10, dtype=np.float32)
        ev = diversity.kmeans_evenness(X, k=2)
        assert ev["evenness"] == 1.0 and ev["largest_share"] == 0.5

    def test_skewed_clusters_score_below_one(self):
        X = np.array([[1, 0]] * 18 + [[0, 1]] * 2, dtype=np.float32)
        ev = diversity.kmeans_evenness(X, k=2)
        assert ev["largest_share"] == 0.9
        assert ev["evenness"] < 0.5   # H(0.9, 0.1)/ln 2 ≈ 0.469

    def test_k_defaults_to_n_over_5_capped(self):
        X = np.eye(20, dtype=np.float32)
        assert diversity.kmeans_evenness(X)["k"] == 4


class TestPcaCoords:
    def test_two_components_separate_the_blobs(self):
        X = np.zeros((30, 5), dtype=np.float32)
        X[:15, 0] = 1.0
        X[15:, 1] = 1.0
        P = diversity.pca_coords(X)
        assert P.shape == (30, 2)
        # PC1 puts the two blobs on opposite sides
        assert (P[:15, 0].mean() > 0) != (P[15:, 0].mean() > 0)


class TestIdeaLevel:
    def test_idea_text_prefers_user_message_then_content(self):
        assert diversity.idea_text({"messages": [{"content": "u"}, {"content": "a"}]}) == "u"
        assert diversity.idea_text({"content": "doc body"}) == "doc body"

    def test_summaries_via_stubbed_claude(self, stub_claude):
        calls = stub_claude(lambda um, **kw: "a farmer weighs cheaper cages against hens' welfare")
        out = diversity.idea_summaries(["r1", "r2"], ["text one", "text two"], {"workers": 1})
        assert set(out) == {"r1", "r2"}
        assert all(c["stage"] == "eval_diversity_ideas" for c in calls)
        assert all(c["item_id"] in ("r1", "r2") for c in calls)

    def test_failed_summary_dropped_not_fatal(self, stub_claude):
        def flaky(um, **kw):
            if "bad" in um:
                raise RuntimeError("boom")
            return "fine line"
        stub_claude(flaky)
        out = diversity.idea_summaries(["a", "b"], ["good text", "bad text"], {"workers": 1})
        assert set(out) == {"a"}


class TestScopes:
    def test_scope_text_splits_chat_sides(self):
        rec = {"messages": [{"content": "u"}, {"content": "a"}]}
        assert diversity.scope_text(rec, "prompts") == "u"
        assert diversity.scope_text(rec, "responses") == "a"
        assert diversity.scope_text({"content": "doc"}, "prompts") == ""

    def test_scope_id_labels_dots_by_scope(self):
        # the cloud hover must name what the dot IS: prompt gid (P-) for the
        # prompts scope, response gid (R-) for responses, example gid (E-,
        # the fallback) for combined — not E- for all three (the bug)
        rec = {"record_id": "rec1", "response_gid": "R-0009", "example_gid": "E-0009"}
        pgids = {"rec1": "P-0009"}
        assert diversity.scope_id(rec, "prompts", "E-0009", pgids) == "P-0009"
        assert diversity.scope_id(rec, "responses", "E-0009", pgids) == "R-0009"
        assert diversity.scope_id(rec, "combined", "E-0009", pgids) == "E-0009"
        # missing scope gid (non-DAD record) falls back to the example/combined id
        assert diversity.scope_id({"record_id": "x"}, "prompts", "E-1", {}) == "E-1"
        assert diversity.scope_id({"record_id": "x"}, "responses", "E-1", {}) == "E-1"

    def test_dad_prompt_gids_joins_step3_and_step1(self, tmp_path):
        (tmp_path / "step3").mkdir()
        (tmp_path / "step1").mkdir()
        utils.append_jsonl({"record_id": "rec1", "prompt_id": "AW-0001"},
                           tmp_path / "step3" / "rewrites.jsonl")
        utils.append_jsonl({"prompt_id": "AW-0001", "prompt_gid": "P-0042"},
                           tmp_path / "step1" / "dilemmas.jsonl")
        assert diversity.dad_prompt_gids(tmp_path) == {"rec1": "P-0042"}
        # missing stages / non-DAD run -> empty, no crash
        assert diversity.dad_prompt_gids(tmp_path / "nope") == {}

    def test_dad_run_reports_three_scopes_with_distributions(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch
    ):
        records = [
            {"record_id": f"r{i}", "messages": [
                {"role": "user", "content": f"question {i} about {t}?"},
                {"role": "assistant", "content": f"an answer regarding {t}."},
            ]}
            for i, t in enumerate(["lobsters", "broilers", "shrimp", "mink"])
        ]
        corpus = _write_corpus(tmp_path / "dad_corpus.jsonl", records)
        stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)
        report = json.loads((tmp_path / "audit" / "diversity_report.json").read_text())
        assert set(report["scopes"]) == {"prompts", "responses", "combined"}
        for blk in report["scopes"].values():
            assert blk["n"] == 4 and len(blk["nn_sims"]) == 4
            assert len(blk["cloud"]) == 4
            assert blk["clusters"]["sizes"] == sorted(blk["clusters"]["sizes"], reverse=True)
            # cluster detail: one entry per nonempty cluster, aligned with the
            # sorted sizes, each naming a representative member from its ids
            detail = blk["clusters"]["detail"]
            assert [d["size"] for d in detail] == blk["clusters"]["sizes"]
            assert sum(len(d["ids"]) for d in detail) == blk["n"]
            for d in detail:
                assert d["rep_id"] in d["ids"] and d["rep"]

    def test_document_corpus_gets_combined_scope_only(
        self, stub_embeddings, tmp_path, tiny_config_file, monkeypatch
    ):
        corpus = _write_corpus(tmp_path / "corpus.jsonl", _sdf_records())
        stub_embeddings(dim=32)
        _run_main(monkeypatch, corpus, tiny_config_file)
        report = json.loads((tmp_path / "audit" / "diversity_report.json").read_text())
        assert set(report["scopes"]) == {"combined"}


class TestVendiConfidenceInterval:
    """Comparing two arms needs a sense of how much of a gap is sampling noise."""

    def test_the_interval_is_deterministic_under_a_fixed_seed(self):
        X = np.eye(8, dtype=np.float32)
        assert diversity.vendi_ci(X, n_boot=40, seed=1) == diversity.vendi_ci(X, n_boot=40, seed=1)

    def test_a_wider_interval_for_a_mixed_corpus_than_for_identical_docs(self):
        same = np.stack([_unit([1, 0, 0])] * 12)
        mixed = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])] * 4)
        lo_s, hi_s = diversity.vendi_ci(same, n_boot=60, seed=0)
        lo_m, hi_m = diversity.vendi_ci(mixed, n_boot=60, seed=0)
        assert hi_s - lo_s == pytest.approx(0.0, abs=1e-6)  # no spread to resample
        assert hi_m - lo_m > 0

    def test_the_interval_is_ordered_and_bounded_by_n(self):
        X = np.eye(10, dtype=np.float32)
        lo, hi = diversity.vendi_ci(X, n_boot=50, seed=0)
        assert 1.0 <= lo <= hi <= 10.0

    def test_too_few_documents_or_resamples_report_no_interval(self):
        assert diversity.vendi_ci(np.eye(1, dtype=np.float32)) is None
        assert diversity.vendi_ci(np.eye(5, dtype=np.float32), n_boot=1) is None

    def test_the_block_carries_score_ratio_and_spread(self):
        block = diversity.vendi_block(np.eye(6, dtype=np.float32), n_boot=30, seed=0)
        assert block["score"] == pytest.approx(6.0, abs=1e-3)
        assert block["ratio"] == pytest.approx(1.0, abs=1e-3)
        assert block["spread"] == pytest.approx(block["boot_hi"] - block["boot_lo"], abs=1e-6)

    def test_the_block_omits_the_spread_when_it_cannot_be_computed(self):
        assert "spread" not in diversity.vendi_block(np.eye(6, dtype=np.float32), n_boot=0)

    def test_the_resample_range_sits_below_the_point_estimate(self):
        """Resampling duplicates documents, so it cannot bracket the estimate.

        Only the WIDTH is meaningful, which is why the report prints a spread
        and never an interval around the score.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 16)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        block = diversity.vendi_block(X, n_boot=60, seed=0)
        assert block["boot_hi"] < block["score"]

    def test_what_is_reported_is_the_width_not_the_endpoints(self):
        block = diversity.vendi_block(np.eye(6, dtype=np.float32), n_boot=30, seed=0)
        note = diversity.spread_note(block)
        assert note.startswith(" \u00b1")
        assert str(block["boot_lo"]) not in note


class TestRecordLanguage:
    def test_the_corpus_field_wins(self):
        assert diversity.record_language({"language": "Urdu"}) == "Urdu"

    def test_falls_back_to_the_dealt_culture_card(self):
        rec = {"variables": {"culture": "Japan, written in Japanese, with local idioms"}}
        assert diversity.record_language(rec) == "Japanese"

    def test_a_record_with_neither_takes_the_default(self):
        """English for callers that need a unit or a bucket..."""
        assert diversity.record_language({"variables": {}}) == "English"
        assert diversity.record_language({}) == "English"

    def test_grouping_callers_ask_for_no_default(self):
        """...and None where grouping BY language, so an unknown one is not
        silently counted as English."""
        assert diversity.record_language({"variables": {}}, default=None) is None
        assert diversity.record_language({}, default=None) is None


class TestGroupVendi:
    def _recs(self, languages):
        return [{"language": lang} for lang in languages]

    def test_per_language_vendi_is_computed_within_each_group(self):
        recs = self._recs(["English"] * 3 + ["Spanish"] * 3)
        # English docs are orthogonal (diverse); Spanish docs are identical
        X = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])]
                     + [_unit([1, 1, 0])] * 3)
        rows = diversity.group_vendi(recs, X, "language", min_group=2, n_boot=0)

        by_key = {r["key"]: r for r in rows}
        assert by_key["English"]["score"] == pytest.approx(3.0, abs=1e-3)
        assert by_key["Spanish"]["score"] == pytest.approx(1.0, abs=1e-3)

    def test_groups_are_ordered_largest_first(self):
        recs = self._recs(["English"] * 2 + ["Spanish"] * 5)
        X = np.eye(7, dtype=np.float32)
        rows = diversity.group_vendi(recs, X, "language", min_group=1, n_boot=0)
        assert [r["key"] for r in rows] == ["Spanish", "English"]

    def test_a_group_under_the_floor_reports_null_rather_than_a_size_artefact(self):
        recs = self._recs(["English"] * 12 + ["Norwegian"] * 2)
        X = np.eye(14, dtype=np.float32)
        rows = diversity.group_vendi(recs, X, "language", min_group=10, n_boot=0)

        by_key = {r["key"]: r for r in rows}
        assert by_key["Norwegian"]["score"] is None
        assert "fewer than 10" in by_key["Norwegian"]["note"]
        assert by_key["English"]["score"] is not None

    def test_no_record_carrying_the_axis_reports_nothing(self):
        assert diversity.group_vendi([{}, {}], np.eye(2, dtype=np.float32),
                                     "language", n_boot=0) is None

    def test_the_axis_can_also_be_type_id(self):
        recs = [{"type_id": "a"}, {"type_id": "a"}, {"type_id": "b"}, {"type_id": "b"}]
        rows = diversity.group_vendi(recs, np.eye(4, dtype=np.float32), "type_id",
                                     min_group=2, n_boot=0)
        assert {r["key"] for r in rows} == {"a", "b"}


def test_a_free_rerun_keeps_the_paid_idea_block(tmp_path, monkeypatch, stub_embeddings):
    """The idea pass costs a call per document; running the free pass again
    must not silently discard it."""
    import json
    from evals import diversity

    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    (run / "audit").mkdir()
    (run / "final" / "sdf_corpus.jsonl").write_text(
        "\n".join(json.dumps({"doc_id": f"d{i}", "content": f"Document number {i}. " * 40})
                  for i in range(4)), encoding="utf-8")
    report = run / "audit" / "diversity_report.json"
    report.write_text(json.dumps({"ideas": {"n": 4, "vendi": {"score": 3.1}}}), encoding="utf-8")

    stub_embeddings()
    monkeypatch.setattr("sys.argv", ["diversity.py", "--input", str(run)])
    diversity.main()

    kept = json.loads(report.read_text(encoding="utf-8"))
    assert kept["ideas"]["n"] == 4, "the paid idea block was overwritten by the free pass"
    assert kept["ideas"]["vendi"]["score"] == 3.1
