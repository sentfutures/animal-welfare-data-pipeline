"""Staging two experiment arms as blind preference pairs.

The properties that make the rating trustworthy: a pair holds the SAME document
from two arms, the arm labels survive into the record so a rating can be
deblinded afterwards, interleaved comparisons stay distinguishable in the data
but not to the rater, and an early stop leaves a random sample rather than a
prefix of the corpus.
"""

import json

import pytest

from pref_pipeline import from_sdf_arms, prefdata
from shared import utils


def make_arm(run_dir, doc_ids, body):
    """An arm run dir carrying one rewrite per doc_id."""
    records = [{
        "doc_id": d,
        "variables": {"culture": "the United States, written in English"},
        "description": f"Spec for {d}.",
        "review": "Problems.",
        "content": f"{body} {d}",
    } for d in doc_ids]
    utils.save_jsonl(records, run_dir / "layer3" / "rewrites.jsonl")
    return run_dir


@pytest.fixture
def exp_dir(tmp_path):
    """An experiment of three arms over the same five documents."""
    docs = [f"matrix_{i:06d}" for i in range(5)]
    arms = {}
    for name in ("control", "control-b", "split"):
        arms[name] = str(make_arm(tmp_path / "runs" / name, docs, f"{name} text"))
    exp = tmp_path / "exp"
    utils.ensure_dir(exp)
    (exp / "manifest.json").write_text(json.dumps(
        {"exp_id": "split", "arms": arms, "n": len(docs)}), encoding="utf-8")
    return exp


class TestBuildPairs:
    def test_a_pair_holds_the_same_document_from_both_arms(self, exp_dir):
        pairs = from_sdf_arms.build_pairs(exp_dir, [("control", "split")])

        assert len(pairs) == 5
        for p in pairs:
            assert p["arm_names"] == {"a": "control", "b": "split"}
            assert p["response_a"] == f"control text {p['doc_id']}"
            assert p["response_b"] == f"split text {p['doc_id']}"

    def test_the_language_column_is_derived_from_the_sdf_culture_card(self, exp_dir):
        """SDF deals {culture}; `cultural_setting` is the DAD key and is absent here.

        Reading the wrong one is silent — every row is just None — so this
        asserts a real language rather than the column's presence.
        """
        pairs = from_sdf_arms.build_pairs(exp_dir, [("control", "split")])
        assert {p["language"] for p in pairs} == {"English"}

    def test_a_record_with_no_culture_card_leaves_a_visible_gap(self, tmp_path):
        """None, not "English": an invented label would be worse than a hole."""
        docs = ["matrix_000000"]
        arms = {}
        for name in ("control", "split"):
            run = tmp_path / "runs" / name
            utils.save_jsonl([{"doc_id": docs[0], "variables": {},
                               "description": "Spec.", "content": f"{name} text"}],
                             run / "layer3" / "rewrites.jsonl")
            arms[name] = str(run)
        exp = utils.ensure_dir(tmp_path / "exp")
        (exp / "manifest.json").write_text(json.dumps(
            {"exp_id": "x", "arms": arms, "n": 1}), encoding="utf-8")

        pairs = from_sdf_arms.build_pairs(exp, [("control", "split")])
        assert pairs[0]["language"] is None

    def test_the_spec_both_versions_answer_rides_on_the_pair(self, exp_dir):
        pairs = from_sdf_arms.build_pairs(exp_dir, [("control", "split")])

        for p in pairs:
            assert f"Spec for {p['doc_id']}." in p["user_message"]
            assert "NATURALNESS" in p["user_message"]

    def test_interleaved_comparisons_are_labelled_and_uniquely_keyed(self, exp_dir):
        pairs = from_sdf_arms.build_pairs(
            exp_dir, [("control", "split"), ("control", "control-b")])

        assert len(pairs) == 10
        assert len({p["pair_id"] for p in pairs}) == 10
        kinds = {p["comparison"] for p in pairs}
        assert kinds == {"control-vs-split", "control-vs-control-b"}

    def test_the_pair_id_leaks_neither_the_arms_nor_the_document(self, exp_dir):
        """rate.py prints the pair id above the documents, so it must say nothing.

        An id naming its arms tells the rater which comparison they are in, and
        a calibration comparison the rater can recognise calibrates nothing.
        """
        pairs = from_sdf_arms.build_pairs(
            exp_dir, [("control", "split"), ("control", "control-b")])

        for p in pairs:
            for leak in ("control", "split", "control-b", p["doc_id"]):
                assert leak not in p["pair_id"], f"{p['pair_id']} leaks {leak!r}"
        assert len({p["pair_id"] for p in pairs}) == len(pairs)

    def test_an_arm_outside_the_manifest_is_refused(self, exp_dir):
        with pytest.raises(SystemExit):
            from_sdf_arms.build_pairs(exp_dir, [("control", "nope")])


class TestTheStagedRun:
    def _run(self, monkeypatch, tmp_path, exp_dir, argv):
        monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(tmp_path / "out"))
        monkeypatch.setattr("sys.argv", ["from_sdf_arms.py", "--exp", str(exp_dir), *argv])
        from_sdf_arms.main()
        runs = sorted((tmp_path / "out" / "pref" / "runs").iterdir())
        assert len(runs) == 1
        return runs[0]

    def test_the_rating_app_can_read_what_was_staged(self, monkeypatch, tmp_path, exp_dir):
        run_dir = self._run(monkeypatch, tmp_path, exp_dir, ["--pair", "control:split"])

        pairs = prefdata.load_pairs(run_dir)
        assert len(pairs) == 5
        assert run_dir in prefdata.list_runs(tmp_path / "out" / "pref" / "runs")
        # every pair is displayable: a side is assigned and both sides resolve
        for p in pairs:
            left, right = prefdata.sides(p)
            assert {left, right} == {"a", "b"}
            assert p[f"response_{left}"] and p[f"response_{right}"]

    def test_a_rating_deblinds_to_the_arm_that_won(self, monkeypatch, tmp_path, exp_dir):
        run_dir = self._run(monkeypatch, tmp_path, exp_dir, ["--pair", "control:split"])
        pair = prefdata.load_pairs(run_dir)[0]
        left, _ = prefdata.sides(pair)

        prefdata.record_rating(run_dir, pair, rater="t", choice="left")

        exported = utils.load_jsonl(run_dir / "final" / "preferences.jsonl")
        assert len(exported) == 1
        assert exported[0]["chosen_arm_name"] == pair["arm_names"][left]
        assert exported[0]["chosen"] == pair[f"response_{left}"]

    def test_the_order_is_seeded_so_an_early_stop_is_a_random_sample(
            self, monkeypatch, tmp_path, exp_dir):
        a = self._run(monkeypatch, tmp_path / "a", exp_dir,
                      ["--pair", "control:split", "--seed", "7"])
        b = self._run(monkeypatch, tmp_path / "b", exp_dir,
                      ["--pair", "control:split", "--seed", "7"])
        c = self._run(monkeypatch, tmp_path / "c", exp_dir,
                      ["--pair", "control:split", "--seed", "8"])

        order = lambda d: [p["pair_id"] for p in prefdata.load_pairs(d)]
        assert order(a) == order(b)
        assert order(a) != order(c)

    def test_every_prefix_is_balanced_across_the_interleaved_comparisons(
            self, monkeypatch, tmp_path, exp_dir):
        """A rater who stops early must leave data on BOTH comparisons."""
        run_dir = self._run(monkeypatch, tmp_path, exp_dir,
                            ["--pair", "control:split", "--pair", "control:control-b"])
        kinds = [p["comparison"] for p in prefdata.load_pairs(run_dir)]

        assert len(kinds) == 10
        for cut in range(2, len(kinds) + 1):
            seen = kinds[:cut]
            a = seen.count("control-vs-split")
            # two equal-sized groups, so any prefix is within one of even
            assert abs(a - (cut - a)) <= 1, f"prefix of {cut} is {a} vs {cut - a}"

    def test_limit_caps_after_the_shuffle_not_before(self, monkeypatch, tmp_path, exp_dir):
        full = self._run(monkeypatch, tmp_path / "f", exp_dir,
                         ["--pair", "control:split", "--seed", "3"])
        cut = self._run(monkeypatch, tmp_path / "c", exp_dir,
                        ["--pair", "control:split", "--seed", "3", "--limit", "2"])

        kept = [p["pair_id"] for p in prefdata.load_pairs(cut)]
        assert kept == [p["pair_id"] for p in prefdata.load_pairs(full)][:2]
