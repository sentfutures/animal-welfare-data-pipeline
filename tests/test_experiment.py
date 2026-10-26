"""The arms driver: shared drafts, matched deals, and no re-billed work.

Three properties make an arms comparison trustworthy, and each is a test here:
every arm sees the SAME drafts, every arm sees the SAME dealt cards, and the
holdout never overlaps the sample that chose the winner. The rest is the usual
money discipline — resume and import must cost nothing.
"""

import json
import random

import pytest
import yaml

from experiments import experiment
from shared import utils

PYEOF_LANGUAGES = {
    "en": "USA, written in English, with American idioms",
    "zh": "China, written in Mandarin Chinese, with Chinese idioms",
    "es": "Mexico, written in Spanish, with Mexican idioms",
}


def make_drafts(counts: dict[str, int]) -> list[dict]:
    """A donor corpus with a known language mix."""
    out, i = [], 0
    for code, n in counts.items():
        for _ in range(n):
            out.append({
                "doc_id": f"matrix_{i:06d}",
                "variables": {"culture": PYEOF_LANGUAGES[code],
                              "document_type": "a blog post"},
                "description": "Spec text.",
                "content": f"Document {i}.",
            })
            i += 1
    return out


class TestStratifiedDraw:
    def test_the_sample_reproduces_the_donor_language_mix(self):
        drafts = make_drafts({"en": 60, "zh": 30, "es": 10})
        buckets = {}
        for r in drafts:
            buckets.setdefault(experiment.language_of(r), []).append(r)
        sample = experiment.stratified_draw(buckets, 20, random.Random(0))

        mix = experiment.language_mix(sample)
        assert mix == {"English": 12, "Mandarin Chinese": 6, "Spanish": 2}

    def test_the_holdout_is_disjoint_from_the_sample(self):
        drafts = make_drafts({"en": 60, "zh": 30, "es": 10})
        buckets = {}
        for r in drafts:
            buckets.setdefault(experiment.language_of(r), []).append(r)
        rng = random.Random(0)
        sample = experiment.stratified_draw(buckets, 30, rng)
        holdout = experiment.stratified_draw(buckets, 30, rng)

        assert not ({d["doc_id"] for d in sample} & {d["doc_id"] for d in holdout})
        assert len(sample) == len(holdout) == 30

    def test_the_same_seed_draws_the_same_sample(self):
        def draw():
            buckets = {}
            for r in make_drafts({"en": 40, "zh": 20}):
                buckets.setdefault(experiment.language_of(r), []).append(r)
            return [d["doc_id"] for d in experiment.stratified_draw(buckets, 15, random.Random(7))]

        assert draw() == draw()

    def test_a_language_too_small_for_its_share_does_not_shorten_the_sample(self):
        """A quota above a bucket's capacity is redistributed, not dropped."""
        drafts = make_drafts({"en": 50, "zh": 2, "es": 48})
        buckets = {}
        for r in drafts:
            buckets.setdefault(experiment.language_of(r), []).append(r)
        sample = experiment.stratified_draw(buckets, 90, random.Random(0))

        assert len(sample) == 90
        assert experiment.language_mix(sample)["Mandarin Chinese"] == 2

    def test_drawing_more_than_exists_fails_loudly(self):
        buckets = {"English": make_drafts({"en": 5})}
        with pytest.raises(SystemExit, match="only 5 available"):
            experiment.stratified_draw(buckets, 10, random.Random(0))

    def test_language_comes_from_the_corpus_field_when_present(self):
        assert experiment.language_of({"language": "Urdu", "variables": {}}) == "Urdu"
        assert experiment.language_of({"variables": {}}) == "English"  # derive's default


class TestTheDeal:
    def test_no_revision_axes_defined_deals_empty_cards(self, tmp_path):
        """A variables file with no revision axes leaves every document inert."""
        variables = tmp_path / "variables.txt"
        variables.write_text("# nothing defined yet\n", encoding="utf-8")
        deal = experiment.deal_revision(make_drafts({"en": 4}), variables, seed=0)
        assert all(d["cards"] == {} for d in deal.values())

    def test_only_revision_axes_are_dealt(self, tmp_path):
        """The file also holds the composition axes, already dealt to the donor."""
        variables = tmp_path / "variables.txt"
        variables.write_text(
            "{document_type}\n    a news article\n"
            "{length}\n    at roughly 0.5x its current length\n",
            encoding="utf-8",
        )
        deal = experiment.deal_revision(make_drafts({"en": 4}), variables, seed=0)
        for d in deal.values():
            assert set(d["cards"]) == {"length"}

    def test_dealt_card_marginals_match_the_declared_weights(self, tmp_path):
        variables = tmp_path / "variables.txt"
        variables.write_text(
            "{length}\n"
            "    0.5 :: at roughly 0.5x its current length\n"
            "    0.5 :: at roughly 1.3x its current length\n",
            encoding="utf-8",
        )
        deal = experiment.deal_revision(make_drafts({"en": 40}), variables, seed=0)
        counts = {}
        for d in deal.values():
            counts[d["cards"]["length"]] = counts.get(d["cards"]["length"], 0) + 1
        assert counts == {"at roughly 0.5x its current length": 20,
                          "at roughly 1.3x its current length": 20}

    def test_the_deal_is_reproducible_so_every_arm_sees_the_same_cards(self, tmp_path):
        variables = tmp_path / "variables.txt"
        variables.write_text(
            "{voice_rhythm}\n    0.5 :: clipped and fast-moving\n"
            "    0.5 :: discursive and willing to digress\n", encoding="utf-8")
        drafts = make_drafts({"en": 12})
        a = experiment.deal_revision(drafts, variables, seed=3)
        b = experiment.deal_revision(drafts, variables, seed=3)
        assert a == b


class TestArmConfig:
    def test_an_arm_sets_only_its_own_revision_knobs(self):
        base = {"sdf": {"n_prompts": 5, "min_score_threshold": 7}}
        merged = experiment.merge_arm_config(
            base, {"name": "voices", "style_mode": "axes", "rewrite_model": "claude-opus-4-8"})

        assert merged["sdf"]["revision"] == {"style_mode": "axes"}
        assert merged["sdf"]["rewrite_model"] == "claude-opus-4-8"
        assert merged["sdf"]["n_prompts"] == 5  # untouched
        assert "revision" not in base["sdf"]  # the base config is not mutated

    def test_an_unrecognised_arm_key_is_not_smuggled_into_the_config(self):
        merged = experiment.merge_arm_config({"sdf": {}}, {"name": "x", "nonsense": True})
        assert "nonsense" not in merged["sdf"]
        assert "nonsense" not in merged["sdf"]["revision"]


class TestTheArmsFile:
    def test_the_shipped_example_parses_and_defines_a_control(self):
        from pathlib import Path
        spec, arms = experiment.load_arms_file(
            Path(__file__).resolve().parent.parent / "experiments" / "arms.example.yaml")
        assert [a["name"] for a in arms] == ["control"]
        assert arms[0]["style_mode"] == "none"
        # the control must also pin the split off, or it is not the condition
        # layer 3 sent before these axes existed
        assert arms[0]["split"] is False
        # length_tolerance is a reporting width, not a bar: the pass/fail
        # criteria are the two that remain
        assert set(spec["success"]) == {
            "min_vendi_gain", "max_pass_rate_drop", "length_tolerance",
        }

    def test_the_length_sweep_parses_and_deals_length_to_one_arm(self):
        from pathlib import Path
        spec, arms = experiment.load_arms_file(
            Path(__file__).resolve().parent.parent / "experiments" / "length" / "arms.yaml")
        assert [a["name"] for a in arms] == ["control", "as-shipped", "length"]
        # exactly one arm renders the length card, and no arm styles voice, or
        # the length effect would not be separable from the voice one
        assert [a["name"] for a in arms if a.get("use_length")] == ["length"]
        assert not any(a.get("style_mode", "none") != "none" for a in arms)

    def test_duplicate_arm_names_are_refused(self, tmp_path):
        path = tmp_path / "arms.yaml"
        path.write_text(yaml.safe_dump(
            {"arms": [{"name": "a"}, {"name": "a"}]}), encoding="utf-8")
        with pytest.raises(SystemExit, match="unique 'name'"):
            experiment.load_arms_file(path)

    def test_an_empty_arms_file_is_refused(self, tmp_path):
        path = tmp_path / "arms.yaml"
        path.write_text("arms: []\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="no arms defined"):
            experiment.load_arms_file(path)


class TestImportRewrites:
    def test_imports_only_the_sampled_ids_and_marks_them_done(self, tmp_path):
        donor = utils.ensure_dir(tmp_path / "donor")
        utils.ensure_dir(donor / "layer3")
        utils.save_jsonl(
            [{"doc_id": f"matrix_{i:06d}", "variables": {}, "description": "d",
              "review": "r", "content": f"doc {i}"} for i in range(5)],
            donor / "layer3" / "rewrites.jsonl",
        )
        arm_layer3 = utils.ensure_dir(tmp_path / "arm" / "layer3")
        n = experiment.import_rewrites(donor, {"matrix_000001", "matrix_000003"}, arm_layer3)

        assert n == 2
        kept = utils.load_jsonl(arm_layer3 / "rewrites.jsonl")
        assert {r["doc_id"] for r in kept} == {"matrix_000001", "matrix_000003"}
        # checkpointed, so a later layer-3 pass over this arm re-bills nothing
        checkpoint = utils.Checkpoint(arm_layer3 / "_checkpoint.json")
        assert checkpoint.is_done("matrix_000001")

    def test_a_donor_with_no_rewrites_fails_loudly(self, tmp_path):
        donor = utils.ensure_dir(tmp_path / "donor")
        with pytest.raises(SystemExit, match="no rewrites"):
            experiment.import_rewrites(donor, {"x"}, utils.ensure_dir(tmp_path / "arm"))

    def test_a_pre_renumber_donor_is_read_from_its_own_layout(self, tmp_path):
        """The committed donor run predates the renumber: its DRAFTS are in
        layer3/ and its REWRITES in layer4/. Reading by layer number would
        return the drafts as rewrites."""
        donor = utils.ensure_dir(tmp_path / "donor")
        utils.ensure_dir(donor / "layer12")  # the marker for the old layout
        utils.ensure_dir(donor / "layer3")
        utils.ensure_dir(donor / "layer4")
        utils.save_jsonl([{"doc_id": "matrix_000000", "content": "a draft"}],
                         donor / "layer3" / "drafts.jsonl")
        utils.save_jsonl([{"doc_id": "matrix_000000", "content": "a rewrite"}],
                         donor / "layer4" / "rewrites.jsonl")

        assert utils.load_jsonl(utils.sdf_stage_file(donor, "draft"))[0]["content"] == "a draft"
        arm = utils.ensure_dir(tmp_path / "arm")
        experiment.import_rewrites(donor, {"matrix_000000"}, arm)
        assert utils.load_jsonl(arm / "rewrites.jsonl")[0]["content"] == "a rewrite"
