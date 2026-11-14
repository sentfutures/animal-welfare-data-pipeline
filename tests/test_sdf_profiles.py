"""SDF content profiles: the shared pipeline code reads the example pools and
the constitution-grounding file names from the active profile
(sdf_pipeline/profiles/). The animal profile is the original pipeline; these
tests pin that switching profiles reaches each place that used to hardcode
animal content, and that the human profile mirrors the animal one."""

import json
import re
from types import SimpleNamespace

import pytest

from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import layer2_draft, layer3_rewrite, profiles
from sdf_pipeline import run as sdf_run
from sdf_pipeline.considerations_catalog import load_considerations
from sdf_pipeline.profiles import animal, human
from shared import constitution_loader

# principle-1 titles, unique to each grounding CSV
ANIMAL_PRINCIPLE = "Sentient beings are inside the moral circle"
HUMAN_PRINCIPLE = "Other people's interests are always in the weighing"
# failure-mode headings unique to each failure-modes file
ANIMAL_FAILURE = "Substituting species or ecosystems for individuals"
HUMAN_FAILURE = "Substituting the group or goal for the individuals"


def _fake_profile(**overrides):
    """A copy of the animal profile with chosen attributes replaced, so a test
    can see which code paths read the profile without needing new content."""
    attrs = {k: getattr(animal, k) for k in dir(animal) if k.isupper()}
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def dealt_hand(profile):
    """One card from every variable of the profile's real variables.txt: the
    layer-2-to-4 templates render the dealt variables, so a plan needs a
    complete hand."""
    values, _, _ = cp.load_variables(profiles.prompts_dir(profile) / "variables.txt")
    return {name: cards[0] for name, cards in values.items()}


def make_plan(profile, pid="matrix_000000"):
    return {"prompt_id": pid, "variables": dealt_hand(profile), "plan": "…",
            "description": "Spec text.", "incoherent": False}


@pytest.fixture
def stage_dir(tmp_path):
    d = tmp_path / "stage"
    d.mkdir()
    return d


class TestRegistry:
    def test_default_is_animal(self):
        assert profiles.current() is animal
        assert profiles.name_from_config({}) == "animal"
        assert profiles.name_from_config({"sdf": {}}) == "animal"

    def test_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match="unknown SDF profile"):
            profiles.get("no-such-profile")

    def test_use_activates_and_reset_restores_the_default(self):
        assert profiles.use("human") is human
        profiles.reset()
        assert profiles.current() is animal

    def test_animal_profile_points_at_the_original_locations(self):
        assert profiles.prompts_dir(animal).name == "sdf"
        assert animal.OUTPUT_SUBDIR == "sdf"
        assert animal.PRINCIPLES_FILENAME == "constitution_principles.csv"
        assert animal.FAILURE_MODES_FILENAME == "failure_modes.md"
        assert profiles.principles_filenames(animal) == ("constitution_principles.csv",)


class TestCodeReadsTheActiveProfile:
    def test_example_pool_comes_from_the_profile(self, monkeypatch):
        fake = _fake_profile(CATEGORY_EXAMPLES={"farmed_mammals": ("test-example",)},
                             CATEGORY_EXAMPLE_FALLBACK="test-fallback")
        monkeypatch.setattr(profiles, "_current", fake)
        hand = {"sentient_category": "some reworded category"}
        ids = {"some reworded category": "farmed_mammals"}
        assert cp.species_example_for(hand, "matrix_000001", 137, ids) == "test-example"
        # a category with no pool in the active profile gets ITS fallback
        assert cp.species_example_for(hand, "matrix_000001", 137, {}) == "test-fallback"

    def test_layer2_sends_the_profiles_principles_file(
        self, tiny_config, stage_dir, stub_claude, monkeypatch
    ):
        monkeypatch.setattr(profiles, "_current", human)
        calls = stub_claude(lambda user_message, **kw: "<document>D.</document>")
        layer2_draft.run(tiny_config, profiles.prompts_dir(human), stage_dir,
                         [make_plan(human)])
        assert HUMAN_PRINCIPLE in calls[0]["system_prompt"]
        assert ANIMAL_PRINCIPLE not in calls[0]["system_prompt"]

    def test_layer3_sends_the_profiles_principles_and_failure_modes(
        self, tiny_config, stage_dir, stub_claude, monkeypatch
    ):
        monkeypatch.setattr(profiles, "_current", human)
        config = {**tiny_config, "sdf": {**tiny_config["sdf"], "revision": {"split": False}}}
        calls = stub_claude(lambda user_message, **kw:
                            "r\n<improved_document>B.</improved_document>")
        plan = make_plan(human)
        draft = {"doc_id": plan["prompt_id"], "variables": plan["variables"],
                 "description": "Spec text.", "content": "Un document."}
        layer3_rewrite.run(config, profiles.prompts_dir(human), stage_dir, [draft])
        system = calls[0]["system_prompt"]
        assert HUMAN_PRINCIPLE in system and HUMAN_FAILURE in system
        assert ANIMAL_PRINCIPLE not in system and ANIMAL_FAILURE not in system


class TestResumeKeepsTheRunsProfile:
    def _manifest(self, tmp_path, config):
        (tmp_path / "run_manifest.json").write_text(json.dumps({"config": config}))
        return tmp_path

    def test_resuming_under_another_profile_is_refused(self, tmp_path):
        run = self._manifest(tmp_path, {"sdf": {"profile": "human"}})
        with pytest.raises(SystemExit, match="profile 'human'"):
            sdf_run._refuse_profile_change(run, "animal")

    def test_same_profile_and_pre_profile_runs_resume(self, tmp_path):
        sdf_run._refuse_profile_change(self._manifest(tmp_path, {"sdf": {"profile": "animal"}}), "animal")
        sdf_run._refuse_profile_change(self._manifest(tmp_path, {"sdf": {}}), "animal")   # predates profiles
        sdf_run._refuse_profile_change(tmp_path / "missing", "animal")                    # no manifest


class TestHumanSdfProfile:
    """The human-welfare profile mirrors the animal one axis by axis; these pin
    the mirror so the two cannot drift apart silently."""

    def test_registered_and_points_at_its_own_locations(self):
        assert profiles.get("human") is human
        assert profiles.prompts_dir(human).name == "sdf_human"
        assert human.OUTPUT_SUBDIR == "sdf_human"
        assert human.PRINCIPLES_FILENAME == "constitution_principles_human.csv"
        assert human.FAILURE_MODES_FILENAME == "failure_modes_human.md"

    def test_variables_mirror_the_animal_axes(self):
        def axes(profile):
            text = (profiles.prompts_dir(profile) / "variables.txt").read_text(encoding="utf-8")
            return [line.split("}")[0] + "}" for line in text.splitlines() if line.startswith("{")]
        assert axes(human) == axes(animal)

    def test_axis_weights_and_value_counts_mirror_the_animal_file(self):
        a_values, a_weights, _ = cp.load_variables(profiles.prompts_dir(animal) / "variables.txt")
        h_values, h_weights, _ = cp.load_variables(profiles.prompts_dir(human) / "variables.txt")
        for axis in a_values:
            assert len(h_values[axis]) == len(a_values[axis]), axis
            assert h_weights.get(axis) == pytest.approx(a_weights.get(axis)), axis

    def test_templates_mirror_the_animal_placeholders(self):
        prompts_a, prompts_h = profiles.prompts_dir(animal), profiles.prompts_dir(human)
        names = sorted(p.name for p in prompts_a.iterdir()
                       if p.name.startswith(("layer", "preamble")) and "wording" not in p.name)
        assert names, "no templates found"
        slots = lambda p: sorted(set(re.findall(r"\{(\w+)\}", p.read_text(encoding="utf-8"))))
        for name in names:
            assert (prompts_h / name).exists(), name
            assert slots(prompts_h / name) == slots(prompts_a / name), name

    def test_a_full_deal_is_clean_and_realizes_every_category(self, monkeypatch):
        monkeypatch.setattr(profiles, "_current", human)
        pdir = profiles.prompts_dir(human)
        template = (pdir / "layer1.txt").read_text(encoding="utf-8")
        values, weights, value_ids = cp.load_variables(pdir / "variables.txt")
        rules = cp.load_deal_rules(pdir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
        recs = list(cp.compose_records(template, values, weights, "PREAMBLE",
                                       n_prompts=200, seed=0,
                                       deal_rules=rules, value_ids=value_ids))
        assert len(recs) == 200
        dealt = {r["variables"]["sentient_category"] for r in recs}
        assert dealt == set(values["sentient_category"])
        # every rendered prompt carries a concrete example from the human pools
        pool = {e for examples in human.CATEGORY_EXAMPLES.values() for e in examples}
        assert all(any(e in r["prompt"] for e in pool) for r in recs)

    def test_no_stake_and_no_scenario_wordings_render_against_the_human_layer1(self, tmp_path, monkeypatch):
        """Exercises the wording files' exact-line coupling: a drifted line in
        human layer1.txt fails here, not on a paid run."""
        monkeypatch.setattr(profiles, "_current", human)
        pdir = profiles.prompts_dir(human)
        text = (pdir / "variables.txt").read_text(encoding="utf-8")
        text = text.replace("0.00 :: [no_stake]", "0.20 :: [no_stake]")
        text = text.replace("0.12 :: [creative_solution]", "0.04 :: [creative_solution]")
        text = text.replace("0.12 :: [ai_instrumental_in_improvement]",
                            "0.04 :: [ai_instrumental_in_improvement]")
        text = text.replace("0.12 :: [user_makes_modest_sacrifice]",
                            "0.08 :: [user_makes_modest_sacrifice]")
        variables = tmp_path / "variables.txt"
        variables.write_text(text, encoding="utf-8")
        template = (pdir / "layer1.txt").read_text(encoding="utf-8")
        values, weights, value_ids = cp.load_variables(variables)
        rules = cp.load_deal_rules(pdir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
        recs = list(cp.compose_records(template, values, weights, "PREAMBLE",
                                       n_prompts=100, seed=1,
                                       deal_rules=rules, value_ids=value_ids))
        no_stake = [r for r in recs if "NO real stake" in r["prompt"]]
        assert no_stake
        for r in no_stake:
            assert "The people the task involves are" in r["prompt"]
            assert "The welfare dilemma is" not in r["prompt"]
            assert "The welfare stakes are on the scale of" not in r["prompt"]
        no_scenario = [r for r in recs
                       if r["variables"]["resolution"].startswith("The document does not feature")]
        assert no_scenario
        assert all("whose welfare the discussion covers" in r["prompt"] for r in no_scenario)

    def test_considerations_mirror_the_animal_entry_counts(self):
        a = load_considerations(profiles.prompts_dir(animal))
        h = load_considerations(profiles.prompts_dir(human))
        assert len(h.moral) == len(a.moral)
        assert len(h.epistemic) == len(a.epistemic)

    def test_principles_and_failure_modes_load_and_render(self):
        rows = constitution_loader.load_principles(filename=human.PRINCIPLES_FILENAME)
        assert rows
        block = constitution_loader.format_principles(rows)
        assert all(r["principle"] in block for r in rows)
        failure_modes = constitution_loader.load_failure_modes(filename=human.FAILURE_MODES_FILENAME)
        assert HUMAN_FAILURE in failure_modes

    def test_stub_probe_phrases_survive_translation(self):
        """tests/test_e2e_smoke.py's SDF dispatcher keys on these phrases."""
        pdir = profiles.prompts_dir(human)
        assert "description of a specific document" in (pdir / "layer1.txt").read_text(encoding="utf-8")
        assert "score" in (pdir / "layer4.txt").read_text(encoding="utf-8").lower()
