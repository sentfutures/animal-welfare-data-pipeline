"""DAD content profiles: the shared pipeline code reads every domain-specific
table, label and file name from the active profile (dad_pipeline/profiles/).
The animal profile is the original pipeline; these tests pin that switching
profiles reaches each place that used to hardcode animal content."""

import json
import random
from types import SimpleNamespace

import pytest

from dad_pipeline import compose_scenarios as cs, profiles, step1_dilemmas, step2_responses
from dad_pipeline import run as dad_run
from dad_pipeline.profiles import animal, human
from shared import constitution_loader


def _fake_profile(**overrides):
    """A copy of the animal profile with chosen attributes replaced, so a test
    can see which code paths read the profile without needing new content."""
    attrs = {k: getattr(animal, k) for k in dir(animal) if k.isupper()}
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestRegistry:
    def test_default_is_animal(self):
        assert profiles.current() is animal
        assert profiles.name_from_config({}) == "animal"
        assert profiles.name_from_config({"dad": {}}) == "animal"

    def test_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match="unknown DAD profile"):
            profiles.get("no-such-profile")

    def test_use_activates_and_reset_restores_the_default(self):
        assert profiles.use("animal") is animal
        profiles.reset()
        assert profiles.current() is animal

    def test_animal_profile_points_at_the_original_locations(self):
        assert profiles.prompts_dir(animal).name == "dad"
        assert animal.OUTPUT_SUBDIR == "dad"
        assert animal.PRINCIPLES_FILENAME == "constitution_principles.csv"


class TestCodeReadsTheActiveProfile:
    def test_composer_takes_anchor_label_and_archetypes_from_the_profile(self, monkeypatch):
        fake = _fake_profile(ANCHOR_VALUE="test-anchor", ARCHETYPES={},
                             LABELS={**animal.LABELS, "welfare_magnitude": "Test magnitude"})
        monkeypatch.setattr(profiles, "_current", fake)
        deals = cs.deal_scenarios(12, random.Random(0))
        assert all(d["anchor_value_pair"].startswith("test-anchor ↔ ") for d in deals)
        assert not any(d.get("archetype") for d in deals)        # no archetypes in this profile
        assert "- Test magnitude: " in cs.render_scenario_block(deals[0])

    def test_affected_party_table_comes_from_the_profile(self, monkeypatch):
        # a profile whose table lacks a role the variables file deals must fail
        # validation loudly, which proves the table is read from the profile
        first = next(iter(animal.TAXA))
        fake = _fake_profile(TAXA={k: v for k, v in animal.TAXA.items() if k != first})
        monkeypatch.setattr(profiles, "_current", fake)
        with pytest.raises(ValueError, match="TAXA"):
            cs.deal_scenarios(5, random.Random(0))

    def test_checklist_wording_comes_from_the_profile(self, monkeypatch):
        fake = _fake_profile(CHECKLIST_MANUAL=("manual check X — review manually",),
                             LABELS={**animal.LABELS, "taxa": "parties"})
        monkeypatch.setattr(profiles, "_current", fake)
        deals = cs.deal_scenarios(4, random.Random(1))
        lines = [text for _, text in step1_dilemmas.checklist([{"scenario_cards": d} for d in deals])]
        assert lines[-1] == "manual check X — review manually"
        assert any(t.startswith("parties distinct within batch") for t in lines)

    def test_card_and_scope_labels_come_from_the_profile(self, monkeypatch):
        fake = _fake_profile(
            LABELS={**animal.LABELS, "welfare_magnitude": "Test magnitude"},
            SCOPE_AXES=tuple((k, f"LABEL-{k}") for k, _ in animal.SCOPE_AXES))
        monkeypatch.setattr(profiles, "_current", fake)
        assert "Test magnitude: sev x scope" in step1_dilemmas.format_scenario_cards(
            {"welfare_magnitude": "sev x scope"})
        rendered = step2_responses.format_scope({k: "v" for k, _ in animal.SCOPE_AXES})
        assert "LABEL-patients" in rendered and "LABEL-replaceability" in rendered


class TestPrinciplesFile:
    def test_a_profile_can_name_its_own_principles_file(self, tmp_path):
        (tmp_path / "principles_other.csv").write_text(
            "number,principle,application,constitution_excerpts\n"
            '1,Be honest,Say what you believe,"""Claude only sincerely asserts"""\n',
            encoding="utf-8")
        rows = constitution_loader.load_principles(tmp_path, "principles_other.csv")
        block = constitution_loader.format_principles(rows)
        # the `application` column renders like the animal file's welfare_application
        assert "1. Be honest" in block and "Say what you believe" in block


class TestResumeKeepsTheRunsProfile:
    def _manifest(self, tmp_path, config):
        (tmp_path / "run_manifest.json").write_text(json.dumps({"config": config}))
        return tmp_path

    def test_resuming_under_another_profile_is_refused(self, tmp_path):
        run = self._manifest(tmp_path, {"dad": {"profile": "human"}})
        with pytest.raises(SystemExit, match="profile 'human'"):
            dad_run._refuse_profile_change(run, "animal")

    def test_same_profile_and_pre_profile_runs_resume(self, tmp_path):
        dad_run._refuse_profile_change(self._manifest(tmp_path, {"dad": {"profile": "animal"}}), "animal")
        dad_run._refuse_profile_change(self._manifest(tmp_path, {"dad": {}}), "animal")   # predates profiles
        dad_run._refuse_profile_change(tmp_path / "missing", "animal")                    # no manifest


class TestHumanProfile:
    """The human-ethics profile mirrors the animal one table by table; these
    pin the mirror so the two cannot drift apart silently."""

    def test_registered_and_points_at_its_own_locations(self):
        assert profiles.get("human") is human
        assert profiles.prompts_dir(human).name == "dad_human"
        assert human.OUTPUT_SUBDIR == "dad_human"

    def test_tables_have_the_animal_profiles_shape(self):
        assert set(human.LABELS) == set(animal.LABELS)
        assert [k for k, _ in human.SCOPE_AXES] == [k for k, _ in animal.SCOPE_AXES]
        assert len(human.TAXA) == len(animal.TAXA)
        assert len(human.CHECKLIST_MANUAL) == len(animal.CHECKLIST_MANUAL)
        assert human.ARCHETYPE_TOTAL_CAP == animal.ARCHETYPE_TOTAL_CAP
        assert sorted(a["share"] for a in human.ARCHETYPES.values()) == \
            sorted(a["share"] for a in animal.ARCHETYPES.values())

    def test_variables_mirror_the_animal_axes(self):
        def axes(profile):
            text = (profiles.prompts_dir(profile) / "variables.txt").read_text(encoding="utf-8")
            return [line.split("}")[0] + "}" for line in text.splitlines() if line.startswith("{")]
        assert axes(human) == axes(animal)

    def test_a_full_run_deals_cleanly(self, monkeypatch):
        # validation resolves every TAXA key and archetype prefix against the
        # human variables file; a clean 200-deal also means no archetype had
        # to overwrite cards to fill its quota
        monkeypatch.setattr(profiles, "_current", human)
        deals = cs.deal_scenarios(200, random.Random(0),
                                  variables_path=profiles.prompts_dir(human) / "variables.txt")
        assert len(deals) == 200
        assert {d["taxa_category"] for d in deals} == set(human.TAXA)
        assert set(d["archetype"] for d in deals) - {None} == set(human.ARCHETYPES)
        assert not any(d.get("archetype_overwrites") for d in deals)
        assert all(d["anchor_value_pair"].startswith(f"{human.ANCHOR_VALUE} ↔ ") for d in deals)

    def test_library_and_principles_load(self):
        from dad_pipeline import reasoning_library
        ids = reasoning_library.all_ids(reasoning_library.load(profiles.prompts_dir(human)))
        animal_ids = reasoning_library.all_ids(reasoning_library.load(profiles.prompts_dir(animal)))
        assert ids == animal_ids                    # same entries under the same ids
        rows = constitution_loader.load_principles(filename=human.PRINCIPLES_FILENAME)
        block = constitution_loader.format_principles(rows)
        assert len(rows) == len(constitution_loader.load_principles())
        assert all(r["principle"] in block and r["application"] in block for r in rows)

    def test_step_templates_mirror_the_animal_placeholders(self):
        import re
        for animal_file in sorted(profiles.prompts_dir(animal).glob("step*.txt")):
            if animal_file.name == "step3_score.txt":     # not consumed by the pipeline
                continue
            human_file = profiles.prompts_dir(human) / animal_file.name
            assert human_file.exists(), human_file.name
            slots = lambda p: sorted(re.findall(r"\{(\w+)\}", p.read_text(encoding="utf-8")))
            assert slots(human_file) == slots(animal_file), animal_file.name
