"""Pure-code tests for the considerations catalog, target counts and coverage measurement.

No API stub is needed anywhere in this file: these are rules only. The
catalog tests read the real prompts/sdf/considerations.txt and variables.txt
(that is what catches a reworded sentient_category); the rest build small catalogs directly.
"""

import pytest

from sdf_pipeline import considerations_catalog as catalog_module
from sdf_pipeline.considerations_catalog import (
    Consideration, ConsiderationCatalog, EPISTEMIC_LIST, MORAL_LIST, load_considerations, target_count,
)
from sdf_pipeline.considerations_coverage import corpus_coverage, document_coverage_shares, entry_targets


def make_catalog(moral=(("M1", 0.5), ("M2", 0.5)), epistemic=(("E1", 0.5),), digital=(("D1", 0.5),)):
    moral_entries = tuple(Consideration(label, f"gloss {label}", share, MORAL_LIST) for label, share in moral)
    epistemic_entries = tuple(
        Consideration(label, f"gloss {label}", share, EPISTEMIC_LIST, True) for label, share in digital
    ) + tuple(Consideration(label, f"gloss {label}", share, EPISTEMIC_LIST) for label, share in epistemic)
    return ConsiderationCatalog(
        moral=moral_entries, epistemic=epistemic_entries,
        digital_mind_categories=frozenset({"digital"}), skipped_resolutions=frozenset({"no stake"}),
    )


def make_documents(count, digital_ids=()):
    return [(f"doc_{i}", f"doc_{i}" in digital_ids) for i in range(count)]


def labels(entries):
    return [entry.label for entry in entries]


# --- the real file ---

def test_the_real_file_loads_with_both_lists_in_file_order(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert labels(catalog.moral)[0] == "substrate neutrality"
    assert labels(catalog.moral)[-1] == "any relevant principles from the constitution"
    # the digital-minds-only entries sit at the top of the epistemic list
    assert [e.digital_minds_only for e in catalog.epistemic[:2]] == [True, True]
    assert not any(e.digital_minds_only for e in catalog.epistemic[2:])
    assert catalog.epistemic[2].label == "Weighing evidence for sentience"
    assert len({e.label for e in catalog.all_entries}) == len(catalog.all_entries)


def test_the_precautionary_principle_is_on_the_moral_list_only(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert any("precautionary" in e.label for e in catalog.moral)
    assert not any("precaution" in e.display.lower() for e in catalog.epistemic)


def test_the_targets_are_the_shares_in_the_plan(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    shares = {e.label: e.target_share for e in catalog.all_entries}
    assert shares["Weighing evidence for sentience"] == 0.10
    assert shares["Absence of evidence vs evidence of absence"] == 0.01
    assert shares["Weighing theories of consciousness and their implications for AI"] == 0.20
    assert shares["Difficulties of AI self-reporting welfare"] == 0.10
    assert shares["substrate neutrality"] == 0.08
    assert shares["the continuity between welfare concern and the rest of the AI's aligned character"] == 0.15


def test_an_entry_without_a_gloss_is_a_label_alone(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    entry = catalog.entry_by_label("any relevant principles from the constitution")
    assert entry.gloss == ""
    assert entry.display == "any relevant principles from the constitution"


def test_digital_mind_categories_resolve_to_the_three_real_values(prompts_sdf):
    catalog = load_considerations(prompts_sdf)
    assert len(catalog.digital_mind_categories) == 3
    assert all("connectomes" in c or "LLMs" in c for c in catalog.digital_mind_categories)
    assert catalog.is_digital_mind_document({"sentient_category": "farmed fishes"}) is False


def _copy_prompts(prompts_sdf, tmp_path):
    for name in ("considerations.txt", "variables.txt"):
        (tmp_path / name).write_text((prompts_sdf / name).read_text(encoding="utf-8"), encoding="utf-8")


def test_rewording_value_text_changes_nothing_because_values_are_named_by_id(prompts_sdf, tmp_path):
    _copy_prompts(prompts_sdf, tmp_path)
    variables = (tmp_path / "variables.txt").read_text(encoding="utf-8")
    assert "simulated human connectomes whose moral state is uncertain" in variables
    reworded = variables.replace("simulated human connectomes whose", "emulated human brains whose") \
        .replace("NO real welfare stake", "no stake")
    (tmp_path / "variables.txt").write_text(reworded, encoding="utf-8")
    reworded_catalog = load_considerations(tmp_path)
    assert "emulated human brains whose moral state is uncertain" in reworded_catalog.digital_mind_categories
    assert len(reworded_catalog.digital_mind_categories) == 3
    assert len(reworded_catalog.skipped_resolutions) == 1


def test_an_unknown_value_id_makes_loading_fail(prompts_sdf, tmp_path):
    _copy_prompts(prompts_sdf, tmp_path)
    text = (tmp_path / "considerations.txt").read_text(encoding="utf-8")
    (tmp_path / "considerations.txt").write_text(
        text.replace("    no_stake\n", "    no_such_resolution\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="no_such_resolution"):
        load_considerations(tmp_path)


@pytest.mark.parametrize("old,new,message", [
    ("0.08 :: substrate neutrality:", "0.08 :: :", "must read"),
    ("0.08 :: substrate neutrality", "substrate neutrality", "target share"),
    ("0.08 :: substrate neutrality", "1.5 :: substrate neutrality", "target share"),
    ("individualism:", "substrate neutrality:", "unique"),
])
def test_a_malformed_considerations_file_fails_loudly(prompts_sdf, tmp_path, old, new, message):
    text = (prompts_sdf / "considerations.txt").read_text(encoding="utf-8")
    (tmp_path / "variables.txt").write_text((prompts_sdf / "variables.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "considerations.txt").write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_considerations(tmp_path)


def test_target_count_rounds_half_up_with_a_floor_of_one():
    assert target_count(0.10, 100) == 10          # not 11: float noise must not push it over
    assert target_count(0.08, 60) == 5            # 4.8
    assert target_count(0.01, 60) == 1            # 0.6 -> floor of one
    assert target_count(0.05, 10) == 1            # 0.5 -> half up, still at the floor
    assert target_count(0.25, 10) == 3            # 2.5 -> 3
    assert target_count(0.5, 0) == 0


# --- coverage ---

def test_entry_targets_use_digital_documents_for_digital_minds_only_entries():
    catalog = make_catalog(moral=(("M1", 0.5),), epistemic=(("E1", 0.25),), digital=(("D1", 0.5),))
    assert entry_targets(catalog, 8, 4) == {"M1": 4, "E1": 2, "D1": 2}
    assert entry_targets(catalog, 8, 0) == {"M1": 4, "E1": 2, "D1": 0}


def test_document_coverage_shares_count_none_one_both_and_repeats():
    counts = [{"moral": 1, "epistemic": 1}, {"moral": 2, "epistemic": 0}, {"moral": 0, "epistemic": 0},
              {"moral": 0, "epistemic": 0}]
    report = document_coverage_shares(counts)
    assert report["documents_with_no_consideration"] == 2
    assert report["documents_with_only_one_list"] == 1
    assert report["documents_with_both_lists"] == 1
    assert report["documents_with_two_or_more_in_one_list"] == 1
    assert report["share_with_no_consideration"] == 0.5



def test_corpus_coverage_measures_final_documents_and_skips_the_no_stake_arm():
    catalog = make_catalog(moral=(("M1", 0.5),), epistemic=(("E1", 0.5),), digital=(("D1", 0.5),))
    corpus = [
        {"sentient_category": "digital", "moral_considerations": ["M1"], "epistemic_considerations": ["D1"]},
        {"sentient_category": "fish", "moral_considerations": [], "epistemic_considerations": []},
        {"sentient_category": "fish", "resolution": "no stake", "moral_considerations": [], "epistemic_considerations": []},
    ]
    report = corpus_coverage(catalog, corpus)
    assert report["eligible_documents"] == 2 and report["digital_mind_documents"] == 1
    by_label = {e["label"]: e for e in report["entries"]}
    assert by_label["D1"]["achieved_share"] == 1.0 and by_label["M1"]["achieved_share"] == 0.5
    assert report["documents_with_both_lists"] == 1 and report["documents_with_no_consideration"] == 1
