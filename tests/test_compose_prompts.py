"""Tests for sdf_pipeline/compose_prompts.py (offline, no API)."""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from sdf_pipeline import compose_prompts as cp
from shared import entity_pools, utils
from shared import matrix as _matrix

REPO_ROOT = Path(__file__).resolve().parent.parent


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# --- variables.txt parsing ---

def test_parse_plain_and_weighted_values(tmp_path):
    path = write(tmp_path / "variables.txt", (
        "# comment\n"
        "{tone}  # description\n"
        "    0.5 :: neutral\n"
        "    0.5 :: skeptical, with :: in the prose\n"
        "\n"
        "{culture}\n"
        "    # --- section divider ---\n"
        "    France\n"
        "    the United States\n"
    ))
    parsed = cp.parse_variables(path)
    assert parsed["tone"] == [("neutral", 0.5), ("skeptical, with :: in the prose", 0.5)]
    assert parsed["culture"] == [("France", None), ("the United States", None)]


def test_split_weights_uniform_fallback():
    values, weights = cp.split_weights({"a": [("x", None), ("y", None)]})
    assert values["a"] == ["x", "y"]
    assert weights["a"] == [0.5, 0.5]


def test_split_weights_rejects_mixed():
    with pytest.raises(ValueError, match="all-or-nothing"):
        cp.split_weights({"a": [("x", 0.5), ("y", None)]})


def test_split_weights_rejects_bad_sum():
    with pytest.raises(ValueError, match="sum"):
        cp.split_weights({"a": [("x", 0.5), ("y", 0.4)]})


@pytest.mark.parametrize("name", ["preamble", "fictional_names", "fictional_orgs", "sentient_example"])
def test_parse_rejects_reserved_definition(tmp_path, name):
    path = write(tmp_path / "variables.txt", f"{{{name}}}\n    text\n")
    with pytest.raises(ValueError, match="reserved"):
        cp.parse_variables(path)


def test_parse_rejects_duplicate_variable(tmp_path):
    path = write(tmp_path / "variables.txt", "{a}\n    x\n{a}\n    y\n")
    with pytest.raises(ValueError, match="duplicate"):
        cp.parse_variables(path)


def test_parse_rejects_value_before_header(tmp_path):
    path = write(tmp_path / "variables.txt", "    orphan value\n")
    with pytest.raises(ValueError, match="before any"):
        cp.parse_variables(path)


# --- quota arithmetic and deck sampling ---

def test_largest_remainder_quotas():
    assert cp.largest_remainder([0.5, 0.3, 0.2], 7) == [4, 2, 1]
    assert cp.largest_remainder([0.5, 0.5], 5) == [3, 2]
    assert sum(cp.largest_remainder([0.11, 0.29, 0.6], 100)) == 100


def test_deck_sample_exact_marginals():
    values = {"a": ["a1", "a2"], "b": ["b1", "b2", "b3"]}
    weights = {"a": [0.7, 0.3], "b": [1 / 3] * 3}
    rows = cp.deck_sample(values, weights, ["a", "b"], 10, random.Random(0))
    assert len(rows) == 10
    assert Counter(r["a"] for r in rows) == {"a1": 7, "a2": 3}
    assert Counter(r["b"] for r in rows) == {"b1": 4, "b2": 3, "b3": 3}


def test_deck_sample_is_seed_deterministic():
    values = {"a": [str(i) for i in range(5)]}
    weights = {"a": [0.2] * 5}
    one = cp.deck_sample(values, weights, ["a"], 20, random.Random(7))
    two = cp.deck_sample(values, weights, ["a"], 20, random.Random(7))
    assert one == two


# --- template handling ---

def test_template_placeholders_and_axes():
    template = "{preamble} {a} {b} {a}"
    assert cp.template_placeholders(template) == ["preamble", "a", "b"]
    assert cp.matrix_axes(template) == ["a", "b"]


def test_template_placeholders_rejects_format_specs():
    with pytest.raises(ValueError, match="plain names"):
        cp.template_placeholders("{a:>10}")


def test_split_sections():
    system, user = cp.split_sections(
        "=== SYSTEM PROMPT ===\n\nSYS\n\n=== USER PROMPT ===\n\nUSR\n"
    )
    assert (system, user) == ("SYS", "USR")
    assert cp.split_sections("just a user prompt\n") == (None, "just a user prompt")


def test_module_defaults_point_at_real_files():
    """The standalone CLI's default paths must exist — a template rename that
    misses these constants breaks `python sdf_pipeline/compose_prompts.py`
    while every pipeline test (which passes explicit paths) stays green."""
    for default in (cp.DEFAULT_TEMPLATE, cp.DEFAULT_VARIABLES, cp.DEFAULT_PREAMBLE,
                    cp.DEFAULT_DEAL_CONSTRAINTS):
        assert default.is_file(), f"{default} does not exist"


def test_real_templates_render_with_canonical_loader():
    """Both matrix templates must render through utils.load_prompt (str.format),
    like every other pipeline template — a stray literal brace fails here."""
    plan_path = REPO_ROOT / "prompts" / "sdf" / "layer1.txt"
    plan_system, plan_user = cp.split_sections(utils.load_prompt(
        plan_path,
        preamble="P",
        fictional_names="N1; N2",
        fictional_orgs="O1; O2",
        sentient_example="S1",
        **{n: "x" for n in cp.matrix_axes(plan_path.read_text(encoding="utf-8"))},
    ))
    assert plan_system == "P"
    assert "N1; N2" in plan_user
    assert "=" * 3 not in plan_user  # markers never reach the API
    assert "{" not in plan_user

    l3_system, l3_user = cp.split_sections(utils.load_prompt(
        REPO_ROOT / "prompts" / "sdf" / "layer2.txt",
        preamble="P",
        constitution_claude="CC",
        constitution_principles="CP",
        document_description="DESC",
        considerations_spec="SPEC-BLOCK",
        variable_lines="VARIABLE-BLOCK",
    ))
    assert "CC" in l3_system and "CP" in l3_system  # constitution lives in system
    assert "<document_description>" in l3_user and "DESC" in l3_user
    assert "SPEC-BLOCK" in l3_user  # the considerations spec rendered in the draft prompt
    assert "VARIABLE-BLOCK" in l3_user  # the dealt variables reach the drafter
    assert "{" not in l3_user

    # layer 4 is a two-part file: everything static (constitution, principles,
    # the nine checks) lives in the SYSTEM section; the USER section holds only
    # the closing judgment text and the two variable blocks, spec then document.
    l4_system, l4_user = cp.split_sections(utils.load_prompt(
        REPO_ROOT / "prompts" / "sdf" / "layer3.txt",
        constitution_claude="CC",
        constitution_principles="CP",
        document_description="DESC",
        document="DOC",
        considerations_spec="SPEC-BLOCK",
        variable_lines="VARIABLE-BLOCK",
    ))
    assert l4_system.index("<constitution>") < l4_system.index("CC") \
        < l4_system.index("<constitution_principles>") < l4_system.index("CP") \
        < l4_system.index("TEACH WHY") < l4_system.index("HOUSE STYLE")
    assert l4_user.index("DESC") < l4_user.index("VARIABLE-BLOCK") < l4_user.index("DOC") < l4_user.index("<improved_document>")
    assert "SPEC-BLOCK" in l4_user
    assert "{" not in l4_system and "{" not in l4_user

    # layer 5 mirrors the layer-4 split; the judge scores spec_conformance
    # (not diversity — a single-doc judge can't see the corpus)
    l5_system, l5_user = cp.split_sections(utils.load_prompt(
        REPO_ROOT / "prompts" / "sdf" / "layer4.txt",
        constitution_claude="CC",
        document_description="DESC",
        improved_document="DOC",
        variable_lines="VARIABLE-BLOCK",
    ))
    assert "spec_conformance" in l5_system and "diversity" not in l5_system.lower()
    assert l5_system.index("CC") < l5_system.index("SPEC CONFORMANCE")
    assert l5_user.index("DESC") < l5_user.index("VARIABLE-BLOCK") < l5_user.index("DOC")
    assert "{" not in l5_system and "{" not in l5_user


def test_downstream_only_axis_sampled_recorded_but_not_in_plan_prompt():
    """An axis defined in variables.txt but absent from the plan template is a
    downstream-only axis: it is deck-sampled to its weights and recorded in
    each record's `variables` (so a later stage can render it), yet never
    appears in the plan prompt the plan model sees."""
    template = "Type: {document_type}.\n"  # references only document_type
    values = {
        "document_type": ["a news article", "a blog post"],
        "reasoning_featured": ["individualism", "scope sensitivity"],
    }
    weights = {"document_type": [0.5, 0.5], "reasoning_featured": [0.5, 0.5]}
    recs = list(cp.compose_records(template, values, weights, None, n_prompts=8, seed=0))
    assert len(recs) == 8
    for r in recs:
        assert r["variables"]["reasoning_featured"] in ("individualism", "scope sensitivity")
        assert "reasoning_featured" not in r["prompt"]
        assert r["variables"]["reasoning_featured"] not in r["prompt"]
    # deck-sampled to weights exactly (largest remainder): 4 each of 8
    counts = Counter(r["variables"]["reasoning_featured"] for r in recs)
    assert counts["individualism"] == 4 and counts["scope sensitivity"] == 4


# --- locale-matched entity pools ---

def test_locale_for_culture_mapping():
    assert entity_pools.locale_for_culture(
        "Japan, written in Japanese, with Japanese idioms and references") == "ja_JP"
    assert entity_pools.locale_for_culture(
        "the United States, written in English, with American idioms and references") == "en_US"
    assert entity_pools.locale_for_culture("Kenya, written in English") is None
    assert entity_pools.locale_for_culture("Atlantis, written in Atlantean") is None


def test_every_culture_value_maps_or_falls_back():
    """Every culture in the real variables file must hit the mapping table
    (a new culture that silently misses would get the fallback without anyone
    deciding that on purpose)."""
    values, _ = cp.split_weights(cp.parse_variables(
        REPO_ROOT / "prompts" / "sdf" / "variables.txt"))
    for culture in values["culture"]:
        head = culture.casefold()
        assert any(c in head for c in entity_pools._CULTURE_LOCALES), culture


def test_species_pool_covers_every_sentient_category():
    """Every sentient_category value in the real variables file must have a
    non-empty species pool. A category without one silently renders the generic
    SPECIES_FALLBACK for {sentient_example} — a coverage gap no one chose. The pools
    are keyed by value ID, so rewording a category never breaks the lookup."""
    _, _, value_ids = cp.load_variables(REPO_ROOT / "prompts" / "sdf" / "variables.txt")
    for category_id in value_ids["sentient_category"]:
        assert cp.SPECIES_EXAMPLES.get(category_id), category_id
    assert set(cp.SPECIES_EXAMPLES) == set(value_ids["sentient_category"])


def test_species_example_follows_the_category_id_not_its_wording():
    reworded = {"sentient_category": "pigs and cattle and their kin"}
    ids = {"pigs and cattle and their kin": "farmed_mammals"}
    example = cp.species_example_for(reworded, "matrix_000001", 137, ids)
    assert example in cp.SPECIES_EXAMPLES["farmed_mammals"]
    assert cp.species_example_for(reworded, "matrix_000001", 137, {}) == cp.SPECIES_FALLBACK


def test_sentient_example_injected_from_pool_and_deterministic():
    """{sentient_example} is drawn per prompt from the drawn category's pool,
    is always a member of it, and is stable for a given seed (so --resume
    re-renders identical prompts)."""
    template = "Minds: {sentient_category}. Example: {sentient_example}.\n"
    values = {"sentient_category": ["farmed fishes", "pets/companion animals"]}
    weights = {"sentient_category": [0.5, 0.5]}
    recs = list(cp.compose_records(template, values, weights, None, n_prompts=6, seed=0))
    for r in recs:
        cat = r["variables"]["sentient_category"]
        example = r["prompt"].split("Example: ")[1].rstrip(".")
        assert example in cp.SPECIES_EXAMPLES[cp.default_sentient_category_ids()[cat]], (cat, example)
        assert "sentient_example" not in r["variables"]  # reserved slot stays out
    again = list(cp.compose_records(template, values, weights, None, n_prompts=6, seed=0))
    assert [r["prompt"] for r in recs] == [r["prompt"] for r in again]


def test_real_template_axes_match_real_variables():
    """Contract between variables.txt and the templates that consume it:

    - every plan-template placeholder is an axis with values (a placeholder
      without values is a fatal composer error);
    - every axis in variables.txt is consumed by SOME stage template — the
      plan template (layer1.txt) or the draft template (layer2.txt) — so a
      variable that nothing renders (a typo, or a dropped consumer) is caught
      here rather than silently sampled into a dead column;
    - all weights validate (split_weights raises on a bad sum)."""
    plan_tmpl = (REPO_ROOT / "prompts" / "sdf" / "layer1.txt").read_text(encoding="utf-8")
    draft_tmpl = (REPO_ROOT / "prompts" / "sdf" / "layer2.txt").read_text(encoding="utf-8")
    plan_axes = cp.matrix_axes(plan_tmpl)
    draft_placeholders = cp.template_placeholders(draft_tmpl)
    values, _ = cp.split_weights(cp.parse_variables(
        REPO_ROOT / "prompts" / "sdf" / "variables.txt"))
    for axis in ("naming", "domain", "decision_scale"):
        assert axis in plan_axes, axis
        assert values[axis], axis
    # Every plan placeholder must have values.
    assert set(plan_axes) <= set(values)
    # reasoning_featured is no longer dealt: the selector (layer1_select_considerations)
    # picks considerations after the plan, and only the drafter receives the result.
    assert "reasoning_featured" not in values
    assert "reasoning_featured" not in plan_axes
    assert "reasoning_featured" not in draft_placeholders
    assert "considerations_spec" in draft_placeholders
    # No orphan axes: everything defined is rendered by the plan or draft stage.
    consumed = set(plan_axes) | set(draft_placeholders)
    assert set(values) <= consumed, set(values) - consumed


def test_build_pools_for_locale_native_script_and_determinism():
    people_a, orgs_a = entity_pools.build_pools_for_locale("ja_JP", n_people=10, n_orgs=5, seed=1)
    people_b, _ = entity_pools.build_pools_for_locale("ja_JP", n_people=10, n_orgs=5, seed=1)
    assert people_a and orgs_a
    assert people_a == people_b
    assert any(any(ord(ch) > 127 for ch in name) for name in people_a)  # native script
    banned = {"chen", "johnson", "miller", "smith", "martinez", "sarah", "emily"}
    for name in people_a + orgs_a:
        assert not ({t.strip(".,").casefold() for t in name.split()} & banned)


# --- DOCUMENT DESCRIPTION handoff ---

def test_extract_description_tags_bound_both_ends():
    plan = ("<document_planning>notes here</document_planning>\n"
            "<document_description>\nA spec line.\nSecond line.\n</document_description>\n"
            "Note: kept under 300 words.")
    assert cp.extract_description(plan) == "A spec line.\nSecond line."


def test_extract_description_tagged_incoherent_deep_in_output():
    # INCOHERENT sits inside the tags, past any fixed-prefix scan window
    plan = ("<document_planning>" + "x" * 3000 + "</document_planning>\n"
            "<document_description>INCOHERENT: a marketing page cannot be a court ruling.</document_description>")
    assert cp.is_incoherent(plan)
    assert cp.extract_description(plan) is None


def test_extract_description_empty_tags_fail_closed():
    assert cp.extract_description("<document_description>  </document_description>") is None


def test_extract_description_variants():
    for heading in ("DOCUMENT DESCRIPTION", "## DOCUMENT DESCRIPTION",
                    "**DOCUMENT DESCRIPTION**", "# Document Description:"):
        plan = f"## Notes\nstuff\n\n{heading}\nA spec line.\nSecond line."
        assert cp.extract_description(plan) == "A spec line.\nSecond line."


def test_extract_description_fail_closed():
    assert cp.extract_description("## Notes\nno final heading here") is None
    assert cp.extract_description("## Notes\n\nDOCUMENT DESCRIPTION\n\n") is None
    incoherent = "INCOHERENT: a government FAQ cannot be a first-person diary."
    assert cp.is_incoherent(incoherent)
    assert cp.extract_description(incoherent) is None


def test_extract_description_ignores_inline_mention():
    plan = "notes mention the document description concept\n\nDOCUMENT DESCRIPTION\nSpec."
    assert cp.extract_description(plan) == "Spec."


# --- CLI end-to-end ---

def test_cli_deck_sample_writes_jsonl(tmp_path, monkeypatch, capsys):
    template = write(tmp_path / "template.txt", "{preamble}\n\nType: {doc}\nTone: {tone}\n")
    variables = write(tmp_path / "variables.txt", (
        "{doc}\n"
        "    0.75 :: [article] article\n"
        "    0.25 :: [memo] memo\n"
        "{tone}\n"
        "    [neutral] neutral\n"
        "    [skeptical] skeptical\n"
    ))
    preamble = write(tmp_path / "preamble.txt", "PREAMBLE TEXT\n")
    out = tmp_path / "prompts.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "compose_prompts.py", "--template", str(template), "--variables", str(variables),
        "--preamble", str(preamble), "--n-prompts", "8", "--seed", "0", "--out", str(out),
    ])
    cp.main()

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    assert Counter(r["variables"]["doc"] for r in records) == {"article": 6, "memo": 2}
    assert Counter(r["variables"]["tone"] for r in records) == {"neutral": 4, "skeptical": 4}
    for rec in records:
        assert set(rec["variables"]) == {"doc", "tone"}  # preamble stays out
        assert rec["system"] is None  # markerless template = single user prompt
        assert rec["prompt"].startswith("PREAMBLE TEXT\n")
        assert "{" not in rec["prompt"]


def test_cli_injects_locale_matched_names(tmp_path, monkeypatch, capsys):
    template = write(tmp_path / "template.txt", (
        "Culture: {culture}\nPeople: use {fictional_names}.\nOrgs: use {fictional_orgs}.\n"
    ))
    variables = write(tmp_path / "variables.txt", (
        "{culture}\n"
        "    [japan] Japan, written in Japanese, with Japanese idioms and references\n"
        "    [kenya] Kenya, written in English, with Kenyan idioms and references\n"
    ))
    out = tmp_path / "prompts.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "compose_prompts.py", "--template", str(template), "--variables", str(variables),
        "--out", str(out),
    ])
    cp.main()
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    japan = next(r for r in records if r["variables"]["culture"].startswith("Japan"))
    kenya = next(r for r in records if r["variables"]["culture"].startswith("Kenya"))
    # Japan gets real locale-matched suggestions (native script); Kenya has no
    # Faker locale and gets the instruction-only fallback.
    assert "names in the style of: " in japan["prompt"]
    assert any(ord(ch) > 127 for ch in japan["prompt"].split("People: use ")[1])
    assert cp.FALLBACK_NAMES in kenya["prompt"]
    assert cp.FALLBACK_ORGS in kenya["prompt"]
    # reserved slots never leak into the variables dict
    assert set(japan["variables"]) == {"culture"}


def test_cli_full_matrix_without_n(tmp_path, monkeypatch, capsys):
    template = write(tmp_path / "template.txt", "A: {a} B: {b}\n")
    variables = write(tmp_path / "variables.txt", "{a}\n    [one] 1\n    [two] 2\n{b}\n    [x] x\n    [y] y\n    [z] z\n")
    out = tmp_path / "all.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "compose_prompts.py", "--template", str(template), "--variables", str(variables),
        "--out", str(out),
    ])
    cp.main()
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert len({r["prompt"] for r in records}) == 6


def test_cli_errors_on_missing_values(tmp_path, monkeypatch):
    template = write(tmp_path / "template.txt", "{a} {ghost}\n")
    variables = write(tmp_path / "variables.txt", "{a}\n    [one] 1\n")
    monkeypatch.setattr(sys, "argv", [
        "compose_prompts.py", "--template", str(template), "--variables", str(variables),
    ])
    with pytest.raises(SystemExit, match="ghost"):
        cp.main()


# --- stable value IDs ---

ID_VARIABLES_TEXT = (
    "{tone}  # description\n"
    "    0.5 :: [supportive] supportive, like an advocate\n"
    "    0.5 :: [neutral] neutral\n"
    "{culture}\n"
    "    [france] France\n"
    "    [japan] Japan\n"
)


def test_id_prefix_is_stripped_from_value_text_and_exposed_separately(tmp_path):
    path = write(tmp_path / "variables.txt", ID_VARIABLES_TEXT)
    parsed = cp.parse_variables(path)
    assert parsed["tone"] == [("supportive, like an advocate", 0.5), ("neutral", 0.5)]
    assert parsed["culture"] == [("France", None), ("Japan", None)]
    parsed_again, ids = _matrix.parse_variables_with_ids(path)
    assert parsed_again == parsed and ids == {"tone": ["supportive", "neutral"], "culture": ["france", "japan"]}
    assert _matrix.value_ids_by_variable(parsed, ids)["tone"] == {
        "supportive": "supportive, like an advocate", "neutral": "neutral"}


def test_stripping_ids_leaves_value_text_identical_to_the_same_file_without_them(tmp_path):
    with_ids = write(tmp_path / "with_ids.txt", ID_VARIABLES_TEXT)
    without_ids = write(tmp_path / "without_ids.txt", (
        "{tone}  # description\n    0.5 :: supportive, like an advocate\n    0.5 :: neutral\n"
        "{culture}\n    France\n    Japan\n"))
    assert cp.parse_variables(with_ids) == cp.parse_variables(without_ids)


def test_files_without_ids_parse_as_before_and_report_no_ids(tmp_path):
    path = write(tmp_path / "variables.txt", "{a}\n    0.5 :: x\n    0.5 :: y\n")
    parsed, ids = _matrix.parse_variables_with_ids(path)
    assert parsed == {"a": [("x", 0.5), ("y", 0.5)]} and ids == {"a": [None, None]}
    assert _matrix.value_ids_by_variable(parsed, ids) == {"a": {}}


@pytest.mark.parametrize("bad_value", ["[Bad_Id] text", "[1bad] text", "[no_space]text", "[unclosed text", "[] text", "[id]"])
def test_malformed_id_prefix_fails_naming_variable_and_line(tmp_path, bad_value):
    path = write(tmp_path / "variables.txt", f"{{tone}}\n    [ok] fine\n    {bad_value}\n")
    with pytest.raises(ValueError, match=r"variables\.txt:3: \{tone\}.*'\['"):
        _matrix.parse_variables_with_ids(path)


def test_duplicate_id_within_a_variable_fails_but_the_same_id_in_two_variables_is_fine(tmp_path):
    path = write(tmp_path / "variables.txt", "{a}\n    [x] one\n    [x] two\n")
    with pytest.raises(ValueError, match=r"duplicate ID \[x\] in \{a\}"):
        _matrix.parse_variables_with_ids(path)
    ok = write(tmp_path / "ok.txt", "{a}\n    [x] one\n{b}\n    [x] two\n")
    assert _matrix.parse_variables_with_ids(ok)[1] == {"a": ["x"], "b": ["x"]}


def test_sdf_loader_requires_every_value_to_carry_an_id(tmp_path):
    path = write(tmp_path / "variables.txt", "{a}\n    [one] 1\n    two without an id\n")
    with pytest.raises(ValueError, match=r"\{a\} value has no \[id\] prefix: 'two without an id'"):
        cp.load_variables(path)
    values, weights, value_ids = cp.load_variables(write(tmp_path / "ok.txt", "{a}\n    [one] 1\n    [two] 2\n"))
    assert value_ids == {"a": {"one": "1", "two": "2"}} and values == {"a": ["1", "2"]}
    # the DAD-style opt-out parses the same file
    assert cp.load_variables(path, require_ids=False)[2] == {"a": {"one": "1"}}


def test_sdf_cli_rejects_a_values_file_without_ids(tmp_path, monkeypatch):
    template = write(tmp_path / "template.txt", "{a}\n")
    variables = write(tmp_path / "variables.txt", "{a}\n    1\n")
    monkeypatch.setattr(sys, "argv", [
        "compose_prompts.py", "--template", str(template), "--variables", str(variables)])
    with pytest.raises(SystemExit, match="no \\[id\\] prefix"):
        cp.main()


def test_real_variables_file_has_an_id_on_every_value_and_ids_are_snake_case():
    values, weights, value_ids = cp.load_variables(REPO_ROOT / "prompts" / "sdf" / "variables.txt")
    for variable, by_id in value_ids.items():
        assert len(by_id) == len(values[variable])
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", value_id) for value_id in by_id)


def test_dad_variables_still_parse_without_ids():
    parsed, ids = _matrix.parse_variables_with_ids(REPO_ROOT / "prompts" / "dad" / "variables.txt")
    assert parsed and all(value_id is None for per_variable in ids.values() for value_id in per_variable)


# --- render_variable_lines (shared by the screen, the description rewrite and the validation scripts) ---

def _real_matrix():
    prompts = REPO_ROOT / "prompts" / "sdf"
    template = (prompts / "layer1.txt").read_text(encoding="utf-8")
    values, weights, value_ids = cp.load_variables(prompts / "variables.txt")
    rules = cp.load_deal_rules(prompts / "deal_constraints.txt", value_ids)
    return template, values, weights, value_ids, rules


def test_render_variable_lines_match_the_lines_the_planner_sees():
    template, values, weights, _, rules = _real_matrix()
    preamble = (REPO_ROOT / "prompts" / "sdf" / "preamble.txt").read_text(encoding="utf-8")
    records = list(cp.compose_records(template, values, weights, preamble, n_prompts=12, seed=3, deal_rules=rules))
    for record in records:
        lines = cp.render_variable_lines(record, template, rules)
        assert lines in record["prompt"]  # contiguous, verbatim, in the planner's order
        assert lines.startswith("Type: the document is ") and (
            "The AI assistant character" in lines or "The AI the discussion is about" in lines)
        assert "{" not in lines
    assert len({cp.render_variable_lines(r, template, rules) for r in records}) == len(records)


def test_render_variable_lines_apply_the_deal_rule_rendering_rules():
    template, values, weights, value_ids, rules = _real_matrix()
    base = cp.deal_hands(template, values, weights, 5, 0, rules)[0]
    no_stake = {**base, "resolution": value_ids["resolution"]["no_stake"]}
    for variables, omits in ((base, False), (no_stake, True)):
        record = {"prompt_id": "matrix_000000", "variables": variables}
        lines = cp.render_variable_lines(record, template, rules)
        assert ("The main tradeoff with welfare" not in lines) is omits
        assert ("The welfare stakes are on the scale" not in lines) is omits
        assert ("The welfare dilemma is" not in lines) is omits
        assert variables["domain"] in lines  # the dealt card is still shown elsewhere
        prompt = next(cp.compose_records(template, values, weights, "p", None, deal_rules=rules, hands=[record]))
        assert lines in prompt["prompt"]


def test_render_variable_lines_use_the_ids_example_species_and_a_runs_snapshot_template():
    template, values, weights, _, rules = _real_matrix()
    hand = cp.deal_hands(template, values, weights, 5, 0, rules)[0]
    first = cp.render_variable_lines({"prompt_id": "matrix_000000", "variables": hand}, template, rules)
    assert first == cp.render_variable_lines({"prompt_id": "matrix_000000", "variables": hand}, template, rules)
    species = {cp.species_example_for(hand, f"matrix_{i:06d}", 137) for i in range(40)}
    assert len(species) > 1  # drawn per id, so it can differ between hands dealt the same category
    default_template = cp.render_variable_lines({"prompt_id": "matrix_000000", "variables": hand}, None, rules)
    assert default_template == first  # template=None reads the repo's layer1.txt


def test_compose_records_renders_given_hands_under_their_own_ids():
    template, values, weights, _, rules = _real_matrix()
    hands = cp.hand_records(cp.deal_hands(template, values, weights, 6, 0, rules))
    kept = [hands[4], hands[1]]
    records = list(cp.compose_records(template, values, weights, "p", None, deal_rules=rules, hands=kept))
    assert [r["prompt_id"] for r in records] == ["matrix_000004", "matrix_000001"]
    assert [r["variables"] for r in records] == [h["variables"] for h in kept]


def test_render_variable_lines_apply_the_reword_rules_so_later_stages_see_the_planners_lines():
    template, values, weights, value_ids, rules = _real_matrix()
    base = cp.deal_hands(template, values, weights, 5, 0, rules)[0]
    for resolution_id, reworded in (
        ("general_discussion_no_scenario", True), ("constitution_inclusion_no_scenario", True),
        ("no_stake", True), ("creative_solution", False),
    ):
        variables = {**base, "resolution": value_ids["resolution"][resolution_id]}
        record = {"prompt_id": "matrix_000000", "variables": variables}
        lines = cp.render_variable_lines(record, template, rules)
        prompt = next(cp.compose_records(
            template, values, weights, "p", None, deal_rules=rules, hands=[record]))["prompt"]
        assert lines in prompt  # the same reworded lines the planner saw
        if resolution_id.endswith("no_scenario"):
            assert "The AI the discussion is about is" in lines
            assert "The AI assistant character in the document is" not in lines
        elif resolution_id == "no_stake":
            assert "The task arises from" in lines and "The welfare issues discussed arise from" not in lines
        else:
            assert "The AI assistant character in the document is" in lines
    # the later-stage renderer built from a prompts directory applies them too
    renderer = cp.make_variable_line_renderer(REPO_ROOT / "prompts" / "sdf")
    no_scenario = {**base, "resolution": value_ids["resolution"]["general_discussion_no_scenario"]}
    assert "The AI the discussion is about is" in renderer({"prompt_id": "matrix_000000", "variables": no_scenario})
