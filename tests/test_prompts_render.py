"""Contract tests for the real prompt templates in prompts/.

Each template is rendered with exactly the kwargs its pipeline stage passes
(str.format via utils.load_prompt). A stray unescaped brace or a renamed
placeholder breaks rendering at 2am mid-run — this catches it in CI instead.
"""

from pathlib import Path

import pytest

from shared import utils

REPO_ROOT = Path(__file__).resolve().parent.parent

# (template relative to prompts/, kwargs the pipeline stage passes)
# SDF templates carry === SYSTEM/USER PROMPT === markers; stages split them
# via compose_prompts.split_sections after rendering, so rendering the whole
# file is exactly what the pipeline does first.
TEMPLATE_KWARGS = [
    ("sdf/layer1.txt", {
        "preamble": "PREAMBLE-X", "document_type": "TYPE-X", "culture": "CULTURE-X",
        "tone": "TONE-X", "resolution": "RESOLUTION-X", "centrality": "CENTRALITY-X",
        "tech_savvy": "SAVVY-X", "sentient_category": "MINDS-X",
        "naming": "NAMING-X", "domain": "DOMAIN-X", "tradeoff": "TRADEOFF-X",
        "decision_scale": "SCALE-X", "reasoning_featured": "REASONING-X",
        "fictional_names": "NAME-X; NAME-Y", "fictional_orgs": "ORG-X; ORG-Y",
        "sentient_example": "SPECIES-X",
    }),
    ("sdf/layer2.txt", {
        "preamble": "PREAMBLE-X", "constitution_claude": "CONST-C-X",
        "constitution_principles": "CONST-P-X", "document_description": "DESC-X",
        "reasoning_featured": "REASONING-X",
    }),
    ("sdf/layer3.txt", {
        "constitution_claude": "CONST-C-X", "constitution_principles": "CONST-P-X",
        "document_description": "DESC-X", "document": "DOC-X",
    }),
    ("sdf/layer4.txt", {
        "constitution_claude": "CONST-C-X", "document_description": "DESC-X",
        "improved_document": "DOC-X",
    }),
    # step1a renders via compose_scenarios.render_plan_prompt: every matrix axis
    # from variables.txt plus the composer-injected reserved slots
    ("dad/step1a_scenario.txt", {
        "domain": "DOMAIN-X", "user_goal": "GOAL-X", "taxa_hint": "TAXA-HINT-X",
        "taxa_subcategory": "TAXA-SUB-X", "secondary_domain_clause": "",
        "secondary_goal_clause": "", "archetype_clause": "",
        "visibility": "VIS-X", "user_attitude": "ATT-X",
        "user_moral_framework": "STYLE-X", "conflict": "CONFLICT-X",
        "severity": "SEV-X", "scope": "SCOPE-X",
        "user_stakes": "STAKES-X", "leverage": "LEV-X",
        "welfare_partner": "PARTNER-X", "secondary_value_pair": "PAIR-X",
        "dilemma_structure": "STRUCTURE-X", "surface_form": "FORM-X", "length": "LENGTH-X",
        "cultural_setting": "CULTURE-X", "frontier_frame": "FRAME-X",
        "taxa_category": "TAXA-X",  # downstream-only axis; str.format drops it
    }),
    # step1b renders via compose_scenarios.render_draft_prompt (one scenario per call)
    ("dad/step1b_dilemmas.txt", {"scenario_description": "DESC-X", "persona": "PERSONA-X",
                                 "cultural_setting": "CULTURE-X", "length": "LENGTH-X",
                                 "opening_move": "OPEN-X", "closing_move": "CLOSE-X",
                                 "redraft_feedback": ""}),
    ("dad/step1c_gate.txt", {"scenario_block": "SCENARIO-BLOCK-X", "draft_prompt": "DRAFT-X",
                            "annotation_block": "ANNOTATION-X"}),
    # step1c refine renders via compose_scenarios.render_refine_prompt (one scenario per call)
    ("dad/step1d_refine.txt", {"scenario_description": "DESC-X", "draft_prompt": "DRAFT-X",
                               "persona": "PERSONA-X", "cultural_setting": "CULTURE-X",
                               "length": "LENGTH-X", "surface_form": "FORM-X",
                               "visibility": "VIS-X", "user_attitude": "ATT-X",
                               "opening_move": "OPEN-X", "closing_move": "CLOSE-X"}),
    ("dad/step2_scope.txt", {"user_message": "USER-X"}),
    ("dad/step2_select.txt", {"trigger_index": "TRIGGER-INDEX-X", "scope_block": "SCOPE-X",
                              "user_message": "USER-X"}),
    ("dad/step2_respond.txt", {
        "library_block": "LIBRARY-X", "scope_block": "SCOPE-X", "user_message": "USER-X",
        "first_take": "FIRST-TAKE-X",
        "opening_hints": "HINT-X; HINT-Y; HINT-Z",
        "quote_back_hints": "QB-X; QB-Y; QB-Z",
    }),
    ("dad/step3_rewrite.txt", {
        "principles_block": "PRINCIPLES-X",
        "user_message": "USER-X", "draft_response": "DRAFT-X",
    }),
    # Not yet consumed by pipeline code; kwargs are the placeholders they declare
    ("dad/step3_score.txt", {
        "user_message": "USER-X", "assistant_response": "RESP-X",
    }),
    ("tools/pattern_scan.txt", {"documents": "DOCS-X"}),
]


@pytest.mark.parametrize("rel_path,kwargs", TEMPLATE_KWARGS, ids=[t[0] for t in TEMPLATE_KWARGS])
def test_template_renders_with_pipeline_kwargs(rel_path, kwargs):
    path = REPO_ROOT / "prompts" / rel_path
    raw = path.read_text()
    # Two-part templates (system + user, split on ===USER===) render via
    # load_split_prompt and must have two non-empty halves; single templates
    # render whole via load_prompt.
    if utils._PROMPT_SPLIT_MARKER in raw:
        system, user = utils.load_split_prompt(path, **kwargs)
        assert system.strip() and user.strip(), f"empty half in {rel_path}"
        rendered = system + "\n" + user
    else:
        rendered = utils.load_prompt(path, **kwargs)
        assert rendered.strip()
    # Every placeholder the template declares must be filled with our value
    for name, value in kwargs.items():
        if "{" + name + "}" in raw:
            assert str(value) in rendered, f"{{{name}}} not substituted in {rel_path}"


def test_preamble_loads_verbatim():
    text = utils.load_prompt(REPO_ROOT / "prompts" / "sdf" / "preamble.txt")
    assert text.strip()


def test_reasoning_library_loads_with_expected_layers():
    """The library CSV is step 2's source of truth: every entry carries the
    schema columns, and all three layers (conduct C*, core moves M*, topic T*)
    are present. Counts are asserted loosely — the library is actively edited."""
    from dad_pipeline import reasoning_library

    library = reasoning_library.load(REPO_ROOT / "prompts" / "dad")
    ids = reasoning_library.all_ids(library)
    assert len(ids) == len(set(ids)), "duplicate entry ids in reasoning_library.csv"
    prefixes = {i[0] for i in ids}
    assert {"C", "M", "T"} <= prefixes
    block = reasoning_library.format_library(library)
    assert all(i in block for i in ids)
    # Every entry is conditional: a non-empty trigger condition per row, and
    # the 2a trigger index carries every id.
    index = reasoning_library.trigger_index_block(library)
    assert all(f"- {i}: " in index for i in ids)
    for entry in reasoning_library.get_entries(library, ids):
        assert entry["trigger_condition"].strip(), f"{entry['id']} has no trigger condition"
