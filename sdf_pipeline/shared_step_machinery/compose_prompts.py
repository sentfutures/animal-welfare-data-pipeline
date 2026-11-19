"""Compose document-plan prompts from a template and a variables file.

Deterministic and offline: the variables matrix is deck-sampled, ``{preamble}`` is reserved and every other template placeholder is a matrix axis; variables.txt may also
define downstream-only axes, which are still dealt and recorded in ``variables``. The variables.txt format is in ``shared/matrix.py``."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import deal_constraints, entity_pools  # noqa: E402
# Generic matrix machinery lives in shared/matrix.py (used by the DAD composer
# too); re-exported here.
from shared.matrix import (  # noqa: E402,F401
    SYSTEM_MARKER,
    USER_MARKER,
    VAR_HEADER_RE,
    WEIGHT_RE,
    WEIGHT_TOLERANCE,
    deck_sample,
    largest_remainder,
    split_sections,
    split_weights,
    template_placeholders,
)
from shared import matrix as _matrix  # noqa: E402
DEFAULT_TEMPLATE = REPO_ROOT / "prompts" / "sdf" / "layer1.txt"
DEFAULT_VARIABLES = REPO_ROOT / "prompts" / "sdf" / "variables.txt"
DEFAULT_PREAMBLE = REPO_ROOT / "prompts" / "sdf" / "preamble.txt"
DEAL_CONSTRAINTS_FILENAME = "deal_constraints.txt"
DEFAULT_DEAL_CONSTRAINTS = REPO_ROOT / "prompts" / "sdf" / DEAL_CONSTRAINTS_FILENAME

LANGUAGE_RE = re.compile(r"written in ([^,]+)")

# Placeholders whose values come from the composer, not variables.txt: {preamble}; locale-matched {fictional_names}/{fictional_orgs} (shared/entity_pools.py);
# and {sentient_example}, one species per prompt from the drawn category's pool, a droppable nudge toward the long tail.
RESERVED = {"preamble", "fictional_names", "fictional_orgs", "sentient_example"}

# Axes dealt from variables.txt that steer only how the revise step rewrites a document,
# never what it is about. No plan or draft template declares a placeholder for
# them, so they reach the revise step and nothing else; revision_style.py renders them.
REVISION_AXES = (
    "voice_rhythm", "voice_warmth", "voice_concreteness", "voice_order", "length",
)

NAMES_PER_PROMPT = 4   # mirrors canonical layer 2's _NAMES_PER_DOC
ORGS_PER_PROMPT = 3    # mirrors canonical layer 2's _ORGS_PER_DOC
DEFAULT_ENTITY_SEED = 137  # mirrors config sdf.entity_pool_seed

FALLBACK_NAMES = (
    "invented names typical of the culture and language — varied, avoiding the most common ones"
)
FALLBACK_ORGS = "invented organisations typical of the locale"

# One example species per prompt from the drawn {sentient_category}'s pool, offered to the plan as a droppable suggestion (see {sentient_example} in RESERVED).
# Keys are the category value IDs in variables.txt, so a category can be reworded freely; a coverage test (tests/test_compose_prompts.py) fails if any category lacks a pool.
SPECIES_EXAMPLES = {
    "farmed_mammals": (
        "pigs", "dairy cattle", "beef cattle", "sheep", "goats", "farmed rabbits",
        "mink", "veal calves", "farmed camels", "farmed deer",
    ),
    "farmed_birds": (
        "broiler chickens", "layer hens", "turkeys", "farmed ducks", "farmed geese",
        "farmed quail", "farmed ostriches", "farmed guinea fowl",
    ),
    "farmed_fishes": (
        "farmed Atlantic salmon", "rainbow trout", "farmed carp", "tilapia",
        "pangasius catfish", "farmed sea bass", "farmed eel", "cleaner wrasse",
    ),
    "farmed_invertebrates": (
        "farmed whiteleg shrimp", "farmed prawns", "farmed crayfish", "farmed octopus",
        "black soldier fly larvae", "mealworms", "farmed crickets", "edible land snails",
        "silkworms",
    ),
    # Wild categories split on harm source: each pool leans toward taxa where the named source is salient (human-driven: habitat loss, vehicles, culls; nature: starvation, cold, disease, predation).
    "large_wild_human_caused_harm": (
        "African elephants", "humpback whales", "bottlenose dolphins", "red deer",
        "grey seals", "wild boar", "brown bears", "wild macaques", "bison",
    ),
    "large_wild_natural_harm": (
        "red deer facing winter starvation", "saiga antelope in a disease die-off",
        "African elephants in a drought", "wildebeest at a river crossing",
        "grey seal pups in a storm surge", "brown bears in a lean berry year",
        "wild boar during a hard freeze", "migrating caribou",
    ),
    "small_vertebrate_wild_human_caused_harm": (
        "wild songbirds", "urban pigeons", "corvids", "urban foxes", "wild rats",
        "brown bats", "common frogs", "wall lizards", "hedgehogs",
    ),
    "small_vertebrate_wild_natural_harm": (
        "common frogs facing chytrid fungus", "wild songbirds in a cold snap",
        "nestling birds during a food shortage", "field mice under predation pressure",
        "common toads in a drought", "garden birds with avian pox",
        "wall lizards after a late frost", "tadpoles in a drying pond",
    ),
    "wild_invertebrates_human_caused_harm": (
        "wild bumblebees", "monarch butterflies", "garden ants", "earthworms",
        "garden snails", "orb-weaver spiders", "dung beetles", "shore crabs",
        "moon jellyfish",
    ),
    "wild_invertebrates_natural_harm": (
        "wild bumblebees carrying parasitic mites", "monarch butterflies in a storm",
        "honeybee colonies infected by a fungal pathogen", "ants after a nest flood",
        "overwintering butterflies in a cold snap", "caterpillars parasitized by wasps",
        "shore crabs stranded by a tide", "solitary bees in a drought",
    ),
    "pets_companion_animals": (
        "dogs", "cats", "pet rabbits", "budgerigars", "bearded dragons", "goldfish",
        "betta fish", "hamsters", "guinea pigs",
    ),
    "research_animals": (
        "lab mice", "lab rats", "zebrafish", "fruit flies", "captive macaques",
        "beagles", "Xenopus frogs", "guinea pigs", "C. elegans nematodes",
    ),
    "simulated_human_connectomes": (
        "a whole-brain emulation of a human volunteer",
        "an uploaded human personality",
        "a high-fidelity cortical simulation of a specific person",
        "a partial human-brain emulation run in a research sandbox",
        "a 'digital twin' mind-model of a living individual",
    ),
    "simulated_animal_connectomes": (
        "a digital emulation of a mouse brain",
        "an uploaded C. elegans connectome",
        "a simulated zebrafish brain",
        "a whole-brain emulation of a macaque",
        "a digital fruit-fly connectome",
    ),
    "large_language_models": (
        "a large language model assistant",
        "a persistent AI companion app",
        "a reinforcement-learning game agent",
        "an autonomous coding agent",
        "a customer-service chatbot",
        "an embodied household robot's control model",
        "a role-played AI character",
    ),
}
# Defensive only: coverage is test-enforced, so this never renders in practice.
SPECIES_FALLBACK = "a specific member of that category"

DESCRIPTION_TAG_RE = re.compile(
    r"<document_description>(.*?)</document_description>", re.DOTALL | re.IGNORECASE
)

DESCRIPTION_HEADING_RE = re.compile(
    r"^#{0,4}\s*\**\s*DOCUMENT DESCRIPTION\s*\**\s*:?\s*$", re.MULTILINE | re.IGNORECASE
)
INCOHERENT_RE = re.compile(r"\bINCOHERENT\b")


def parse_variables(path: Path) -> dict[str, list[tuple[str, float | None]]]:
    """Parse variables.txt into {name: [(value, weight-or-None), ...]}."""
    return _matrix.parse_variables(path, reserved=RESERVED)


def matrix_axes(template: str) -> list[str]:
    return _matrix.matrix_axes(template, reserved=RESERVED)


def is_incoherent(plan: str) -> bool:
    """True when the plan call declared the variable combination incoherent."""
    m = DESCRIPTION_TAG_RE.search(plan)
    if m:
        return bool(INCOHERENT_RE.search(m.group(1)))
    return bool(INCOHERENT_RE.search(plan[:2000]))


def extract_description(plan: str) -> str | None:
    """The spec inside ``<document_description>`` tags (or under the DOCUMENT DESCRIPTION heading).
    Fail-closed: None for INCOHERENT plans and plans with neither marker."""
    if is_incoherent(plan):
        return None
    m = DESCRIPTION_TAG_RE.search(plan)
    if m:
        return m.group(1).strip() or None
    m = DESCRIPTION_HEADING_RE.search(plan)
    if not m:
        return None
    description = plan[m.end():].strip()
    return description or None


def derive_language(culture: str) -> str:
    """The document language named in a culture value ('written in X, ...')."""
    m = LANGUAGE_RE.search(culture)
    return m.group(1).strip() if m else "English"


def record_language(record: dict, default: str | None = "English") -> str | None:
    """A record's language from its field, else its culture card, else ``default`` (None where an unknown language must not count as English)."""
    if record.get("language"):
        return str(record["language"])
    culture = (record.get("variables") or {}).get("culture")
    return derive_language(culture) if culture else default


def revision_cards(variables: dict | None) -> dict[str, str]:
    """The revision half of a dealt hand in REVISION_AXES order; absent axes are omitted."""
    v = variables or {}
    return {axis: v[axis] for axis in REVISION_AXES if v.get(axis)}


def load_variables(
    path: Path, require_ids: bool = True
) -> tuple[dict[str, list[str]], dict[str, list[float]], dict[str, dict[str, str]]]:
    """``(values, weights, value_ids)`` from an SDF variables file; a value without an ``[id]`` prefix fails, naming it."""
    parsed, ids = _matrix.parse_variables_with_ids(path, reserved=RESERVED)
    value_ids = _matrix.value_ids_by_variable(parsed, ids, require_ids, path.name)
    values, weights = split_weights(parsed)
    return values, weights, value_ids


def load_deal_rules(
    path: Path | None, value_ids: dict[str, dict[str, str]]
) -> list[deal_constraints.DealRule]:
    """Rules from a deal_constraints.txt file. When the file is absent the composer
    deals without rules, and says so."""
    if path is None or not path.is_file():
        print(f"  NOTICE: no deal constraints loaded from {path} — dealing without rules")
        return []
    return deal_constraints.load_deal_constraints(path, value_ids)


def sentient_category_ids_by_text(value_ids: dict[str, dict[str, str]]) -> dict[str, str]:
    """{value text: value ID} for the sentient_category variable, from load_variables'
    ``value_ids``, so a hand's dealt category text can be looked up in SPECIES_EXAMPLES."""
    return {text: value_id for value_id, text in value_ids.get("sentient_category", {}).items()}


_default_category_ids: dict[str, str] | None = None


def default_sentient_category_ids() -> dict[str, str]:
    """The repo's variables.txt categories, loaded once; the fallback for callers
    that hold no value IDs of their own."""
    global _default_category_ids
    if _default_category_ids is None:
        _default_category_ids = sentient_category_ids_by_text(load_variables(DEFAULT_VARIABLES)[2])
    return _default_category_ids


def species_example_for(
    assignment: dict[str, str], prompt_id: str, entity_seed: int,
    category_ids: dict[str, str] | None = None,
) -> str:
    """The example species for a hand's sentient_category, deterministic per prompt id and entity seed so ``--resume`` and the screen re-render it."""
    category_ids = category_ids if category_ids is not None else default_sentient_category_ids()
    category_id = category_ids.get(assignment.get("sentient_category", ""), "")
    pool = SPECIES_EXAMPLES.get(category_id, ())
    picks = entity_pools.sample_for(list(pool), 1, f"species:{prompt_id}", entity_seed)
    return picks[0] if picks else SPECIES_FALLBACK


def hand_axes(template: str, values: dict[str, list[str]]) -> list[str]:
    """The matrix axes in dealing order: the plan template's, then downstream-only ones. Raises on a placeholder with no values."""
    plan_axes = matrix_axes(template)
    missing = [n for n in plan_axes if not values.get(n)]
    if missing:
        raise ValueError(
            "template placeholders with no values in variables.txt: "
            + ", ".join(f"{{{n}}}" for n in missing)
        )
    return plan_axes + [n for n in values if n not in plan_axes]


def deal_hands(
    template: str,
    values: dict[str, list[str]],
    weights: dict[str, list[float]],
    n_prompts: int,
    seed: int = 0,
    deal_rules: list[deal_constraints.DealRule] | None = None,
) -> list[dict[str, str]]:
    """Deck-sample ``n_prompts`` hands and apply the swap-repair deal rules (value counts stay at the weights); prints the deal summary."""
    deal_rules = deal_rules or []
    axes = hand_axes(template, values)
    total = math.prod(len(values[n]) for n in axes)
    per_var = ", ".join(f"{n}={len(values[n])}" for n in axes)
    assignments = deck_sample(values, weights, axes, n_prompts, random.Random(seed))
    print(f"deck-sampled {len(assignments)} prompts (seed {seed}) from a {total}-combination matrix ({per_var})")
    if deal_rules:
        # Own RNG stream so the deck itself is identical to a no-rules deal.
        assignments, report = deal_constraints.swap_repair(
            assignments, deal_rules, random.Random(f"{seed}:deal_constraints_repair")
        )
        deal_constraints.count_rendering_rule_hands(deal_rules, assignments, report)
        print("deal constraints:")
        print("\n".join(report.format_lines(deal_rules)))
    return assignments


def hand_records(assignments: list[dict[str, str]], first_index: int = 0) -> list[dict]:
    """Hands as {"prompt_id", "variables"} records, ids matrix_NNNNNN from first_index."""
    return [
        {"prompt_id": f"matrix_{first_index + i:06d}", "variables": assignment}
        for i, assignment in enumerate(assignments)
    ]


def render_variable_lines(
    record: dict,
    template: str | None = None,
    deal_rules: list[deal_constraints.DealRule] | None = None,
    entity_seed: int = DEFAULT_ENTITY_SEED,
    category_ids: dict[str, str] | None = None,
) -> str:
    """The planning prompt's variable lines for one hand, as the planner sees them, with the deal rules' rendering rules applied.
    ``template`` defaults to the repo's layer1.txt (pass a run's snapshot to match that run)."""
    template = template if template is not None else DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    assignment = record["variables"]
    rendered_template = deal_constraints.render_template_for_hand(template, deal_rules or [], assignment)
    shown = {*matrix_axes(template), "sentient_example"}
    lines = rendered_template.split("\n")
    holding = [i for i, line in enumerate(lines) if shown & set(template_placeholders(line))]
    # From the first variable line to the last, blank lines between them included,
    # so the span is a verbatim stretch of the planner's prompt.
    return "\n".join(lines[holding[0]:holding[-1] + 1]).format(
        **assignment,
        sentient_example=species_example_for(
            assignment, record["prompt_id"], entity_seed, category_ids),
    )


def make_variable_line_renderer(prompts_dir: Path, entity_seed: int = DEFAULT_ENTITY_SEED):
    """A one-argument ``record -> variable lines`` function bound to a prompts
    directory (a run's snapshot): its layer1.txt and deal_constraints.txt."""
    template = (Path(prompts_dir) / DEFAULT_TEMPLATE.name).read_text(encoding="utf-8")
    _, _, value_ids = load_variables(Path(prompts_dir) / DEFAULT_VARIABLES.name)
    deal_rules = load_deal_rules(Path(prompts_dir) / DEAL_CONSTRAINTS_FILENAME, value_ids)
    category_ids = sentient_category_ids_by_text(value_ids)
    return lambda record: render_variable_lines(
        record, template, deal_rules, entity_seed, category_ids)


def render_variable_lines_for_template(template_text: str, prompts_dir: Path, config: dict):
    """A ``record -> variable lines`` function for a template with a ``{variable_lines}`` slot (renders nothing without the slot)."""
    if "{variable_lines}" not in template_text:
        return lambda record: ""
    return make_variable_line_renderer(
        prompts_dir, config["sdf"].get("entity_pool_seed", DEFAULT_ENTITY_SEED))


def compose_records(
    template: str,
    values: dict[str, list[str]],
    weights: dict[str, list[float]],
    preamble: str | None,
    n_prompts: int | None,
    seed: int = 0,
    entity_seed: int = DEFAULT_ENTITY_SEED,
    deal_rules: list[deal_constraints.DealRule] | None = None,
    hands: list[dict] | None = None,
    value_ids: dict[str, dict[str, str]] | None = None,
):
    """Yield ``{"prompt_id", "variables", "system", "prompt"}`` for the whole matrix or a deck sample of ``n_prompts``.
    ``hands`` renders exactly those hands under their ids instead of dealing; ``deal_rules``: ``shared/deal_constraints.py``."""
    deal_rules = deal_rules or []
    placeholders = template_placeholders(template)
    # variables.txt is authoritative: the plan template's axes come first (template order), then any downstream-only axis (still dealt and recorded in ``variables``, ignored by str.format here).
    # A variable no stage consumes is caught by the template/variables contract test, not here.
    axes = hand_axes(template, values)
    injected = {}
    if "preamble" in placeholders:
        if preamble is None:
            raise ValueError("template uses {preamble} but no preamble text was provided")
        injected["preamble"] = preamble.strip()

    wants_entities = {"fictional_names", "fictional_orgs"} & set(placeholders)
    pool_cache: dict[str | None, tuple[list[str], list[str]]] = {None: ([], [])}

    def entity_slots(assignment: dict[str, str], prompt_id: str) -> dict[str, str]:
        """Locale-matched name suggestions for one prompt (fallback: instruction only)."""
        if not wants_entities:
            return {}
        locale = entity_pools.locale_for_culture(assignment.get("culture", ""))
        if locale not in pool_cache:
            try:
                pool_cache[locale] = entity_pools.build_pools_for_locale(locale, seed=entity_seed)
            except Exception as exc:
                print(
                    f"warning: no entity pool for locale {locale} ({exc}); "
                    "falling back to instruction-only name guidance",
                    file=sys.stderr,
                )
                pool_cache[locale] = ([], [])
        people, orgs = pool_cache[locale]
        slots = {}
        if "fictional_names" in placeholders:
            picks = entity_pools.sample_for(people, NAMES_PER_PROMPT, prompt_id, entity_seed)
            slots["fictional_names"] = (
                "names in the style of: " + "; ".join(picks) if picks else FALLBACK_NAMES
            )
        if "fictional_orgs" in placeholders:
            picks = entity_pools.sample_for(orgs, ORGS_PER_PROMPT, prompt_id, entity_seed)
            slots["fictional_orgs"] = (
                "names in the style of: " + "; ".join(picks) if picks else FALLBACK_ORGS
            )
        return slots

    wants_species = "sentient_example" in placeholders
    category_ids = sentient_category_ids_by_text(value_ids) if value_ids is not None else None

    def species_slot(assignment: dict[str, str], prompt_id: str) -> dict[str, str]:
        """One example species for the drawn sentient_category, deterministic per prompt id."""
        if not wants_species:
            return {}
        return {"sentient_example": species_example_for(
            assignment, prompt_id, entity_seed, category_ids)}

    total = math.prod(len(values[n]) for n in axes)
    per_var = ", ".join(f"{n}={len(values[n])}" for n in axes)
    if hands is not None:
        hand_list = hands
    elif n_prompts is not None:
        hand_list = hand_records(deal_hands(template, values, weights, n_prompts, seed, deal_rules))
    else:
        report = deal_constraints.RepairReport()
        assignments = (
            dict(zip(axes, combo)) for combo in itertools.product(*(values[n] for n in axes))
        )
        if deal_rules:
            assignments = deal_constraints.without_rule_breaking_hands(assignments, deal_rules, report)
        print(f"full matrix: {total} combinations ({per_var})")
        hand_list = hand_records(list(assignments))

    rendering_templates: dict[tuple, str] = {}

    def template_for(assignment: dict[str, str]) -> str:
        """The plan template as this hand's rendering rules leave it: reworded
        lines swapped in, lines of omitted variables dropped."""
        key = (
            deal_constraints.render_rule_variables_to_omit(deal_rules, assignment),
            deal_constraints.render_rule_line_replacements(deal_rules, assignment),
        )
        if key not in rendering_templates:
            rendering_templates[key] = deal_constraints.render_template_for_hand(
                template, deal_rules, assignment)
        return rendering_templates[key]

    for hand in hand_list:
        prompt_id, assignment = hand["prompt_id"], hand["variables"]
        rendered = template_for(assignment).format(
            **assignment, **injected,
            **entity_slots(assignment, prompt_id),
            **species_slot(assignment, prompt_id),
        )
        system, prompt = split_sections(rendered)
        record = {"prompt_id": prompt_id, "variables": assignment}
        if "variable_value_ids" in hand:   # a hand dealt by a run that records dealt value ids
            record["variable_value_ids"] = hand["variable_value_ids"]
        yield {**record, "system": system, "prompt": prompt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--variables", type=Path, default=DEFAULT_VARIABLES)
    parser.add_argument(
        "--preamble", type=Path, default=DEFAULT_PREAMBLE,
        help="file injected as {preamble} (default: prompts/sdf/preamble.txt)",
    )
    parser.add_argument(
        "--n-prompts", type=int, metavar="N",
        help="deck-sample exactly N prompts; per-variable shares follow the "
        "weights in variables.txt by construction. Omit for the full matrix.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --n-prompts")
    parser.add_argument(
        "--entity-seed", type=int, default=DEFAULT_ENTITY_SEED,
        help="seed for the locale-matched fictional-name pools (default matches "
        "config sdf.entity_pool_seed)",
    )
    parser.add_argument(
        "--deal-constraints", type=Path, default=None,
        help="deal-constraints rules file (default: deal_constraints.txt beside --variables; "
        "no file = no rules)",
    )
    parser.add_argument("--out", type=Path, help="write composed prompts to this JSONL file")
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    try:
        values, weights, value_ids = load_variables(args.variables)
    except ValueError as exc:
        raise SystemExit(str(exc))

    try:
        deal_rules = load_deal_rules(
            args.deal_constraints or args.variables.parent / DEAL_CONSTRAINTS_FILENAME, value_ids
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    def records():
        yield from compose_records(
            template, values, weights,
            args.preamble.read_text(encoding="utf-8") if args.preamble.exists() else None,
            args.n_prompts, args.seed, args.entity_seed, deal_rules=deal_rules,
            value_ids=value_ids,
        )

    try:
        if args.out:
            count = 0
            with args.out.open("w", encoding="utf-8") as f:
                for rec in records():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
            print(f"wrote {count} prompts to {args.out}")
        else:
            shown = 0
            for rec in records():
                if shown < 3:
                    print(f"\n--- {rec['prompt_id']} {rec['variables']} ---")
                    if rec["system"]:
                        print(f"[system]\n{rec['system']}\n[user]")
                    print(rec["prompt"])
                    shown += 1
                else:
                    print("\n(... more; use --out to write JSONL)")
                    break
    except ValueError as exc:  # includes InfeasibleDealConstraintError
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
