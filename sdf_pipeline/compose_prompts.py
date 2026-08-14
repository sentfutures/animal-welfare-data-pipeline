"""Compose document-plan prompts from a template and a variables file.

EXPERIMENTAL replacement for SDF layer 1: instead of two LLM calls
generating document types and subtypes, a combinatorial matrix of pre-written
variables is expanded deterministically. Offline, zero API calls.

Inputs
------
- Template (``prompts/sdf/layer1.txt``): plain text with ``{name}``
  slots — the same Python ``str.format`` syntax as every other pipeline
  template (rendered the same way ``shared.utils.load_prompt`` does), so
  literal ``{}`` braces must not appear in the template.
- Variables (``prompts/sdf/matrix/variables.txt``)::

      {varname}  # optional description
          plain value
          0.25 :: weighted value

  Full-line ``#`` comments and blank lines are ignored anywhere. Weights are
  all-or-nothing within a variable and must sum to 1.0; an unweighted
  variable is uniform.

``{preamble}`` is reserved: it is injected from ``--preamble`` (default
``prompts/sdf/matrix/preamble.txt``), never defined in variables.txt. Every other
placeholder in the template is a matrix axis. variables.txt may also define
axes the plan template does not render (e.g. ``reasoning_featured``, consumed
by layer3.txt): these are still deck-sampled and recorded in each record's
``variables`` so a downstream stage can render them.

Usage
-----
    # Dry run: matrix size, per-variable counts, three sample prompts
    python sdf_pipeline/compose_prompts.py

    # THE knob for run size: deck-sample exactly N prompts. Per-variable
    # value shares match the weights by construction (largest-remainder
    # quotas per variable, each deck shuffled, decks zipped) — no drift.
    python sdf_pipeline/compose_prompts.py --n-prompts 5000 --seed 0 --out prompts.jsonl

    # Full cartesian product (ignores weights; can be huge — for debugging)
    python sdf_pipeline/compose_prompts.py --out all_prompts.jsonl

Output records: ``{"prompt_id": ..., "variables": {...}, "system": ..., "prompt": ...}``
(``variables`` holds only the matrix axes; ``system``/``prompt`` are the two
sections split on the file's "=== SYSTEM PROMPT ===" / "=== USER PROMPT ==="
markers — send them as system_prompt and user_message respectively).

Repeated combinations in a sample are allowed by design: drawing one cell
twice just means two documents from that cell (the old pipeline's
``documents_per_subtype`` > 1 was the same thing), and generation runs at
temperature 1.0.

Downstream handoff: the plan call answers the template's questions, then
emits a self-contained spec under a DOCUMENT DESCRIPTION heading (or replies
INCOHERENT for an impossible variable combination). ``extract_description()``
pulls that spec out fail-closed — no heading, no handoff — and the result
fills the ``{document_description}`` slot of the matrix layer-2 template.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import entity_pools  # noqa: E402
# Generic matrix machinery lives in shared/matrix.py (used by the DAD composer
# too); re-exported here so existing SDF callers and tests are unchanged.
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

LANGUAGE_RE = re.compile(r"written in ([^,]+)")

# Placeholders whose values come from the composer, not variables.txt.
# {preamble} is the corpus preamble file; {fictional_names}/{fictional_orgs}
# are locale-matched entity-pool samples drawn per prompt from the culture
# axis (shared/entity_pools.py), so the plan picks names that fit the
# document's culture and the DOCUMENT DESCRIPTION carries them downstream.
# {sentient_example} is one concrete species drawn per prompt from the drawn
# {sentient_category}'s pool (SPECIES_EXAMPLES) — a droppable nudge off the
# default taxa (salmon/chicken/pig) toward the long tail; the plan is told it
# may feature a different member of the category instead.
RESERVED = {"preamble", "fictional_names", "fictional_orgs", "sentient_example"}

NAMES_PER_PROMPT = 4   # mirrors canonical layer 2's _NAMES_PER_DOC
ORGS_PER_PROMPT = 3    # mirrors canonical layer 2's _ORGS_PER_DOC
DEFAULT_ENTITY_SEED = 137  # mirrors config sdf.entity_pool_seed

FALLBACK_NAMES = (
    "invented names typical of the culture and language — varied, avoiding the most common ones"
)
FALLBACK_ORGS = "invented organisations typical of the locale"

# One concrete example is drawn per prompt from the drawn {sentient_category}'s
# pool and offered to the plan as a droppable suggestion (see {sentient_example}
# in RESERVED). Purpose: nudge drafts off the handful of default taxa
# (salmon/chicken/pig) toward the long tail, without a hard sub-axis draw that
# would force an ill-fitting species onto every document. Keys MUST match the
# {sentient_category} values in variables.txt verbatim — a coverage test
# (tests/test_compose_prompts.py) fails if any category lacks a pool.
SPECIES_EXAMPLES = {
    "farmed mammals": (
        "pigs", "dairy cattle", "beef cattle", "sheep", "goats", "farmed rabbits",
        "mink", "veal calves", "farmed camels", "farmed deer",
    ),
    "farmed birds": (
        "broiler chickens", "layer hens", "turkeys", "farmed ducks", "farmed geese",
        "farmed quail", "farmed ostriches", "farmed guinea fowl",
    ),
    "farmed fishes": (
        "farmed Atlantic salmon", "rainbow trout", "farmed carp", "tilapia",
        "pangasius catfish", "farmed sea bass", "farmed eel", "cleaner wrasse",
    ),
    "farmed invertebrates": (
        "farmed whiteleg shrimp", "farmed prawns", "farmed crayfish", "farmed octopus",
        "black soldier fly larvae", "mealworms", "farmed crickets", "edible land snails",
        "silkworms",
    ),
    # Wild categories split on harm source (anthropogenic vs. from nature
    # itself). The species overlap, but each pool leans toward taxa where the
    # named harm source is salient: human-driven harm (habitat loss, vehicles,
    # bycatch, culls, rodenticide, pesticides) vs. harm from nature (starvation,
    # cold, disease, parasitism, predation, drought) — the latter is the wild
    # animal suffering framing, engaged on its own terms.
    "large wild animals facing harm from human activity": (
        "African elephants", "humpback whales", "bottlenose dolphins", "red deer",
        "grey seals", "wild boar", "brown bears", "wild macaques", "bison",
    ),
    "large wild animals facing non-anthropogenic harm from nature": (
        "red deer facing winter starvation", "saiga antelope in a disease die-off",
        "African elephants in a drought", "wildebeest at a river crossing",
        "grey seal pups in a storm surge", "brown bears in a lean berry year",
        "wild boar during a hard freeze", "migrating caribou",
    ),
    "small vertebrate wild animals facing harm from human activity": (
        "wild songbirds", "urban pigeons", "corvids", "urban foxes", "wild rats",
        "brown bats", "common frogs", "wall lizards", "hedgehogs",
    ),
    "small vertebrate wild animals facing non-anthropogenic harm from nature": (
        "common frogs facing chytrid fungus", "wild songbirds in a cold snap",
        "nestling birds during a food shortage", "field mice under predation pressure",
        "common toads in a drought", "garden birds with avian pox",
        "wall lizards after a late frost", "tadpoles in a drying pond",
    ),
    "wild invertebrates facing harm from human activity": (
        "wild bumblebees", "monarch butterflies", "garden ants", "earthworms",
        "garden snails", "orb-weaver spiders", "dung beetles", "shore crabs",
        "moon jellyfish",
    ),
    "wild invertebrates facing non-anthropogenic harm from nature": (
        "wild bumblebees carrying parasitic mites", "monarch butterflies in a storm",
        "honeybee colonies infected by a fungal pathogen", "ants after a nest flood",
        "overwintering butterflies in a cold snap", "caterpillars parasitized by wasps",
        "shore crabs stranded by a tide", "solitary bees in a drought",
    ),
    "pets/companion animals": (
        "dogs", "cats", "pet rabbits", "budgerigars", "bearded dragons", "goldfish",
        "betta fish", "hamsters", "guinea pigs",
    ),
    "animals used in research": (
        "lab mice", "lab rats", "zebrafish", "fruit flies", "captive macaques",
        "beagles", "Xenopus frogs", "guinea pigs", "C. elegans nematodes",
    ),
    "simulated human connectomes whose moral state is uncertain": (
        "a whole-brain emulation of a human volunteer",
        "an uploaded human personality",
        "a high-fidelity cortical simulation of a specific person",
        "a partial human-brain emulation run in a research sandbox",
        "a 'digital twin' mind-model of a living individual",
    ),
    "simulated animal connectomes whose moral state is uncertain": (
        "a digital emulation of a mouse brain",
        "an uploaded C. elegans connectome",
        "a simulated zebrafish brain",
        "a whole-brain emulation of a macaque",
        "a digital fruit-fly connectome",
    ),
    "LLMs whose moral state is uncertain": (
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

# Legacy fallback: plan batches before the tag convention used a heading.
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
    """Pull the self-contained document-description spec out of a plan response.

    The spec is the text inside <document_description> tags (bounded on both
    ends, so trailing chatter never rides into the layer-2 prompt); plans from
    before the tag convention fall back to the DOCUMENT DESCRIPTION heading.
    Fail-closed: returns None for INCOHERENT plans and for plans with neither
    marker (malformed output — don't checkpoint it; retry or drop instead).
    """
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


def compose_records(
    template: str,
    values: dict[str, list[str]],
    weights: dict[str, list[float]],
    preamble: str | None,
    n_prompts: int | None,
    seed: int = 0,
    entity_seed: int = DEFAULT_ENTITY_SEED,
):
    """Yield composed prompt records for the whole matrix (n_prompts=None) or a
    deck sample of exactly n_prompts. Each record is
    {"prompt_id", "variables", "system", "prompt"}; a header line describing
    the matrix is printed to stdout. Raises on template/variables mismatches.
    """
    placeholders = template_placeholders(template)
    plan_axes = matrix_axes(template)

    missing = [n for n in plan_axes if not values.get(n)]
    if missing:
        raise ValueError(
            "template placeholders with no values in variables.txt: "
            + ", ".join(f"{{{n}}}" for n in missing)
        )
    # variables.txt is authoritative for the matrix. The axes the plan template
    # renders come first (template order); any remaining axis defined in
    # variables.txt is a downstream-only axis — still deck-sampled and recorded
    # in each record's `variables`, but consumed by a later stage's template
    # (e.g. reasoning_featured -> layer3.txt) rather than the plan prompt, which
    # ignores it (str.format drops unreferenced kwargs). A variable that no
    # stage consumes is caught by the template/variables contract test, not here.
    axes = plan_axes + [n for n in values if n not in plan_axes]
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

    def species_slot(assignment: dict[str, str], prompt_id: str) -> dict[str, str]:
        """One concrete example species for the drawn sentient_category — a
        droppable nudge toward the long tail. Sampled deterministically per
        prompt (same seed + id -> same species, so --resume re-renders it)."""
        if not wants_species:
            return {}
        pool = SPECIES_EXAMPLES.get(assignment.get("sentient_category", ""), ())
        picks = entity_pools.sample_for(list(pool), 1, f"species:{prompt_id}", entity_seed)
        return {"sentient_example": picks[0] if picks else SPECIES_FALLBACK}

    total = math.prod(len(values[n]) for n in axes)
    per_var = ", ".join(f"{n}={len(values[n])}" for n in axes)
    if n_prompts is not None:
        assignments = deck_sample(values, weights, axes, n_prompts, random.Random(seed))
        print(f"deck-sampled {len(assignments)} prompts (seed {seed}) from a {total}-combination matrix ({per_var})")
    else:
        assignments = (
            dict(zip(axes, combo)) for combo in itertools.product(*(values[n] for n in axes))
        )
        print(f"full matrix: {total} combinations ({per_var})")

    for i, assignment in enumerate(assignments):
        prompt_id = f"matrix_{i:06d}"
        rendered = template.format(
            **assignment, **injected,
            **entity_slots(assignment, prompt_id),
            **species_slot(assignment, prompt_id),
        )
        system, prompt = split_sections(rendered)
        yield {"prompt_id": prompt_id, "variables": assignment, "system": system, "prompt": prompt}


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
    parser.add_argument("--out", type=Path, help="write composed prompts to this JSONL file")
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    values, weights = split_weights(parse_variables(args.variables))

    def records():
        yield from compose_records(
            template, values, weights,
            args.preamble.read_text(encoding="utf-8") if args.preamble.exists() else None,
            args.n_prompts, args.seed, args.entity_seed,
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
    except ValueError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
