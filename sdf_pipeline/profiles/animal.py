"""The animal (sentient-being) welfare profile — the original SDF pipeline.

The example pools were moved verbatim from sdf_pipeline/compose_prompts.py,
so an animal run is byte-identical to a pre-profile run.
"""

NAME = "animal"
PROMPTS_SUBDIR = "sdf"
OUTPUT_SUBDIR = "sdf"

PRINCIPLES_FILENAME = "constitution_principles.csv"
LEGACY_PRINCIPLES_FILENAMES: tuple[str, ...] = ()
FAILURE_MODES_FILENAME = "failure_modes.md"

# One concrete example is drawn per prompt from the drawn {sentient_category}'s
# pool and offered to the plan as a droppable suggestion (see {sentient_example}
# in compose_prompts.RESERVED). Purpose: nudge drafts off the handful of default
# taxa (salmon/chicken/pig) toward the long tail, without a hard sub-axis draw
# that would force an ill-fitting species onto every document. Keys are the
# {sentient_category} value IDs in this profile's variables.txt, so a category
# can be reworded without touching this table — a coverage test
# (tests/test_compose_prompts.py) fails if any category lacks a pool.
CATEGORY_EXAMPLES = {
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
    # Wild categories split on harm source (anthropogenic vs. from nature
    # itself). The species overlap, but each pool leans toward taxa where the
    # named harm source is salient: human-driven harm (habitat loss, vehicles,
    # bycatch, culls, rodenticide, pesticides) vs. harm from nature (starvation,
    # cold, disease, parasitism, predation, drought) — the latter is the wild
    # animal suffering framing, engaged on its own terms.
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
CATEGORY_EXAMPLE_FALLBACK = "a specific member of that category"
