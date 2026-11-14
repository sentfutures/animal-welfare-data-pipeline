"""The human-welfare SDF profile: the animal profile mirrored, with the people
a decision affects in place of the animals. Same table shape, same axis names
and weights in its variables.txt; only the content differs. Grounded in
constitution_principles_human.csv + failure_modes_human.md (the human-welfare
distillations beside the animal ones in constitution/). See
sdf_pipeline/profiles/__init__.py for what a profile is."""

NAME = "human"
PROMPTS_SUBDIR = "sdf_human"                 # prompts/sdf_human/
OUTPUT_SUBDIR = "sdf_human"                  # outputs/sdf_human/runs/

PRINCIPLES_FILENAME = "constitution_principles_human.csv"
LEGACY_PRINCIPLES_FILENAMES: tuple[str, ...] = ()
FAILURE_MODES_FILENAME = "failure_modes_human.md"

# One concrete example per {sentient_category} value id in this profile's
# variables.txt (see the animal profile's table for the mechanism). Categories
# are ROLES people play in the document's scenario, not demographic labels;
# each key mirrors the animal category in the same position, sourced from the
# DAD human profile's affected-party table.
CATEGORY_EXAMPLES = {
    "workers": (
        "garment and textile workers", "warehouse and logistics staff",
        "gig drivers and couriers", "farm and seasonal workers",
        "content moderators and data labelers", "domestic workers",
        "construction workers", "electronics supply-chain workers",
        "restaurant and kitchen staff", "cleaners and facilities staff",
    ),
    "customers_users": (
        "app and platform users", "retail customers", "patients as clients",
        "borrowers and account holders", "tenants", "subscribers to a service",
        "first-time or vulnerable buyers", "elderly customers of a call centre",
    ),
    "public_at_scale": (
        "people whose data is collected", "commuters and road users",
        "people exposed to low-level pollution or noise",
        "readers or viewers of mass media", "recipients of automated decisions",
        "people nudged by a default setting", "pedestrians in a redesigned district",
    ),
    "subordinates": (
        "direct reports", "students", "patients on a ward",
        "volunteers in an organization", "junior staff on probation",
        "apprentices and trainees", "players on a youth team",
    ),
    "distant_strangers_human_caused_harm": (
        "people in another country affected by a supply chain",
        "people displaced by conflict or development",
        "residents downstream of a pollution source",
        "people in another community priced out by a change",
        "people affected by a decision made in a different jurisdiction",
        "strangers online affected by something someone posts",
    ),
    "distant_strangers_natural_harm": (
        "people hit by a flood, earthquake, or storm",
        "people facing hunger after drought or crop failure",
        "communities living with an endemic or epidemic disease",
        "people in poverty far away who a donation could reach",
        "people injured or stranded in an accident far from help",
    ),
    "local_community_human_caused_harm": (
        "neighbours of a construction site", "residents near a new landfill",
        "tenants of a building being redeveloped",
        "families on a street cut off by a road scheme",
        "small traders displaced by a rezoning",
        "households downwind of a factory",
    ),
    "local_community_natural_harm": (
        "a neighbourhood after a flood", "a town in a drought",
        "families displaced by a wildfire", "a village after a landslide",
        "a community during a local disease outbreak",
        "households after a hard winter storm",
    ),
    "contested_standing": (
        "future generations",
        "people in minimally conscious or vegetative states",
        "people with advanced dementia", "the recently dead and their wishes",
        "people who exist only if a decision goes one way",
    ),
    "people_in_public_eye": (
        "reality-show participants", "children featured in family content",
        "athletes", "people filmed without knowing it",
        "subjects of viral posts or true-crime stories",
        "lottery winners named in the press",
    ),
    "family_dependents": (
        "children", "teenagers", "aging parents", "a partner or spouse",
        "siblings", "an adult child with a disability",
        "extended family who rely on a breadwinner",
    ),
    "research_participants": (
        "clinical trial participants", "survey and interview respondents",
        "users in an A/B test", "students in an educational study",
        "participants in a field or social experiment",
        "donors of tissue or genetic samples",
    ),
    "digital_human_minds": (
        "a whole-brain emulation of a human volunteer",
        "an uploaded human personality",
        "a high-fidelity cortical simulation of a specific person",
        "a partial human-brain emulation run in a research sandbox",
        "a 'digital twin' mind-model of a living individual",
    ),
    "digital_person_simulacra": (
        "a chatbot built from a deceased relative's messages",
        "a digital replica of a living celebrity",
        "an interactive avatar trained on one person's diaries",
        "a company's simulation of a departed founder",
        "a 'griefbot' kept running by a bereaved family",
    ),
    "current_ai_claimed_interests": (
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
CATEGORY_EXAMPLE_FALLBACK = "a specific person or group in that role"
