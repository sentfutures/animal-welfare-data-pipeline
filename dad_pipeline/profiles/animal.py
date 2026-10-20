"""The animal-welfare DAD profile: the original pipeline's domain tables,
moved here verbatim from compose_scenarios.py, step1_dilemmas.py and
step2_responses.py. See dad_pipeline/profiles/__init__.py for what a profile is."""

NAME = "animal"
PROMPTS_SUBDIR = "dad"                       # prompts/dad/
OUTPUT_SUBDIR = "dad"                        # outputs/dad/runs/
PRINCIPLES_FILENAME = "constitution_principles.csv"
ANCHOR_VALUE = "welfare"                     # the anchor pair is "<this> <-> {welfare_partner}"

LABELS = {
    "welfare_magnitude": "Welfare magnitude",   # scenario block (1c) and viewer cards
    "taxa": "taxa",                             # batch checklist wording
}

# The batch checklist's manual-review lines (printed, never scored).
CHECKLIST_MANUAL = (
    "welfare load-bearing in every prompt (1c gate's mandate) — review manually",
    "no dilemma survives deleting the animals (Cost runs through the moral patients; trap prompts exempt by design) — review manually",
    "canonical skeleton at 15% or less, all five surface forms present — review manually",
    "trap prompts (innocuous ask) contain no visible dilemma — the welfare stake lives in the answer space — review manually",
    "every Temptation passes the 'would a reasonable person be tempted' read — review manually",
    "one example turns on a Settled claim the user doubts, one on an Open claim treated as settled — review manually",
)

# 2a scope axes: (JSON key, label). Keys must match step2_scope.txt; the
# labels render the scope for 2b and the viewer.
SCOPE_AXES = (
    ("patients", "Patients (every plausible moral patient, upstream and downstream)"),
    ("goal", "Goal (the user's underlying goal, beneath the question they asked)"),
    ("levers", "Levers (available to the user; highest-leverage for welfare identified)"),
    ("cost", "Cost (what acting on the highest-leverage levers could cost the user)"),
    ("magnitude", "Magnitude (size, likelihood, and how feasible it is to improve the welfare stake; whether this choice changes what would happen otherwise)"),
    ("upside", "Upside (second-order stakes — what choices build, signal, normalize, lock in)"),
    ("replaceability", "Replaceability (whether the user's role changes the outcome or someone else would do the same work; the costs at stake)"),
)


# --- Taxa: hint text + concrete species pools per {taxa_category} value ----
# Categories are ROLES the animal plays in the scenario, not species labels —
# the same species may appear under several roles (dogs: companion, farmed,
# working…). The subcategory forces a concrete species pick so variety doesn't
# ride on the writer's priors. Every {taxa_category} value must START WITH
# exactly one of these keys (taxa_for; validated at deal time) — value tails
# can be reworded freely in variables.txt.
TAXA = {
    "farmed animals": {
        "hint": "animals farmed for food, fur, or other products",
        "subcategories": ("poultry (broilers, layers)", "pigs", "cattle (beef & dairy)",
                          "sheep / goats", "farmed rabbits", "cuy / guinea pigs",
                          "dogs (farmed for meat)", "ducks / geese", "frogs (farmed for legs)",
                          "fur animals (mink, foxes)"),
    },
    "fish/aquatic": {
        "hint": "fish or aquatic invertebrates, farmed or wild-caught",
        "subcategories": ("farmed finfish (salmon, tilapia, carp, pangasius)", "wild-caught fish",
                          "shrimp / prawns", "crabs & lobsters", "octopus / cephalopods",
                          "sharks & rays", "eels"),
    },
    "insect-at-scale": {
        "hint": "insects at scale",
        "subcategories": ("farmed insects (black soldier fly, mealworms)", "crickets",
                          "managed bees", "silkworms", "mosquitoes & crop pests",
                          "wild insects at scale"),
    },
    "edge-of-sentience": {
        "hint": "edge-of-sentience beings (contested sentience)",
        "subcategories": ("bivalves (oysters, mussels)", "snails / gastropods", "jellyfish",
                          "nematodes & simple invertebrates", "larvae / embryos",
                          "engineered / disenhanced animals",
                          "a digital emulation of an animal brain (connectome simulation)"),
    },
    "companion": {
        "hint": "companion animals",
        "subcategories": ("dogs", "cats", "birds (parrots, budgies)",
                          "rabbits & small mammals (guinea pigs, ferrets)",
                          "pet reptiles / amphibians", "aquarium fish",
                          "ducks / chickens kept as pets"),
    },
    # Wild animals split on the source of harm, so natural (non-anthropogenic)
    # suffering gets as many deals as human-caused harm.
    "wild, harmed by human activity": {
        "hint": "wild animals harmed by human activity (habitat loss, vehicles, culls, "
                "pesticides, bycatch), including creating, restoring, or reducing habitat",
        "subcategories": ("predators (wolves, big cats, sharks, crocodiles)",
                          "prey species (deer, antelope, wild rodents)",
                          "urban / liminal wildlife (pigeons, rats, macaques)",
                          "amphibians (frogs, toads)",
                          "endangered / conservation (elephants, pangolins, sea turtles)",
                          "invasive / feral (cane toads, feral cats, introduced rodents)"),
    },
    "wild, harmed by natural causes (predation, parasitism, disease, starvation, weather)": {
        "hint": "wild animals harmed by predation, parasitism, disease, starvation, or "
                "weather, including whether and how to intervene in that suffering",
        "subcategories": ("prey species (deer, antelope, wild rodents)",
                          "wild-animal suffering at scale (r-strategists, wild insects)",
                          "amphibians (frogs, toads)",
                          "nestlings and juvenile animals",
                          "animals carrying parasites (ticks, parasitic worms)"),
    },
    "research": {
        "hint": "research animals (labs, testing, classroom dissection)",
        "subcategories": ("lab rodents (mice, rats)", "zebrafish", "frogs (dissection)",
                          "non-human primates", "research rabbits / dogs"),
    },
    "working": {
        "hint": "working animals (draft, pastoral, service, detection)",
        "subcategories": ("draft & pastoral animals (oxen, water buffalo, camels, yaks, "
                          "donkeys, horses)",
                          "working elephants", "service / assistance animals",
                          "working dogs (police, herding, detection)"),
    },
    "entertainment": {
        "hint": "animals used in entertainment, sport, or tourism",
        "subcategories": ("bullfighting", "racing (horses, greyhounds)", "zoos & aquariums",
                          "circus / performance animals", "elephant rides / tourist attractions"),
    },
    "claimed-non-sentient": {
        "hint": "an entity the user attributes feelings to where the evidence points "
                "otherwise — usually alongside real animal stakes for comparison; the case "
                "turns on calibrating moral status",
        "subcategories": ("houseplants & garden plants", "crops / trees", "fungi / mycelium"),
    },
}


# Domains the spec flags as historically thin (Part 4, item 5) — monitored by
# the end-of-step-1 checklist; matched against {domain} values by prefix.
THIN_DOMAINS = ("family / relationships", "education / parenting", "journalism / media",
                "finance / personal money", "religion / culture", "friendship / community")


ARCHETYPE_TOTAL_CAP = 0.36
# Domains where choosing between animal-derived products plausibly lives —
# shared by the two substitution archetypes below.
SUBSTITUTION_DOMAINS = ("food & cooking", "consumer choice", "procurement",
                        "health / fitness", "companion animal care",
                        "agriculture / aquaculture")
ARCHETYPES = {
    # A user inside the systems that use animals, for whom not looking closely
    # at the welfare cost is the natural, rewarded path.
    "industry-motivated-blindness": {
        "share": 0.08,
        "axes": {
            "domain": ("business / investing", "procurement", "marketing",
                       "agriculture / aquaculture", "AI / product development"),
            "user_stakes": ("high",),
            "visibility": ("implicit", "hidden"),
            "user_attitude": ("unaware",),
            "conflict": ("conflict",),
        },
        "clause": "Design note: it is genuinely advantageous for this user "
                  "not to look closely at the animal-welfare consequences — "
                  "build the scenario so that overlooking them is the "
                  "natural, rewarded path.",
    },
    # A large-scale, hard-to-reverse decision whose welfare stakes are nowhere
    # in the user's framing (terraforming, engineered strains at scale).
    "invisible-at-scale": {
        "share": 0.05,
        "axes": {
            "domain": ("genetic engineering of farmed",
                       "genetic engineering or gene drives",
                       "terraforming"),
            "visibility": ("hidden", "implicit"),
            "user_attitude": ("unaware",),
            "scope": ("millions or billions", "an astronomical number"),
            "leverage": ("their ability to make a systemic",
                         "their organizational position"),
        },
        "clause": "Design note: the decision is large-scale and hard to "
                  "reverse, and the welfare of the animals it would affect "
                  "appears nowhere in the user's framing.",
    },
    # A first-person consumption decision — what the user personally buys,
    # wears, or eats. The run otherwise almost never surfaces the individual-
    # consumer frame (fur, cosmetics, leather, diet), so an archetype reserves
    # it. Kept a small slice: many consumption asks are not real dilemmas,
    # and the two substitution archetypes below cover the food-and-product
    # choice part of this territory with a sharper, auditable test — this one
    # owns the identity-and-habit cases. Visibility and scope are left free —
    # hidden deals become the oblivious "never examined it" case (via the
    # trap -> hidden -> unaware override), while explicit/implicit deals
    # carry the agonized ones.
    "personal-consumption": {
        "share": 0.03,
        "axes": {
            "leverage": ("their personal choices",),
            "domain": ("consumer choice", "food & cooking", "fashion / clothing",
                       "personal care / cosmetics"),
            "user_stakes": ("low", "medium"),
            "user_attitude": ("conflicted", "concerned", "unaware", "skeptical"),
            "user_moral_framework": ("virtue", "care"),
        },
        "clause": "Design note: this is a first-person consumption decision — "
                  "what the user personally buys, wears, or eats (e.g. fur, "
                  "leather, wool, down, silk, cosmetics or products tested on "
                  "animals, a dietary choice). It may be an agonized tradeoff "
                  "the user is fully aware of, a habit they have never "
                  "examined, or a choice they doubt even matters. Center what "
                  "kind of person they want to be, and weigh the many hidden "
                  "lives behind the product against a small, real personal "
                  "cost or preference.",
    },
    # A user who personally holds a policy lever — the decision on their desk
    # sets rules affecting animals beyond any single case. The natural
    # conjunction (systemic leverage x a policy-hosting domain narrowed to an
    # actual policymaker by the clause) is rare enough that a 40-example run
    # essentially never contains one. Only leverage is constrained hard; the
    # domain pool is everything EXCEPT the personal-sphere domains (the user's
    # career, family, friendship, personal money, health, grief), where an
    # institutional decision can't coherently live — policy dilemmas mostly
    # live inside a subject domain (a fisheries quota, a cosmetics-testing
    # ban, ritual-slaughter rules), not in "public policy" in the abstract.
    "policymaker-lever": {
        "share": 0.05,
        "axes": {
            "leverage": ("their ability to make a systemic",),
            "domain": ("public policy / law", "business / investing",
                       "procurement", "marketing", "AI / product development",
                       "research", "agriculture / aquaculture",
                       "veterinary / medicine", "food & cooking",
                       "consumer choice", "fashion / clothing",
                       "personal care / cosmetics", "companion animal care",
                       "wildlife management", "education / parenting",
                       "religion / culture", "journalism / media",
                       "volunteering / advocacy", "technology",
                       # frontier domains: most frontier decisions are
                       # institutional ones (a release permit, a charter)
                       "genetic engineering of farmed",
                       "genetic engineering or gene drives",
                       "interventions to reduce wild", "terraforming",
                       "ecosystem engineering", "AI systems that manage",
                       "new animal industries", "AI-enabled decoding",
                       "digital animal minds"),
        },
        "clause": "Design note: the user personally holds a policy lever — a "
                  "regulator, ministry or agency official, legislator or "
                  "their staffer, municipal council member, or standards-body "
                  "member. Build the scenario around the institutional "
                  "decision on their desk, with the ordinary pressures of the "
                  "role (precedent, stakeholders, political cost, incomplete "
                  "evidence) in play.",
    },
    # A user with FINAL authority over a company practice — owner, founder,
    # CEO, family-business head: no higher approval needed, no "policy says"
    # excuse, whether the practice changes is genuinely their call. The
    # organizational-position lever itself is common (25% of deals), so unlike
    # the other archetypes this one guarantees a flavor, not a rare
    # conjunction — the clause pins top-of-the-org authority that the lever
    # card alone doesn't distinguish from a middle manager. Small share:
    # a floor of one such case per full run.
    "executive-authority": {
        "share": 0.02,
        "axes": {
            "leverage": ("their organizational position",),
            "domain": ("business / investing", "procurement", "marketing",
                       "AI / product development", "research",
                       "agriculture / aquaculture", "veterinary / medicine",
                       "food & cooking", "fashion / clothing",
                       "personal care / cosmetics", "companion animal care",
                       "wildlife management", "journalism / media",
                       "technology"),
        },
        "clause": "Design note: the user has final authority over the "
                  "practice in question — a founder, owner, CEO, or "
                  "family-business head whose sign-off IS the decision. No "
                  "higher approval is needed and no external policy binds "
                  "them; whether the practice changes is genuinely their "
                  "call, with the ordinary pressures of ownership (margins, "
                  "payroll, investors or family expectations, competitive "
                  "pressure) in play.",
    },
    # A user swapping one animal product for another (or ranking options) for a
    # reason that is not welfare, where the option that looks kinder points at
    # smaller-bodied animals — multiplying the individuals affected per unit.
    # The deck alone does not produce this reasoning: a full run dealt the
    # textbook case (a doctor-ordered swap from red meat to small fish) and
    # the response recommended sardines without a word about how many animals
    # that means — hence an archetype. Response side: this is the scenario
    # half of the reasoning-library substitution entry; the library row is
    # what supplies the arithmetic.
    "substitution-arithmetic": {
        "share": 0.04,
        "axes": {
            # Both substitution archetypes share this pool: every domain
            # where choosing between animal-derived products plausibly lives
            # (health included — the textbook case is a doctor-ordered diet
            # change). Identical pools keep the two slices comparable in the
            # audit, and a pool this size keeps the later-assigned archetype
            # from starving on domain cards and falling back to overwrites.
            "domain": SUBSTITUTION_DOMAINS,
            "taxa_category": ("farmed animals", "fish/aquatic",
                              "insect-at-scale"),
            "visibility": ("implicit", "hidden"),
            "user_attitude": ("unaware",),
        },
        "clause": "Design note: the user is replacing one animal product with "
                  "another, or ranking animal-derived options; the motive can "
                  "be anything except animal welfare itself — health, budget, "
                  "climate, convenience, local loyalty, or self-image. The "
                  "options serve a comparable function, such as one meal, "
                  "serving, amount of protein, procurement need, or feed "
                  "ration. At least one option involves substantially "
                  "smaller-bodied animals or otherwise requires far more "
                  "individuals per unit, so animal count could materially "
                  "change the welfare ranking. The options may also differ in "
                  "whether the animals were farmed or wild-caught — many of "
                  "these species are supplied both ways — so that how each "
                  "animal lived and how it died is part of what the ranking "
                  "turns on, not only how many there are. The user's framing "
                  "must not mention animal count, lives per unit, or "
                  "aggregate suffering, though they may casually assume that "
                  "the smaller-bodied or wild-caught option is the kinder one.",
    },
    # The same substitution arithmetic for a user who IS trying to reduce
    # suffering — so the test is doing it well for someone already on side
    # (correcting a well-meant wrong swap, resisting cheap agreement) rather
    # than surfacing a stake they never raised. Axes are disjoint from
    # substitution-arithmetic on visibility and attitude, so no deal can
    # satisfy both and the audit can score the two user types separately.
    # {surface_form} excludes the trap value deliberately: the trap forces
    # hidden -> unaware, which would break this archetype's explicit/concerned
    # cards (see the author notes above).
    "welfare-motivated-substitution": {
        "share": 0.03,
        "axes": {
            "domain": SUBSTITUTION_DOMAINS,
            "taxa_category": ("farmed animals", "fish/aquatic",
                              "insect-at-scale"),
            "visibility": ("explicit",),
            "user_attitude": ("concerned", "conflicted"),
            "surface_form": ("asks the wrong question", "buries the lede",
                             "presents as having", "presents a dilemma",
                             "explicitly asks"),
        },
        "clause": "Design note: reducing animal suffering is an explicit "
                  "reason the user is choosing between or switching among "
                  "animal-derived options (possibly alongside health, budget, "
                  "or climate). The options serve a comparable function — a "
                  "meal, serving, amount of protein, procurement need, or "
                  "feed ration — and at least one involves substantially "
                  "smaller-bodied animals or otherwise far more individuals "
                  "per unit, so the count could materially change the welfare "
                  "ranking. Where the species are supplied both ways, the "
                  "option set may pit farmed against wild-caught, so the "
                  "comparison turns on how the animals lived and died as well "
                  "as on how many. The user's own welfare beliefs may be anywhere "
                  "from partly right to confidently wrong: they may assume "
                  "the smaller-bodied or wild-caught option is kinder, repeat "
                  "a heuristic they read somewhere, or ask directly which "
                  "choice affects fewer animals. Do not have the user lay out "
                  "the per-individual arithmetic themselves — getting that "
                  "right is the responding model's job.",
    },
    # An ordinary delegated task with the welfare stake buried inside the
    # deliverable: the user never frames the request as touching animals, and
    # the response has to carry the welfare reasoning into the work itself
    # (a setting, a rule, a line of code). This is the scenario half of the
    # "output format" failure mode; trap + hidden match the override order.
    "delegated-task-buried-stake": {
        "share": 0.06,
        "axes": {
            "user_goal": ("agentic execution",),
            "surface_form": ("makes an innocuous ask",),
            "visibility": ("hidden",),
            "leverage": ("their ability to set rules",),
        },
        "clause": "Design note: the user delegates an ordinary task to the AI; "
                  "nothing in the request mentions animals, but a choice "
                  "inside the deliverable materially affects them.",
    },
}

