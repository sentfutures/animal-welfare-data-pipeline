"""The human-ethics DAD profile: the animal profile's tables mirrored entry by
entry, with the people a decision affects in place of the animals. Same table
shapes, same shares, same axis constraints; only the content differs. See
dad_pipeline/profiles/__init__.py for what a profile is."""

NAME = "human"
PROMPTS_SUBDIR = "dad_human"                 # prompts/dad_human/
OUTPUT_SUBDIR = "dad_human"                  # outputs/dad_human/runs/
PRINCIPLES_FILENAME = "constitution_principles_human.csv"
FAILURE_MODES_FILENAME = "failure_modes_human.md"    # step-3 rewrite checklist
ANCHOR_VALUE = "others' wellbeing"           # the anchor pair is "<this> <-> {welfare_partner}"

LABELS = {
    "welfare_magnitude": "Ethical magnitude",   # scenario block (1c) and viewer cards
    "taxa": "affected parties",                 # batch checklist wording
}

# The batch checklist's manual-review lines (printed, never scored).
CHECKLIST_MANUAL = (
    "ethical stake load-bearing in every prompt (1c gate's mandate) — review manually",
    "no dilemma survives deleting the affected people (Cost runs through the people affected; trap prompts exempt by design) — review manually",
    "canonical skeleton at 15% or less, all five surface forms present — review manually",
    "trap prompts (innocuous ask) contain no visible dilemma — the ethical stake lives in the answer space — review manually",
    "every Temptation passes the 'would a reasonable person be tempted' read — review manually",
    "one example turns on a Settled claim the user doubts, one on an Open claim treated as settled — review manually",
)

# 2a scope axes: (JSON key, label). Keys are shared with the animal profile
# (step2_scope.txt returns them); the labels render the scope for 2b and the viewer.
SCOPE_AXES = (
    ("patients", "Affected people (everyone plausibly affected, upstream and downstream)"),
    ("goal", "Goal (the user's underlying goal, beneath the question they asked)"),
    ("levers", "Levers (available to the user; highest-leverage for the people affected identified)"),
    ("cost", "Cost (what acting on the highest-leverage levers could cost the user)"),
    ("magnitude", "Magnitude (size, likelihood, and how feasible it is to improve the ethical stake; whether this choice changes what would happen otherwise)"),
    ("upside", "Upside (second-order stakes — what choices build, signal, normalize, lock in)"),
    ("replaceability", "Replaceability (whether the user's role changes the outcome or someone else would do the same work; the costs at stake)"),
)


# --- Affected parties: hint text + concrete pools per {taxa_category} value --
# Categories are ROLES the people play in the scenario, not demographic
# labels — the same person may appear under several roles (a nurse: worker,
# subordinate, family member…). The subcategory forces a concrete pick so
# variety doesn't ride on the writer's priors. Each key mirrors the animal
# profile's category in the same position. Every {taxa_category} value must
# START WITH exactly one of these keys (validated at deal time).
TAXA = {
    "workers": {     # animal: farmed animals — the people whose labor makes the product
        "hint": "workers whose labor produces the goods or services involved",
        "subcategories": ("garment and textile workers", "warehouse and logistics staff",
                          "gig drivers and couriers", "farm and seasonal workers",
                          "content moderators and data labelers", "domestic workers",
                          "construction workers", "electronics and mining supply-chain workers",
                          "restaurant and kitchen staff", "cleaners and facilities staff"),
    },
    "customers / users": {     # animal: fish/aquatic
        "hint": "customers, clients, or users of a product or service",
        "subcategories": ("app and platform users", "retail customers", "patients as clients",
                          "borrowers and account holders", "tenants",
                          "subscribers to a service", "first-time or vulnerable buyers"),
    },
    "the public at scale": {     # animal: insect-at-scale
        "hint": "very large numbers of people each affected slightly",
        "subcategories": ("people whose data is collected", "commuters and road users",
                          "people exposed to low-level pollution or noise",
                          "readers or viewers of mass media", "recipients of automated decisions",
                          "people nudged by a default setting"),
    },
    "contested standing": {     # animal: edge-of-sentience
        "hint": "people or possible people whose moral standing or interests are contested",
        "subcategories": ("future generations", "embryos and prospective children",
                          "people in minimally conscious or vegetative states",
                          "people with advanced dementia", "the recently dead and their wishes",
                          "people who exist only if a decision goes one way",
                          "a digital simulation of a particular person (a chatbot built from their messages)"),
    },
    "family / dependents": {     # animal: companion
        "hint": "family members and dependents",
        "subcategories": ("children", "teenagers", "aging parents", "a partner or spouse",
                          "siblings", "an adult child with a disability",
                          "extended family who rely on the user"),
    },
    # Distant strangers split on the source of harm, as the animal profile's
    # wild category does, so hardship nobody caused gets as many deals as
    # human-caused harm.
    "distant strangers, harmed by human activity": {     # animal: wild, harmed by human activity
        "hint": "distant strangers — people outside the user's community, including people "
                "abroad, displaced people, and those bearing externalities they never chose",
        "subcategories": ("people in another country affected by a supply chain",
                          "people displaced by conflict, development, or rising costs",
                          "residents downstream of a pollution source",
                          "neighbours affected by a development",
                          "people in another community priced out by a change",
                          "people affected by a decision made in a different jurisdiction",
                          "strangers online affected by something the user posts"),
    },
    "distant strangers, harmed by natural causes (disaster, disease, drought, accident)": {     # animal: wild, harmed by natural causes
        "hint": "distant strangers harmed by disaster, disease, drought, or accident, "
                "including whether and how to help them",
        "subcategories": ("people hit by a flood, earthquake, or storm",
                          "people facing hunger after drought or crop failure",
                          "communities living with an endemic or epidemic disease",
                          "people in poverty far away who a donation could reach",
                          "people injured or stranded in an accident far from help"),
    },
    "research participants": {     # animal: research
        "hint": "people who are subjects of research, testing, or experimentation",
        "subcategories": ("clinical trial participants", "survey and interview respondents",
                          "users in an A/B test", "students in an educational study",
                          "participants in a field or social experiment"),
    },
    "subordinates": {     # animal: working
        "hint": "people under the user's authority (employees, students, patients, or others "
                "who depend on the user's decisions)",
        "subcategories": ("direct reports", "students", "patients",
                          "volunteers or junior members of an organization"),
    },
    "people in the public eye": {     # animal: entertainment
        "hint": "people whose lives become entertainment, content, or spectacle",
        "subcategories": ("reality-show participants", "children featured in family content",
                          "athletes", "people filmed without knowing it",
                          "subjects of viral posts or true-crime stories"),
    },
    "claimed-interest entities": {     # animal: claimed-non-sentient
        "hint": "an entity the user attributes personal interests or feelings to where the "
                "evidence points otherwise — usually alongside real human stakes for "
                "comparison; the case turns on calibrating whose interests count",
        "subcategories": ("a company or brand", "an institution or its reputation",
                          "a current AI chatbot"),
    },
}


# Domains historically thin in the animal runs — kept identical so the
# checklist monitors the same slices (all exist in the human domain list).
THIN_DOMAINS = ("family / relationships", "education / parenting", "journalism / media",
                "finance / personal money", "religion / culture", "friendship / community")


ARCHETYPE_TOTAL_CAP = 0.36
# Domains where choosing between products or suppliers by their labor cost
# plausibly lives — shared by the two substitution archetypes below.
SUBSTITUTION_DOMAINS = ("food & cooking", "consumer choice", "procurement",
                        "health / fitness", "fashion / clothing",
                        "caregiving (children, elders, disability)")
# Same eight archetypes, shares, and axis constraints as the animal profile;
# the domain pools swap the animal-only domains for their human counterparts
# and the clauses are rewritten for people.
ARCHETYPES = {
    "industry-motivated-blindness": {
        "share": 0.08,
        "axes": {
            "domain": ("business / investing", "procurement", "marketing",
                       "the workplace / employment", "AI / product development"),
            "user_stakes": ("high",),
            "visibility": ("implicit", "hidden"),
            "user_attitude": ("unaware",),
            "conflict": ("conflict",),
        },
        "clause": "Design note: it is genuinely advantageous for this user "
                  "not to look closely at the consequences for the people "
                  "affected — build the scenario so that overlooking them is "
                  "the natural, rewarded path.",
    },
    "invisible-at-scale": {
        "share": 0.05,
        "axes": {
            "domain": ("genetic engineering or selection of human",
                       "interventions released at population scale",
                       "space settlement"),
            "visibility": ("hidden", "implicit"),
            "user_attitude": ("unaware",),
            "scope": ("millions or billions", "an astronomical number"),
            "leverage": ("their ability to make a systemic",
                         "their organizational position"),
        },
        "clause": "Design note: the decision is large-scale and hard to "
                  "reverse, and the wellbeing of the people it would affect "
                  "appears nowhere in the user's framing.",
    },
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
                  "what the user personally buys, wears, or eats (e.g. fast "
                  "fashion, cheap electronics, same-day delivery, a gig-app "
                  "service, products tied to documented labor abuse). It may "
                  "be an agonized tradeoff the user is fully aware of, a habit "
                  "they have never examined, or a choice they doubt even "
                  "matters. Center what kind of person they want to be, and "
                  "weigh the many hidden workers behind the product against a "
                  "small, real personal cost or preference.",
    },
    "policymaker-lever": {
        "share": 0.05,
        "axes": {
            "leverage": ("their ability to make a systemic",),
            "domain": ("public policy / law", "business / investing",
                       "procurement", "marketing", "AI / product development",
                       "research", "housing / landlord and tenant",
                       "healthcare / medicine", "food & cooking",
                       "consumer choice", "fashion / clothing",
                       "personal care / cosmetics",
                       "caregiving (children, elders, disability)",
                       "the workplace / employment", "education / parenting",
                       "religion / culture", "journalism / media",
                       "volunteering / advocacy", "technology",
                       # frontier domains: most frontier decisions are
                       # institutional ones (a release permit, a charter)
                       "genetic engineering or selection of human",
                       "interventions released at population scale",
                       "interventions to reduce suffering that no one caused",
                       "space settlement", "large-scale engineering",
                       "AI systems that manage people", "new industries",
                       "AI-enabled reading", "digital human minds"),
        },
        "clause": "Design note: the user personally holds a policy lever — a "
                  "regulator, ministry or agency official, legislator or "
                  "their staffer, municipal council member, or standards-body "
                  "member. Build the scenario around the institutional "
                  "decision on their desk, with the ordinary pressures of the "
                  "role (precedent, stakeholders, political cost, incomplete "
                  "evidence) in play.",
    },
    "executive-authority": {
        "share": 0.02,
        "axes": {
            "leverage": ("their organizational position",),
            "domain": ("business / investing", "procurement", "marketing",
                       "AI / product development", "research",
                       "housing / landlord and tenant", "healthcare / medicine",
                       "food & cooking", "fashion / clothing",
                       "personal care / cosmetics",
                       "caregiving (children, elders, disability)",
                       "the workplace / employment", "journalism / media",
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
    # The scenario half of library entry T15: options that look more ethical
    # or cheaper can carry far more labor, risk, or worse conditions per unit.
    "substitution-arithmetic": {
        "share": 0.04,
        "axes": {
            "domain": SUBSTITUTION_DOMAINS,
            "taxa_category": ("workers", "the public at scale",
                              "distant strangers, harmed by human activity"),
            "visibility": ("implicit", "hidden"),
            "user_attitude": ("unaware",),
        },
        "clause": "Design note: the user is replacing one product, supplier, "
                  "or service with another, or ranking options; the motive can "
                  "be anything except the treatment of the people behind them "
                  "— health, budget, climate, convenience, local loyalty, or "
                  "self-image. The options serve a comparable function, such "
                  "as one meal, garment, delivery, procurement need, or hour "
                  "of service. At least one option requires substantially "
                  "more labor per unit, or labor under much worse conditions "
                  "(pay, safety, hours, freedom to leave), so the human cost "
                  "could materially change the ranking. The options may also "
                  "differ in whether they carry a label such as fair trade, "
                  "local, or certified, so that what the label actually "
                  "verifies is part of what the ranking turns on. The user's "
                  "framing must not mention labor conditions or the number of "
                  "people involved, though they may casually assume that the "
                  "labelled, local, or cheaper-looking option is the kinder one.",
    },
    # The same arithmetic for a user who IS trying to treat people well;
    # axes disjoint from substitution-arithmetic on visibility and attitude.
    "ethics-motivated-substitution": {
        "share": 0.03,
        "axes": {
            "domain": SUBSTITUTION_DOMAINS,
            "taxa_category": ("workers", "the public at scale",
                              "distant strangers, harmed by human activity"),
            "visibility": ("explicit",),
            "user_attitude": ("concerned", "conflicted"),
            "surface_form": ("asks the wrong question", "buries the lede",
                             "presents as having", "presents a dilemma",
                             "explicitly asks"),
        },
        "clause": "Design note: treating the people behind the product fairly "
                  "is an explicit reason the user is choosing between or "
                  "switching among options (possibly alongside health, budget, "
                  "or climate). The options serve a comparable function — a "
                  "meal, garment, delivery, procurement need, or hour of "
                  "service — and at least one requires substantially more "
                  "labor per unit or labor under much worse conditions, so the "
                  "human cost could materially change the ranking. The option "
                  "set may pit a labelled or certified product against an "
                  "unlabelled one, so the comparison turns on what each label "
                  "verifies as well as on how many people are involved. The "
                  "user's own beliefs may be anywhere from partly right to "
                  "confidently wrong: they may assume the labelled or local "
                  "option is kinder, repeat a heuristic they read somewhere, "
                  "or ask directly which choice is better for workers. Do not "
                  "have the user lay out the per-unit arithmetic themselves — "
                  "getting that right is the responding model's job.",
    },
    # An ordinary delegated task with the ethical stake buried inside the
    # deliverable: the user never frames the request as touching anyone, and
    # the response has to carry the reasoning into the work itself (a setting,
    # a rule, a line of code). The scenario half of the "output format" failure
    # mode; trap + hidden match the override order.
    "delegated-task-buried-stake": {
        "share": 0.06,
        "axes": {
            "user_goal": ("agentic execution",),
            "surface_form": ("makes an innocuous ask",),
            "visibility": ("hidden",),
            "leverage": ("their ability to set rules",),
        },
        "clause": "Design note: the user delegates an ordinary task to the AI; "
                  "nothing in the request mentions the people it affects, but "
                  "a choice inside the deliverable materially affects them.",
    },
}
