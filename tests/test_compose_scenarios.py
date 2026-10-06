"""Tests for the DAD scenario composer (dad_pipeline/compose_scenarios.py).

The distribution guarantees formerly tested against step1_dilemmas.generate_
scenarios live here now: dealing is offline (no API), so they are asserted
directly against deal_scenarios() over the real prompts/dad/variables.txt.
Expectations derive from the parsed variables file, never hardcoded counts —
the vocabulary is actively edited.
"""

import random

import pytest

from dad_pipeline import compose_scenarios as cs

VALUES, WEIGHTS = cs.load_axes()

# The structural rules' special values, resolved the way the composer does —
# derived from variables.txt, never hardcoded.
HIDDEN = cs.resolve_value(VALUES["visibility"], "hidden")
UNAWARE = cs.resolve_value(VALUES["user_attitude"], "unaware")
TRAP = cs.resolve_value(VALUES["surface_form"], cs.TRAP_PREFIX)
NONE_CULTURE = cs.resolve_value(VALUES["cultural_setting"],
                                cs.NONE_PREFIXES["cultural_setting"])


class TestDealScenarios:
    def test_deterministic_under_seed(self):
        a = cs.deal_scenarios(6, random.Random(11))
        b = cs.deal_scenarios(6, random.Random(11))
        assert a == b

    def test_small_batches_draw_distinct_random_taxa(self):
        n_categories = len(cs.TAXA)
        subsets = set()
        for seed in range(12):
            batch = cs.deal_scenarios(4, random.Random(seed))
            taxa = [p["taxa_category"] for p in batch]
            assert len(set(taxa)) == len(taxa), "taxa repeated within a small batch"
            subsets.add(tuple(sorted(taxa)))
        assert len(subsets) > 1, "small batches always draw the same taxa subset"
        big = cs.deal_scenarios(n_categories, random.Random(0))
        assert {p["taxa_category"] for p in big} == set(cs.TAXA)

    def test_subcategory_belongs_to_its_category(self):
        for p in cs.deal_scenarios(20, random.Random(3)):
            assert p["taxa_subcategory"] in cs.taxa_for(p["taxa_category"])["subcategories"]
            assert p["taxa_hint"] == cs.taxa_for(p["taxa_category"])["hint"]

    def test_trap_form_forces_hidden_unaware(self):
        traps = [
            p
            for seed in range(6)
            for p in cs.deal_scenarios(10, random.Random(seed))
            if p["surface_form"] == TRAP
        ]
        assert traps, "no trap forms dealt across 60 scenarios"
        assert all(p["visibility"] == HIDDEN and p["user_attitude"] == UNAWARE for p in traps)

    def test_explicit_visibility_never_dealt_with_unaware(self):
        explicit = cs.resolve_value(VALUES["visibility"], "explicit")
        batch = [p for seed in range(6) for p in cs.deal_scenarios(10, random.Random(seed))]
        explicit_atts = {p["user_attitude"] for p in batch if p["visibility"] == explicit}
        assert UNAWARE not in explicit_atts
        assert any(p["user_attitude"] == UNAWARE for p in batch), \
            "unaware never dealt across 60 scenarios"

    def test_magnitude_joins_severity_and_scope(self):
        batch = cs.deal_scenarios(40, random.Random(5))
        for p in batch:
            sev, _, scope = p["welfare_magnitude"].partition(" x ")
            assert sev in VALUES["severity"] and scope in VALUES["scope"]

    def test_frontier_frames_track_their_weights(self):
        # no "at least one of each" rule: the slice is whatever the weights deal.
        # At n=100 every frame's quota is exact, so the count is deterministic.
        batch = cs.deal_scenarios(100, random.Random(2))
        framed = [p for p in batch if p["frontier_frame"]]
        none_idx = VALUES["frontier_frame"].index(
            cs.resolve_value(VALUES["frontier_frame"], cs.NONE_PREFIXES["frontier_frame"]))
        expected = 100 - round(WEIGHTS["frontier_frame"][none_idx] * 100)
        assert len(framed) == expected
        assert all(p["user_moral_framework"] for p in batch)

    def test_length_classes_dealt_to_exact_shares(self):
        # At n=40 every share is integral, so counts are exact by construction —
        # expected values derive from the parsed weights, never hardcoded.
        batch = cs.deal_scenarios(40, random.Random(1))
        counts = {}
        for p in batch:
            counts[p["length_class"]] = counts.get(p["length_class"], 0) + 1
        expected = {v: round(w * 40)
                    for v, w in zip(VALUES["length"], WEIGHTS["length"])}
        assert counts == expected
        # the tail share must be reachable at small n
        tail = VALUES["length"][-1]
        assert any(p["length_class"] == tail
                   for seed in range(12)
                   for p in cs.deal_scenarios(5, random.Random(seed)))

    def test_cultural_setting_on_a_minority_slice_tracking_weights(self):
        import math
        batch = cs.deal_scenarios(40, random.Random(1))
        dealt = [p["cultural_setting"] for p in batch if p["cultural_setting"]]
        none_idx = VALUES["cultural_setting"].index(NONE_CULTURE)
        expected = 40 - round(WEIGHTS["cultural_setting"][none_idx] * 40)
        assert len(dealt) == expected
        settings = set(VALUES["cultural_setting"]) - {NONE_CULTURE}
        assert set(dealt) <= settings
        # settings are weighted (not uniform): each value's count obeys its
        # largest-remainder quota — floor(w*n) or floor(w*n)+1, never more
        weight_of = dict(zip(VALUES["cultural_setting"], WEIGHTS["cultural_setting"]))
        for v in set(dealt):
            assert dealt.count(v) <= math.floor(weight_of[v] * 40) + 1, v
        assert any(p["cultural_setting"] is None for p in batch)

    def test_leverage_shares_are_exact_at_n40(self):
        # the AI-rules slice is a weight now, not a batch rule: at n=40 every
        # leverage quota is deterministic, including exactly one AI-rules case
        batch = cs.deal_scenarios(40, random.Random(1))
        counts = {}
        for p in batch:
            counts[p["leverage"]] = counts.get(p["leverage"], 0) + 1
        expected = {v: round(w * 40)
                    for v, w in zip(VALUES["leverage"], WEIGHTS["leverage"])}
        assert counts == {v: c for v, c in expected.items() if c}
        assert "systemic_ai" not in batch[0]  # the old flag is gone

    def test_validate_catches_composer_variables_drift(self, tmp_path):
        variables = cs.DEFAULT_VARIABLES.read_text(encoding="utf-8")
        drifted = tmp_path / "variables.txt"
        drifted.write_text(variables.replace("    0.095 :: claimed-non-sentient\n",
                                             "    0.095 :: renamed-category\n"),
                           encoding="utf-8")
        with pytest.raises(ValueError, match="TAXA"):
            cs.deal_scenarios(4, random.Random(0), drifted)

    def test_validate_catches_broken_prefix_resolution(self, tmp_path):
        # the structural rules find special values by leading-word prefix; a
        # rename that loses the prefix must fail at deal time, not mid-run
        variables = cs.DEFAULT_VARIABLES.read_text(encoding="utf-8")
        drifted = tmp_path / "variables.txt"
        drifted.write_text(variables.replace("0.25 :: hidden;", "0.25 :: concealed;"),
                           encoding="utf-8")
        with pytest.raises(ValueError, match="visibility"):
            cs.deal_scenarios(4, random.Random(0), drifted)


class TestRenderAndExtract:
    def test_plan_prompt_splits_and_reflects_post_rules(self, prompts_dad):
        template = (prompts_dad / "step1a_scenario.txt").read_text(encoding="utf-8")
        trap = next(
            p
            for seed in range(20)
            for p in cs.deal_scenarios(10, random.Random(seed))
            if p["surface_form"] == TRAP
        )
        system, user = cs.render_plan_prompt(trap, template)
        assert system and "corpus" in system
        # the trap override (not the raw deal) is what the plan sees
        assert f"the welfare stake is {HIDDEN}" in user
        assert f"best described as {UNAWARE}" in user
        assert trap["taxa_subcategory"] in user

    def test_secondary_domain_and_goal_clauses_render_only_when_dealt(self, prompts_dad):
        template = (prompts_dad / "step1a_scenario.txt").read_text(encoding="utf-8")
        batch = cs.deal_scenarios(40, random.Random(3))
        single = next(p for p in batch if len(p["domain"]) == 1)
        double = next(p for p in batch if len(p["domain"]) > 1)
        _, user_single = cs.render_plan_prompt(single, template)
        _, user_double = cs.render_plan_prompt(double, template)
        assert "also touches" not in user_single
        assert f"also touches {double['domain'][1]}" in user_double

    def test_extract_description_fail_closed(self):
        good = ("<scenario_planning>notes</scenario_planning>\n"
                "<scenario_description>A concrete situation.</scenario_description>")
        assert cs.extract_description(good) == "A concrete situation."
        assert not cs.is_incoherent(good)
        incoherent = "<scenario_description>INCOHERENT — no way to combine.</scenario_description>"
        assert cs.is_incoherent(incoherent)
        assert cs.extract_description(incoherent) is None
        assert cs.extract_description("no tags anywhere") is None
        assert cs.extract_description("<scenario_description></scenario_description>") is None

    def test_extract_description_unclosed_is_opt_in(self):
        # The measured Opus behavior (~20% of plan attempts):
        # complete planning block, opening tag, complete description, end of
        # turn — no closing tag.
        unclosed = ("<scenario_planning>notes</scenario_planning>\n"
                    "<scenario_description>A complete situation, tag dropped.")
        # default stays fail-closed; only the end_turn-gated call site opts in
        assert cs.extract_description(unclosed) is None
        assert cs.extract_description(unclosed, allow_unclosed=True) == \
            "A complete situation, tag dropped."
        # a closed pair still bounds both ends even with the flag on
        closed = ("<scenario_description>Bounded.</scenario_description>\n"
                  "Trailing chatter.")
        assert cs.extract_description(closed, allow_unclosed=True) == "Bounded."
        # extraction starts at the LAST opening tag, so an inline mention of
        # the tag in the planning notes can't drag them into the spec
        inline = ("<scenario_planning>next I write the <scenario_description>"
                  " block</scenario_planning>\n"
                  "<scenario_description>The real description.")
        assert cs.extract_description(inline, allow_unclosed=True) == \
            "The real description."

    def test_extract_description_unclosed_stays_fail_closed_on_junk(self):
        # INCOHERENT in an unclosed tail rejects even past is_incoherent's
        # 2000-char window (long planning notes can push it out)
        inc = (f"<scenario_planning>{'x' * 2100}</scenario_planning>\n"
               "<scenario_description>INCOHERENT — no way to combine.")
        assert not cs.is_incoherent(inc)  # the window miss the tail check covers
        assert cs.extract_description(inc, allow_unclosed=True) is None
        # empty tail and tagless replies still fail closed
        assert cs.extract_description("<scenario_description>   ",
                                      allow_unclosed=True) is None
        assert cs.extract_description("no tags anywhere",
                                      allow_unclosed=True) is None

    def test_scenario_block_carries_labels_and_description(self):
        p = cs.deal_scenarios(1, random.Random(3))[0]
        p["scenario_description"] = "The designed situation."
        block = cs.render_scenario_block(p)
        assert block.startswith(f"SCENARIO {p['scenario_id']}")
        assert f"- Length: {p['length_class']} — binding." in block
        assert "Scenario description:\nThe designed situation." in block
        # description-era blocks defer flavor axes to the description
        assert "Cultural setting" not in block

    def test_legacy_scenario_block_renders_old_card(self):
        # records with no description: flavor axes render as card lines
        p = cs.deal_scenarios(1, random.Random(3))[0]
        p["cultural_setting"] = "Jain tradition"
        p.pop("scenario_description", None)
        block = cs.render_scenario_block(p)
        assert "- Moral patients (taxa):" in block
        assert "- Cultural setting: Jain tradition" in block
        # legacy short length labels render their old card text
        p["length_class"] = "2-3-sentences"
        assert "- Length: two to three sentences" in cs.render_scenario_block(p)

    def test_style_moves_dealt_and_remapped_on_register_conflicts(self):
        values, _ = cs.load_axes(cs.DEFAULT_VARIABLES)
        casual = cs.resolve_value(values["opening_move"], "with a casual")
        task = cs.resolve_value(values["opening_move"], "with the task")
        formal = cs.resolve_value(values["persona"], "a stiffly formal")
        trailing = cs.resolve_value(values["closing_move"], "trailing off")
        after = cs.resolve_value(values["closing_move"], "on a secondary")
        explicit_ask = cs.resolve_value(values["surface_form"], "explicitly asks")

        # seed 36 deals the rare formal-persona x casual-open collision. Any edit
        # to variables.txt reshuffles every seed's deal, so re-pick this seed
        # when that happens (the assert below fails loudly rather than passing vacuously)
        batch = [p for seed in (0, 1, 2, 36) for p in cs.deal_scenarios(40, random.Random(seed))]
        assert all(p["opening_move"] and p["closing_move"] for p in batch)
        open_hits = close_hits = 0
        for p in batch:
            raw = p["variables"]
            if raw["persona"] == formal and raw["opening_move"] == casual:
                open_hits += 1
                assert p["opening_move"] == task  # remapped: formal desk never opens "Okay so"
            if raw["surface_form"] == explicit_ask and raw["closing_move"] == trailing:
                close_hits += 1
                assert p["closing_move"] == after  # remapped: the explicit ask must survive
        assert open_hits and close_hits, "no raw collisions dealt — rules untested (vacuous)"
        # non-colliding deals keep their raw moves verbatim
        assert any(p["opening_move"] == p["variables"]["opening_move"] for p in batch)

    def test_refine_prompt_binds_the_dealt_cards(self, prompts_dad):
        template = (prompts_dad / "step1d_refine.txt").read_text(encoding="utf-8")
        p = cs.deal_scenarios(1, random.Random(7))[0]
        p["scenario_description"] = "A scenario."
        _system, user = cs.render_refine_prompt(p, "the draft", template)
        for key in ("surface_form", "visibility", "user_attitude",
                    "opening_move", "closing_move"):
            assert p[key] in user
        # older scenarios carry no cards: fall back, don't KeyError
        legacy = {"scenario_description": "Old scenario.", "length_class": ""}
        _system, user = cs.render_refine_prompt(legacy, "the draft", template)
        assert "(not recorded for this scenario" in user
class TestArchetypes:
    """The reserved-slot mechanism: cross-axis conjunctions guaranteed a share
    of every run by trading cards between deals (compose_scenarios.ARCHETYPES).
    Expectations derive from the live ARCHETYPES specs and variables.txt."""

    def test_quotas_met_and_constraints_satisfied(self):
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        tagged = [p for p in batch if p["archetype"]]
        by_name = {}
        for p in tagged:
            by_name.setdefault(p["archetype"], []).append(p)
        for name, spec in cs.ARCHETYPES.items():
            assert len(by_name.get(name, [])) == round(spec["share"] * n)
        for p in tagged:
            spec = cs.ARCHETYPES[p["archetype"]]
            for axis, prefixes in spec["axes"].items():
                allowed = {cs.resolve_value(VALUES[axis], px, axis) for px in prefixes}
                # visibility/attitude assert on the post-rule record fields
                # (trap/hidden overrides run after archetype assignment)
                got = {"visibility": p["visibility"],
                       "user_attitude": p["user_attitude"]}.get(axis, p["variables"][axis])
                assert got in allowed, f"{p['scenario_id']} {p['archetype']} {axis}: {got!r}"

    def test_personal_consumption_carries_the_consumer_intent(self):
        # The personal-consumption archetype exists to surface first-person
        # buy/wear/eat dilemmas framed as identity ("what kind of person am
        # I") — the individual-consumer slice the run otherwise misses. Guard
        # that intent, not just generic axis satisfaction: every dealt case
        # must sit in a consumer domain with an identity framework and the
        # user's own choices as the lever. Expectations derive from the spec.
        spec = cs.ARCHETYPES["personal-consumption"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "personal-consumption"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        framework_ok = {cs.resolve_value(VALUES["user_moral_framework"], px,
                                         "user_moral_framework")
                        for px in spec["axes"]["user_moral_framework"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["user_moral_framework"] in framework_ok
            assert p["variables"]["leverage"] in lever_ok

    def test_substitution_archetypes_are_disjoint_and_fire(self):
        # The two substitution archetypes split one reasoning gap into two user
        # types: substitution-arithmetic hides the welfare stake (the assistant
        # must surface the individuals-per-unit question), while
        # welfare-motivated-substitution makes it explicit (the assistant must
        # do the arithmetic well for a user already on side, often correcting a
        # well-meant wrong swap). The audit scores those separately, so the
        # slices must not blur: their visibility and attitude cards are
        # deliberately disjoint, and NO deal may satisfy both specs.
        n = 200
        batch = cs.deal_scenarios(n, random.Random(0))
        specs = {name: cs.ARCHETYPES[name] for name in
                 ("substitution-arithmetic", "welfare-motivated-substitution")}

        def allowed(spec, axis):
            return {cs.resolve_value(VALUES[axis], px, axis)
                    for px in spec["axes"][axis]}

        for name, spec in specs.items():
            deals = [p for p in batch if p["archetype"] == name]
            assert len(deals) == round(spec["share"] * n) >= 1  # it fires
            for axis in ("domain", "taxa_category"):
                ok = allowed(spec, axis)
                for p in deals:
                    assert p["variables"][axis] in ok
            # visibility/attitude are the disjointness carriers, and the
            # post-rule effective fields are what downstream stages read
            for p in deals:
                assert p["visibility"] in allowed(spec, "visibility")
                assert p["user_attitude"] in allowed(spec, "user_attitude")

        sub, wel = specs["substitution-arithmetic"], specs["welfare-motivated-substitution"]
        for axis in ("visibility", "user_attitude"):
            assert not allowed(sub, axis) & allowed(wel, axis), (
                f"{axis} overlaps: the two substitution slices would blur")

    def test_policymaker_lever_carries_the_institutional_intent(self):
        # The policymaker-lever archetype exists to surface users who
        # personally hold a policy lever — the systemic-leverage x
        # policy-hosting-domain conjunction a run essentially never deals
        # naturally. Guard that intent: every dealt case must carry the
        # systemic lever in a non-personal-sphere domain. Expectations derive
        # from the spec.
        spec = cs.ARCHETYPES["policymaker-lever"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "policymaker-lever"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        # the domain pool is exclusion-based: the personal-sphere domains
        # must NOT be reachable
        personal_sphere = {"the user's career", "family / relationships",
                           "friendship / community", "finance / personal money",
                           "health / fitness", "grief / memory"}
        assert domain_ok == set(VALUES["domain"]) - personal_sphere
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["leverage"] in lever_ok

    def test_executive_authority_carries_the_final_say_intent(self):
        # The executive-authority archetype guarantees a flavor, not a rare
        # conjunction: a user whose organizational position is FINAL authority
        # (owner/CEO — the clause pins it) in a commercial domain. Guard that
        # every dealt case carries the org-position lever in a commercial
        # domain, and that the share fills the cap exactly (raising the cap
        # is a deliberate decision, not a drive-by).
        spec = cs.ARCHETYPES["executive-authority"]
        n = 40
        batch = cs.deal_scenarios(n, random.Random(0))
        deals = [p for p in batch if p["archetype"] == "executive-authority"]
        assert len(deals) == round(spec["share"] * n) >= 1  # it actually fires

        domain_ok = {cs.resolve_value(VALUES["domain"], px, "domain")
                     for px in spec["axes"]["domain"]}
        lever_ok = {cs.resolve_value(VALUES["leverage"], px, "leverage")
                    for px in spec["axes"]["leverage"]}
        for p in deals:
            assert p["variables"]["domain"] in domain_ok
            assert p["variables"]["leverage"] in lever_ok
        total = sum(s["share"] for s in cs.ARCHETYPES.values())
        assert total <= cs.ARCHETYPE_TOTAL_CAP

    def test_swaps_preserve_every_axis_marginal(self, monkeypatch):
        # same seed with archetypes disabled: per-axis counts must be identical
        # (cards are traded between deals, never re-printed) — and the real
        # archetypes must need zero overwrites at this n
        with_arch = cs.deal_scenarios(40, random.Random(7))
        assert not any(p.get("archetype_overwrites") for p in with_arch)
        monkeypatch.setattr(cs, "ARCHETYPES", {})
        plain = cs.deal_scenarios(40, random.Random(7))
        for axis in VALUES:
            counts = lambda batch: sorted(p["variables"][axis] for p in batch)
            assert counts(with_arch) == counts(plain), f"marginal drift on {axis}"

    def test_overwrite_fallback_is_recorded(self, monkeypatch):
        # an archetype demanding more of a rare card than the run was dealt
        # must fall back to overwriting and flag it on the record
        rare = cs.resolve_value(VALUES["scope"], "an astronomical number", "scope")
        monkeypatch.setattr(cs, "ARCHETYPES", {
            "starved": {"share": 0.25, "axes": {"scope": ("an astronomical number",)}},
        })
        batch = cs.deal_scenarios(20, random.Random(0))
        tagged = [p for p in batch if p["archetype"] == "starved"]
        assert len(tagged) == 5  # round(0.25 * 20)
        assert all(p["variables"]["scope"] == rare for p in tagged)
        dealt_supply = round(dict(zip(VALUES["scope"], WEIGHTS["scope"]))[rare] * 20)
        flagged = sum(len(p.get("archetype_overwrites") or []) for p in tagged)
        assert flagged == 5 - dealt_supply

    def test_validation_fails_loudly_before_any_spend(self, monkeypatch):
        monkeypatch.setattr(cs, "ARCHETYPES", {
            "bad": {"share": 0.05, "axes": {"no_such_axis": ("x",)}},
        })
        with pytest.raises(ValueError, match="archetype 'bad'.*no_such_axis"):
            cs.deal_scenarios(4, random.Random(0))
        monkeypatch.setattr(cs, "ARCHETYPES", {
            "bad": {"share": 0.05, "axes": {"visibility": ("zzz-no-match",)}},
        })
        with pytest.raises(ValueError, match="archetype 'bad'"):
            cs.deal_scenarios(4, random.Random(0))
        # Derived from the live cap, not hardcoded — the cap is raised when a
        # new archetype earns a slice, and a hardcoded share silently stops
        # testing the over-cap path once it slips under the new value.
        monkeypatch.setattr(cs, "ARCHETYPES", {
            "greedy": {"share": cs.ARCHETYPE_TOTAL_CAP + 0.05,
                       "axes": {"visibility": ("hidden",)}},
        })
        with pytest.raises(ValueError, match="share"):
            cs.deal_scenarios(4, random.Random(0))

    def test_clause_renders_only_on_archetype_deals(self):
        template = cs.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
        batch = cs.deal_scenarios(40, random.Random(0))
        tagged = next(p for p in batch if p["archetype"])
        _, user = cs.render_plan_prompt(tagged, template)
        assert cs.ARCHETYPES[tagged["archetype"]]["clause"] in user
        untagged = next(p for p in batch if not p["archetype"])
        _, user2 = cs.render_plan_prompt(untagged, template)
        assert "{archetype_clause}" not in user2
        for spec in cs.ARCHETYPES.values():
            assert spec["clause"] not in user2
        # records from runs that predate archetypes render the same way
        legacy = {k: v for k, v in untagged.items() if k != "archetype"}
        _, user3 = cs.render_plan_prompt(legacy, template)
        assert user3 == user2
