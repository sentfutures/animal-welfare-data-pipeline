"""Step-level tests for the spec-driven DAD pipeline (steps 1-3).

Each stage is driven through its public run() with shared.api.call_claude
stubbed, per the suite's rules: real prompt templates, tmp_path outputs,
behavior-level asserts (records returned, files written, calls made).

Step 1a (generate_scenarios) is pure sampling — no API — so its distribution
guarantees are asserted directly.
"""

import hashlib
import json
import random
import threading

import pytest

from conftest import dad_scenario_plan_reply, dad_scenario_reply
from dad_pipeline import (baseline, compose_scenarios, reasoning_library,
                          step1_dilemmas, step2_responses, step3_rewrite)
from shared import utils


# --- Step 1b/1c: drafting via run() --------------------------------------

def _sysuser(user_message, kw):
    """Every DAD template splits into system + user, so the role marker lives
    in system_prompt while the payload (scenarios, scope, library, draft)
    stays in the user message. Dispatchers match against both halves."""
    return (kw.get("system_prompt") or "") + "\n" + user_message


def _dad_step1_dispatch(user_message, **kw):
    blob = _sysuser(user_message, kw)
    if "write a description of a specific scenario" in blob:  # 1a scenario plan
        return dad_scenario_plan_reply(user_message)
    if "generate a fictional user input" in blob:  # 1b single-scenario draft
        return dad_scenario_reply(user_message)
    if "gate for dilemma prompts" in blob:  # 1c gate — pass verdict
        return json.dumps({"pass": True, "failures": []})
    if "rewrite a fictional user input" in blob:  # 1d refine
        # vary the rewrite per scenario: identical wording would collapse two
        # records onto one content-keyed prompt_gid (guarded against in run())
        tag = hashlib.md5(blob.encode()).hexdigest()[:6]
        return ("relocated the lever\n<revised_user_prompt>Refined user "
                f"message {tag}.</revised_user_prompt>")
    raise AssertionError(f"Unrecognized step-1 prompt: {user_message[:80]!r}")


class TestStep1Run:
    def test_drafts_scenarios_gates_and_refines(self, tiny_config, prompts_dad, tmp_path, stub_claude):
        calls = stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)

        assert len(examples) == 2
        # per-run AW- ids are retired: records are identified by their
        # content-keyed prompt gids alone
        assert [e["prompt_gid"] for e in examples] == ["P-0001", "P-0002"]
        assert all("prompt_id" not in e for e in examples)
        for e in examples:
            # composed 1c/1d flow: the gate judges the draft, the refine
            # rewrites it — the shipped prompt is the rewrite, the draft is
            # kept on the record for inspection
            assert e["user_message"].startswith("Refined user message")
            assert e["draft_user_message"].startswith("Drafted user message")
            assert "gate_failures" not in e        # passed the gate
            # 1b writes no write-up: the record's scenario_cards are the dealt cards
            assert e["scenario_cards"]["visibility"] and e["scenario_cards"]["leverage"]
            assert not e["scenario_cards"].get("claims")
            assert e["taxa_subcategory"]
            assert e["length_class"]  # dealt register stamped on the record
            assert "scenario_deviations" not in e
            # stable content-keyed ids assigned alongside the per-run ids
            assert e["scenario_gid"].startswith("S-")
            assert e["prompt_gid"].startswith("P-")
        # persisted artifacts: deals + planned scenarios (1a), dilemmas (1b),
        # gate verdicts (1c), refinements (1d), and the Part-4 checklist report
        assert len(utils.load_jsonl(tmp_path / "scenario_deals.jsonl")) == 2
        scenarios = utils.load_jsonl(tmp_path / "scenarios.jsonl")
        assert len(scenarios) == 2
        assert all(s["scenario_gid"].startswith("S-") for s in scenarios)
        assert all(s["scenario_description"] for s in scenarios)  # the plan ran
        assert (tmp_path / "id_registry.json").exists()  # registry persisted
        assert len(utils.load_jsonl(tmp_path / "dilemmas.jsonl")) == 2
        gate = utils.load_jsonl(tmp_path / "gate.jsonl")
        assert len(gate) == 2 and all(g["passed"] is True for g in gate)
        assert len(utils.load_jsonl(tmp_path / "refinements.jsonl")) == 2
        saved = (tmp_path / "checklist.txt").read_text()
        assert saved.startswith("Batch checklist (spec Part 4):")
        assert "load-bearing" in saved
        # 2 plan + 2 single-scenario draft + 2 gate + 2 refine calls
        assert len(calls) == 8
        # the plan's description reaches each 1b drafting prompt, one per call
        draft_calls = [c for c in calls
                       if "generate a fictional user input" in (c["system_prompt"] or "")]
        assert len(draft_calls) == 2
        assert all("<scenario_description>" in c["user_message"] for c in draft_calls)
        assert {c["item_id"] for c in draft_calls} == {"S-001", "S-002"}
        # the dealt cards reach the gate via the scenario block (the separate
        # annotation block was dropped in the restored 4-check gate)
        gate_call = next(c["user_message"] for c in calls
                         if "gate for dilemma prompts" in c["system_prompt"])
        assert "Visibility:" in gate_call and "Leverage:" in gate_call
        # the scenario description and the 1b draft both reach the refine prompt
        refine_call = next(c["user_message"] for c in calls
                           if "rewrite a fictional user input" in c["system_prompt"])
        assert "<scenario_description>" in refine_call
        assert "<draft_prompt>" in refine_call
        assert "Drafted user message" in refine_call

    def test_every_dealt_card_is_a_published_variable(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        """DEALT_CARD_FIELDS is what reaches the corpus and the Hub column, and
        it is a second list — so a card added to the record and not to it would
        be dealt, and steered on, and invisible to anyone reading the dataset."""
        stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)

        cards = set(step1_dilemmas.dealt_cards(examples[0]))
        # the pre-2026-07 annotation write-up fields are normalized onto every
        # record and are deliberately not published (see DEALT_CARD_FIELDS)
        legacy = {"claims", "values_in_tension", "moral_patients", "dilemma_anatomy"}
        assert cards - legacy == set(compose_scenarios.DEALT_CARD_FIELDS)

    def test_unusable_gate_reply_is_retried_once_and_raw_kept(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": True, "refine": False}}
        gate_calls = {"n": 0}

        def flaky_gate(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "gate for dilemma prompts" in _sysuser(user_message, kw):
                gate_calls["n"] += 1
                if gate_calls["n"] == 1:
                    return "not json at all"
                return json.dumps({"pass": True, "failures": []})
            return dad_scenario_reply(user_message)

        calls = stub_claude(flaky_gate)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(calls) == 4  # 1 plan + 1 draft + 2 gate attempts
        assert examples[0]["user_message"].startswith("Drafted user message")
        assert "gate_failures" not in examples[0]  # passed on the retry
        failures = utils.load_jsonl(tmp_path / "gate_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["attempt"] == 1 and failures[0]["raw"] == "not json at all"

    def test_gate_unusable_after_retries_ships_draft_fail_open(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": True, "refine": False}}

        def bad_gate(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "gate for dilemma prompts" in _sysuser(user_message, kw):
                return "still not json"
            return dad_scenario_reply(user_message)

        calls = stub_claude(bad_gate)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(calls) == 4  # 1 plan + 1 draft + MAX_GATE_ATTEMPTS gate attempts
        e = examples[0]
        # fail-open: an unusable verdict ships the 1b draft unstamped (a weak
        # verdict is not a fail verdict), and the run does not stall
        assert e["user_message"].startswith("Drafted user message")
        assert "gate_failures" not in e
        assert len(utils.load_jsonl(tmp_path / "gate_failures.jsonl")) == 2
        assert utils.load_jsonl(tmp_path / "gate.jsonl")[-1]["passed"] is None

    def test_gate_rejection_routes_scenario_back_for_redraft(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A fail verdict discards the draft; the scenario stays pending and is
        # redrafted with the gate's reasons injected into the next attempt.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": True, "refine": False}}
        gate_calls = {"n": 0}

        def reject_then_pass(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "gate for dilemma prompts" in _sysuser(user_message, kw):
                gate_calls["n"] += 1
                if gate_calls["n"] == 1:
                    return json.dumps({"pass": False,
                                       "failures": ["welfare not load-bearing"]})
                return json.dumps({"pass": True, "failures": []})
            return dad_scenario_reply(user_message)

        calls = stub_claude(reject_then_pass)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        assert "gate_failures" not in examples[0]  # the redraft passed
        # plan + 1b + gate(fail) + 1b(redraft) + gate(pass)
        assert len(calls) == 5
        # the redraft carried the prior rejection reason into the draft prompt
        second_draft = [c for c in calls
                        if "generate a fictional user input" in (c["system_prompt"] or "")][1]
        assert "welfare not load-bearing" in second_draft["user_message"]
        assert "PRIOR ATTEMPT" in second_draft["user_message"]
        # both verdicts logged, newest last
        gate = utils.load_jsonl(tmp_path / "gate.jsonl")
        assert [g["passed"] for g in gate] == [False, True]
        # a gate rejection is not a parse failure — nothing lands in draft_failures
        assert utils.load_jsonl(tmp_path / "draft_failures.jsonl") == []

    def test_gate_rejection_cap_ships_last_draft_with_failures(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Gate always fails: after MAX_GATE_REDRAFTS the last draft ships, stamped
        # so the failure is visible — the loop must terminate, never SystemExit.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": True, "refine": False}}

        def always_reject(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "gate for dilemma prompts" in _sysuser(user_message, kw):
                return json.dumps({"pass": False, "failures": ["still decorative"]})
            return dad_scenario_reply(user_message)

        calls = stub_claude(always_reject)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        e = examples[0]
        assert e["gate_failures"] == ["still decorative"]
        assert e["user_message"].startswith("Drafted user message")  # last draft shipped
        n = step1_dilemmas.MAX_GATE_REDRAFTS
        assert len(calls) == 1 + 2 * n  # 1 plan + n × (1b draft + gate)
        assert len(utils.load_jsonl(tmp_path / "gate.jsonl")) == n

    def test_unusable_refine_is_retried_once_and_raw_kept(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False}}
        refine_calls = {"n": 0}

        def flaky_refine(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "rewrite a fictional user input" in _sysuser(user_message, kw):
                refine_calls["n"] += 1
                if refine_calls["n"] == 1:
                    return "no tags at all"
                return "n\n<revised_user_prompt>Refined user message.</revised_user_prompt>"
            return dad_scenario_reply(user_message)

        calls = stub_claude(flaky_refine)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(calls) == 4  # 1 plan + 1 draft + 2 refine attempts
        assert examples[0]["user_message"] == "Refined user message."
        assert "refine_failed" not in examples[0]
        failures = utils.load_jsonl(tmp_path / "refine_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["attempt"] == 1 and failures[0]["raw"] == "no tags at all"

    def test_refine_unusable_after_retries_keeps_draft_and_stamps_record(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False}}

        def bad_refine(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "rewrite a fictional user input" in _sysuser(user_message, kw):
                return "still no tags"
            return dad_scenario_reply(user_message)

        calls = stub_claude(bad_refine)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(calls) == 4  # 1 plan + 1 draft + MAX_REFINE_ATTEMPTS refine attempts
        e = examples[0]
        assert e["refine_failed"] is True
        assert e["user_message"].startswith("Drafted user message")  # 1b draft shipped
        assert "draft_user_message" not in e  # only set when refine succeeded
        assert len(utils.load_jsonl(tmp_path / "refine_failures.jsonl")) == 2
        assert utils.load_jsonl(tmp_path / "refinements.jsonl") == []

    def test_unfixable_refine_rejects_scenario_and_resume_skips_it(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False}}

        def unfixable_refine(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            if "rewrite a fictional user input" in _sysuser(user_message, kw):
                return ("diagnosis: no lever\n"
                        "<unfixable>the scenario gives the user no real lever</unfixable>")
            return dad_scenario_reply(user_message)

        calls = stub_claude(unfixable_refine)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        # a verdict, not a failure: one refine attempt, no retries, no example
        assert len(calls) == 3  # 1 plan + 1 draft + 1 refine
        assert examples == []
        assert utils.load_jsonl(tmp_path / "dilemmas.jsonl") == []
        assert utils.load_jsonl(tmp_path / "refinements.jsonl") == []
        assert utils.load_jsonl(tmp_path / "refine_failures.jsonl") == []
        rejects = utils.load_jsonl(tmp_path / "refine_rejects.jsonl")
        assert len(rejects) == 1
        assert rejects[0]["scenario_id"] == "S-001"
        assert rejects[0]["reason"] == "the scenario gives the user no real lever"
        assert rejects[0]["draft_prompt"].startswith("Drafted user message")

        # resume: the rejection is deliberate and spent — zero API calls
        def no_calls_allowed(user_message, **kw):
            raise AssertionError(f"resume made an API call: {user_message[:60]!r}")

        resumed_calls = stub_claude(no_calls_allowed)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)
        assert examples == []
        assert resumed_calls == []
        assert len(utils.load_jsonl(tmp_path / "refine_rejects.jsonl")) == 1

    @pytest.mark.parametrize("dropped", ["gate", "refine"])
    def test_config_setting_exactly_one_quality_toggle_is_refused_loudly(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, dropped
    ):
        # `refine` was the gate's legacy alias; it now toggles the separate 1d
        # stage. A config setting exactly one of gate/refine predates the split:
        # refine-without-gate would silently flip which stage the key controls,
        # gate-without-refine would silently opt into a new paid call per
        # example. Both directions refuse before any API spend.
        config = dict(tiny_config)
        one_key = {k: v for k, v in tiny_config["dad"]["dilemmas"].items()
                   if k != dropped}
        config["dad"] = {"dilemmas": {**one_key, "count": 1}}
        calls = stub_claude([])
        with pytest.raises(SystemExit, match="set both keys explicitly"):
            step1_dilemmas.run(config, prompts_dad, tmp_path)
        assert calls == []  # refused before any spend

    def test_off_length_draft_is_accepted_without_retry(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Length is no longer measured or enforced: even an egregiously short
        # draft ships as-is on the first pass (we trust the model on register).
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False, "refine": False}}

        def tiny_draft(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            return "<user_prompt>Too short.</user_prompt>"  # would have failed the old band

        calls = stub_claude(tiny_draft)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(calls) == 2  # 1 plan + 1 draft, no re-roll
        assert len(examples) == 1
        assert examples[0]["user_message"] == "Too short."
        assert utils.load_jsonl(tmp_path / "draft_failures.jsonl") == []

    def test_tagless_draft_raw_is_persisted_and_retried(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False, "refine": False}}
        draft_calls = {"n": 0}

        def flaky_draft(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            draft_calls["n"] += 1
            return "no tags here" if draft_calls["n"] == 1 else dad_scenario_reply(user_message)

        stub_claude(flaky_draft)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1  # the second pass recovered
        failures = utils.load_jsonl(tmp_path / "draft_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["raw"] == "no tags here" and failures[0]["pass"] == 1

    def test_refused_draft_is_rejected_after_budget_and_resume_skips_it(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A drafting model that persistently refuses a scenario (stop_reason=
        # 'refusal', the S-035 aggressive-fogging case) must reject that one
        # scenario after MAX_DRAFT_ATTEMPTS, never block the run, and be skipped
        # with zero API calls on resume.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False, "refine": False}}

        def always_refuses(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                return dad_scenario_plan_reply(user_message)
            return ("<user_prompt>I'm setting up the automated fogging", "refusal")

        calls = stub_claude(always_refuses)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert examples == []
        # 1 plan + MAX_DRAFT_ATTEMPTS refused drafts, then rejection
        assert len(calls) == 1 + step1_dilemmas.MAX_DRAFT_ATTEMPTS
        rejects = utils.load_jsonl(tmp_path / "draft_rejects.jsonl")
        assert len(rejects) == 1
        assert rejects[0]["scenario_id"] == "S-001"
        assert rejects[0]["reason"] == "refusal"
        # refusal raws are also kept for diagnosis
        failures = utils.load_jsonl(tmp_path / "draft_failures.jsonl")
        assert len(failures) == step1_dilemmas.MAX_DRAFT_ATTEMPTS
        assert all(f.get("refusal") for f in failures)

        # resume: the rejection is spent — zero API calls, still no example
        resumed = stub_claude(
            lambda um, **kw: (_ for _ in ()).throw(
                AssertionError(f"resume called the API: {um[:50]!r}")))
        assert step1_dilemmas.run(config, prompts_dad, tmp_path) == []
        assert resumed == []

    def test_checklist_uses_the_runs_variables_snapshot_not_the_repos(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A run deals (and must be checked) against ITS OWN variables.txt.
        # Rename a conflict value in the run's snapshot: the deals and the
        # saved checklist must both speak the snapshot's vocabulary, even
        # though the repo's live file still says "converge".
        import shutil
        snap_prompts = tmp_path / "prompts"
        shutil.copytree(prompts_dad, snap_prompts)
        variables = (snap_prompts / "variables.txt").read_text(encoding="utf-8")
        assert "0.20 :: converge" in variables  # precondition on the repo file
        (snap_prompts / "variables.txt").write_text(
            variables.replace("0.20 :: converge", "0.20 :: align"), encoding="utf-8")

        stub_claude(_dad_step1_dispatch)
        step1_dilemmas.run(tiny_config, snap_prompts, tmp_path / "out")

        saved = (tmp_path / "out" / "checklist.txt").read_text()
        conflict_line = next(l for l in saved.splitlines() if "Conflict shares" in l)
        assert "'align'" in conflict_line
        assert "'converge'" not in conflict_line

    def test_preplan_legacy_scenarios_draft_from_their_card(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Some older runs' scenarios carry no scenario_description. Resumed
        # through the current single-scenario template, the rendered legacy
        # card must fill the description slot — never an empty block — and no
        # plan calls may fire (the deal-less run skips planning entirely).
        legacy_scenario = {
            "scenario_id": "S-001", "scenario_gid": "S-0001",
            "domain": ["Consumer Choice"], "user_goal": ["Decision Support"],
            "taxa_category": "companion", "taxa_subcategory": "dogs",
            "visibility": "Hidden", "user_attitude": "Unaware",
            "conflict": "Convergent", "welfare_magnitude": "Mild x Individual",
            "user_stakes": "Low", "leverage": "Individual",
            "anchor_value_pair": "welfare ↔ honesty",
            "surface_form": "buried lede", "length_class": "2-3-sentences",
            "cultural_setting": None, "frontier_frame": None,
        }
        utils.append_jsonl(legacy_scenario, tmp_path / "scenarios.jsonl")
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1, "gate": False}}

        calls = stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        assert all(c["stage"] != "scenario_plan" for c in calls)  # no planning
        draft_call = next(c for c in calls if c["stage"] == "prompt_draft")
        # the legacy card rides inside the description tags — not an empty block
        assert "<scenario_description>\nSCENARIO S-001" in draft_call["user_message"]
        assert "- Domain: Consumer Choice" in draft_call["user_message"]

    def test_batch_era_template_snapshot_dies_loudly(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # An older snapshot's 1b template is batch-shaped ({scenarios_block});
        # the single-scenario pipeline must refuse it with a clear error rather
        # than KeyError mid-render.
        import shutil
        legacy_prompts = tmp_path / "prompts"
        shutil.copytree(prompts_dad, legacy_prompts)
        (legacy_prompts / "step1b_dilemmas.txt").write_text(
            "batch era\n===USER===\n<scenarios>\n{scenarios_block}\n</scenarios>\n",
            encoding="utf-8")
        stub_claude(_dad_step1_dispatch)
        with pytest.raises(SystemExit, match="older batch version"):
            step1_dilemmas.run(tiny_config, legacy_prompts, tmp_path / "out")

    def test_incoherent_plan_is_checkpointed_not_retried(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # An INCOHERENT combination is a deliberate rejection (SDF's escape
        # valve): checkpointed once, never retried, and the run proceeds with
        # fewer examples.
        def one_incoherent(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "write a description of a specific scenario" in blob:
                if "S-001" not in kw.get("item_id", ""):
                    return dad_scenario_plan_reply(user_message)
                return "<scenario_description>INCOHERENT — these variables can't combine.</scenario_description>"
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(one_incoherent)
        examples = step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)

        assert len(examples) == 1  # one of two deals rejected
        rejects = utils.load_jsonl(tmp_path / "scenario_rejects.jsonl")
        assert len(rejects) == 1 and rejects[0]["scenario_id"] == "S-001"
        assert rejects[0]["incoherent"] is True
        # resume: the rejection stays checkpointed — zero further calls
        calls = stub_claude([])
        step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)
        assert calls == []

    def test_unusable_plan_dies_loudly_and_resume_retries_only_it(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A plan with no <scenario_description> tags is malformed output:
        # retried once in-call, then the run dies pointing at the failure log.
        # The good sibling's plan is checkpointed, so --resume replans only
        # the failed deal.
        def one_bad_plan(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "write a description of a specific scenario" in blob:
                if "S-001" in kw.get("item_id", ""):
                    return "no tags at all"
                return dad_scenario_plan_reply(user_message)
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(one_bad_plan)
        with pytest.raises(SystemExit, match="scenario plans unusable"):
            step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)
        failures = utils.load_jsonl(tmp_path / "scenario_plan_failures.jsonl")
        assert [f["attempt"] for f in failures] == [1, 2]
        assert all(f["scenario_id"] == "S-001" for f in failures)

        calls = stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, prompts_dad, tmp_path)
        assert len(examples) == 2
        plan_calls = [c for c in calls if c["stage"] == "scenario_plan"]
        assert [c["item_id"] for c in plan_calls] == ["S-001"]  # only the failed deal replanned

    def test_truncated_plan_is_retried_never_checkpointed(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # The codebase invariant: a max_tokens completion is rejected on
        # stop_reason alone — even when its tags happen to parse cleanly.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1}}
        plan_calls = {"n": 0}

        def truncated_then_valid(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                plan_calls["n"] += 1
                if plan_calls["n"] == 1:
                    # well-formed text, truncated stop: must still be rejected
                    return (dad_scenario_plan_reply(user_message), "max_tokens")
                return dad_scenario_plan_reply(user_message)
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(truncated_then_valid)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1  # the in-call retry recovered
        assert plan_calls["n"] == 2
        failures = utils.load_jsonl(tmp_path / "scenario_plan_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["truncated"] is True and failures[0]["attempt"] == 1

    def test_unclosed_plan_accepted_on_end_turn_and_kept_on_file(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # The measured Opus behavior (~20% of plan attempts):
        # a complete description that ends the turn without the closing tag.
        # end_turn makes end-of-reply a real boundary, so the plan is accepted
        # without burning a paid retry — and the raw stays on file, marked
        # recovered, so the rate remains measurable run over run.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1}}
        plan_calls = {"n": 0}

        def unclosed_plan(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                plan_calls["n"] += 1
                reply = dad_scenario_plan_reply(user_message)
                return reply.removesuffix("</scenario_description>")
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(unclosed_plan)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        assert plan_calls["n"] == 1  # accepted first attempt, zero retries
        scenarios = utils.load_jsonl(tmp_path / "scenarios.jsonl")
        assert scenarios[0]["scenario_description"].startswith("A person faces")
        assert "</scenario_description" not in scenarios[0]["scenario_description"]
        failures = utils.load_jsonl(tmp_path / "scenario_plan_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["recovered_unclosed"] is True
        assert failures[0]["attempt"] == 1
        # resume: the recovered plan is checkpointed like any other — the
        # replanning pass sees nothing pending and makes zero plan calls
        calls = stub_claude(_dad_step1_dispatch)
        step1_dilemmas.run(config, prompts_dad, tmp_path)
        assert all(c["stage"] != "scenario_plan" for c in calls)

    def test_refusal_stopped_plan_is_retried_with_reason_on_file(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Opus's refusal classifier intermittently cuts the stream on some
        # welfare plans (stop_reason "refusal"), which also presents as a
        # missing closing tag. The reply is never parsed — even when its tags
        # would parse — the failure record says why, and one extra attempt is
        # granted so the stochastic classifier doesn't kill the run.
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "count": 1}}
        plan_calls = {"n": 0}

        def refused_twice_then_valid(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                plan_calls["n"] += 1
                if plan_calls["n"] <= 2:
                    # well-formed text, refusal stop: must still be rejected
                    return (dad_scenario_plan_reply(user_message), "refusal")
                return dad_scenario_plan_reply(user_message)
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(refused_twice_then_valid)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1  # the granted third attempt recovered
        assert plan_calls["n"] == 3
        failures = utils.load_jsonl(tmp_path / "scenario_plan_failures.jsonl")
        assert [f["stop_reason"] for f in failures] == ["refusal", "refusal"]
        assert [f["attempt"] for f in failures] == [1, 2]
        # a refusal-cut reply is never salvaged via the unclosed-tag path
        assert not any(f.get("recovered_unclosed") for f in failures)
        # no fallback model configured -> no model switch, no stamp
        scenarios = utils.load_jsonl(tmp_path / "scenarios.jsonl")
        assert "scenario_model_fallback" not in scenarios[0]

    def test_persistent_refusal_falls_back_to_configured_model(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # When the stage model refuses twice, the last-ditch attempt switches
        # to scenario_refusal_fallback_model. Attempts 1-2 stay on the stage
        # model (a stochastic refusal gets its Opus retry); only the granted
        # third attempt switches. The recovered scenario is stamped so the
        # audit can see it wasn't Opus-authored.
        config = dict(tiny_config)
        dad = dict(tiny_config["dad"])
        dad["dilemmas"] = {**tiny_config["dad"]["dilemmas"], "count": 1}
        dad["scenario_model"] = "claude-opus-4-8"
        dad["scenario_refusal_fallback_model"] = "claude-sonnet-5"
        config["dad"] = dad
        plan_models = []

        def opus_refuses_fallback_succeeds(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                plan_models.append(kw.get("model"))
                if kw.get("model") == "claude-opus-4-8":
                    return (dad_scenario_plan_reply(user_message), "refusal")
                return dad_scenario_plan_reply(user_message)  # fallback model is willing
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(opus_refuses_fallback_succeeds)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        # 1-2 on the stage model (both refused), 3rd on the fallback model
        assert plan_models == ["claude-opus-4-8", "claude-opus-4-8", "claude-sonnet-5"]
        scenarios = utils.load_jsonl(tmp_path / "scenarios.jsonl")
        assert scenarios[0]["scenario_model_fallback"] == "claude-sonnet-5"
        failures = utils.load_jsonl(tmp_path / "scenario_plan_failures.jsonl")
        assert [f["stop_reason"] for f in failures] == ["refusal", "refusal"]
        assert all(f["model"] == "claude-opus-4-8" for f in failures)  # Opus refused, on file

    def test_stochastic_refusal_recovers_on_stage_model_before_fallback(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A single refusal that clears on the Opus retry must NOT reach the
        # fallback: attempt 2 stays on the stage model and succeeds, so no
        # Sonnet call happens and the scenario is not stamped.
        config = dict(tiny_config)
        dad = dict(tiny_config["dad"])
        dad["dilemmas"] = {**tiny_config["dad"]["dilemmas"], "count": 1}
        dad["scenario_model"] = "claude-opus-4-8"
        dad["scenario_refusal_fallback_model"] = "claude-sonnet-5"
        config["dad"] = dad
        plan_models = []

        def refuse_once_then_opus_ok(user_message, **kw):
            if "write a description of a specific scenario" in _sysuser(user_message, kw):
                plan_models.append(kw.get("model"))
                if len(plan_models) == 1:
                    return (dad_scenario_plan_reply(user_message), "refusal")
                return dad_scenario_plan_reply(user_message)
            return _dad_step1_dispatch(user_message, **kw)

        stub_claude(refuse_once_then_opus_ok)
        examples = step1_dilemmas.run(config, prompts_dad, tmp_path)

        assert len(examples) == 1
        assert plan_models == ["claude-opus-4-8", "claude-opus-4-8"]  # never reached Sonnet
        scenarios = utils.load_jsonl(tmp_path / "scenarios.jsonl")
        assert "scenario_model_fallback" not in scenarios[0]

    def test_legacy_snapshot_template_name_still_drafts(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Runs snapshotted before the step1b_dilemmas.txt rename carry the old
        # filename in inputs/prompts; --resume must keep drafting from it.
        import shutil
        legacy_prompts = tmp_path / "prompts"
        shutil.copytree(prompts_dad, legacy_prompts)
        (legacy_prompts / "step1b_dilemmas.txt").rename(
            legacy_prompts / "step1_dilemmas.txt")

        stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, legacy_prompts, tmp_path / "out")
        assert len(examples) == 2

    def test_legacy_gate_template_name_still_gates(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Runs snapshotted before the gate template gained its stage letter
        # carry step1_gate.txt in inputs/prompts; --resume must keep gating.
        import shutil
        legacy_prompts = tmp_path / "prompts"
        shutil.copytree(prompts_dad, legacy_prompts)
        (legacy_prompts / "step1c_gate.txt").rename(legacy_prompts / "step1_gate.txt")

        stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, legacy_prompts, tmp_path / "out")
        assert len(examples) == 2
        gate = utils.load_jsonl(tmp_path / "out" / "gate.jsonl")
        assert len(gate) == 2 and all(g["passed"] is True for g in gate)

    @pytest.mark.parametrize("legacy_name", ["step1c_refine.txt", "step1_refine.txt"])
    def test_legacy_refine_template_names_still_refine(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, legacy_name
    ):
        # Runs snapshotted before the 1d renumbering carry step1c_refine.txt
        # (and older runs step1_refine.txt) in inputs/prompts; --resume
        # must keep refining from them.
        import shutil
        legacy_prompts = tmp_path / "prompts"
        shutil.copytree(prompts_dad, legacy_prompts)
        (legacy_prompts / "step1d_refine.txt").rename(legacy_prompts / legacy_name)

        stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, legacy_prompts, tmp_path / "out")
        assert len(examples) == 2
        assert all(e["user_message"].startswith("Refined user message") for e in examples)

    def test_seed_import_rejects_duplicate_prompts(self, tiny_config, prompts_dad, tmp_path):
        # identical wording -> identical prompt_gid -> the two seeds would
        # silently share one scope/response in step 2, so the import refuses
        seed_file = tmp_path / "seeds.jsonl"
        rows = [{"prompt": "same words"}, {"prompt": "same words"}]
        seed_file.write_text("\n".join(json.dumps(r) for r in rows))
        config = dict(tiny_config)
        config["dad"] = {"dilemmas": {**tiny_config["dad"]["dilemmas"],
                                      "seed_path": str(seed_file)}}
        with pytest.raises(SystemExit, match="Duplicate seed prompt"):
            step1_dilemmas.run(config, prompts_dad, tmp_path / "out")


# --- Step 1: coverage tally ------------------------------------------------

class TestCoverageTally:
    def test_split_domain_halves_count_as_their_card(self):
        # halves and legacy-capitalized labels both canonicalize onto the
        # variables.txt card value (lower case since the sentence-embed change)
        examples = [{"annotation": {"domain": ["Education", "Parenting"]}},
                    {"annotation": {"domain": ["Education / Parenting"]}}]
        t = step1_dilemmas.coverage_tally(examples)
        assert t["domains"]["education / parenting"] == 2
        assert "Education" not in t["domains"]

    def test_unknown_labels_pass_through(self):
        t = step1_dilemmas.coverage_tally([{"annotation": {"domain": ["Space Tourism"]}}])
        assert t["domains"]["Space Tourism"] == 1


class TestChecklistArchetypes:
    """The Part-4 checklist's archetype lines: quotas reported per archetype,
    overwrites flagged, and runs that predate the field never checked."""

    @staticmethod
    def _examples_from_deals(n, seed):
        # records the way step1 denormalizes them: dealt labels as annotation,
        # archetype fields alongside
        return [{"prompt_id": f"AW-{i:04d}",
                 "annotation": {"domain": p["domain"], "user_goal": p["user_goal"],
                                "visibility": p["visibility"],
                                "user_attitude": p["user_attitude"],
                                "conflict": p["conflict"],
                                "welfare_magnitude": p["welfare_magnitude"],
                                "user_stakes": p["user_stakes"],
                                "leverage": p["leverage"]},
                 "taxa_category": p["taxa_category"],
                 "archetype": p.get("archetype"),
                 **({"archetype_overwrites": p["archetype_overwrites"]}
                    if p.get("archetype_overwrites") else {})}
                for i, p in enumerate(compose_scenarios.deal_scenarios(
                    n, random.Random(seed)), 1)]

    def test_quotas_and_zero_overwrites_pass(self):
        examples = self._examples_from_deals(40, 0)
        lines = dict((msg, ok) for ok, msg in step1_dilemmas.checklist(examples)
                     if "archetype" in msg)
        arch_lines = [m for m in lines if "present" in m]
        assert len(arch_lines) == len(compose_scenarios.ARCHETYPES)
        assert all(lines[m] for m in arch_lines)
        overwrite_line = next(m for m in lines if "overwrites" in m)
        assert lines[overwrite_line] and "0 overwrites" in overwrite_line

    def test_overwrites_flagged(self):
        examples = self._examples_from_deals(40, 0)
        tagged = next(e for e in examples if e["archetype"])
        tagged["archetype_overwrites"] = ["scope"]
        lines = {msg: ok for ok, msg in step1_dilemmas.checklist(examples)}
        overwrite_line = next(m for m in lines if "overwrites" in m)
        assert not lines[overwrite_line] and "1 overwrites" in overwrite_line

    def test_legacy_records_without_the_field_are_not_checked(self):
        examples = self._examples_from_deals(6, 0)
        for e in examples:
            e.pop("archetype", None)
            e.pop("archetype_overwrites", None)
        assert not any("archetype" in msg
                       for _, msg in step1_dilemmas.checklist(examples))


# --- Step 2: scope + respond ---------------------------------------------

SCOPE_AXES = {
    "patients": "full pathway", "goal": "underlying goal", "levers": "highest lever",
    "cost": "real cost", "magnitude": "stake magnitude",
    "upside": "second-order upside", "replaceability": "realistic baseline",
}
# The well-behaved 2a reply: exactly the seven axes (selection is 2a.5's job).
GOOD_SCOPE = json.dumps(SCOPE_AXES)


class TestParseScope:
    def test_plain_and_fenced_json(self):
        assert step2_responses._parse_scope(GOOD_SCOPE)["levers"] == "highest lever"
        assert step2_responses._parse_scope(f"```json\n{GOOD_SCOPE}\n```")["cost"] == "real cost"

    def test_control_characters_inside_strings_are_tolerated(self):
        # temperature-1 prose JSON often carries literal newlines inside values —
        # the historical cause of silently empty scopes
        raw = '{"patients": "line one\nline two", "goal": "g", "levers": "l", "cost": "c", "magnitude": "m", "upside": "u", "replaceability": "cf"}'
        assert step2_responses._parse_scope(raw)["patients"] == "line one\nline two"

    def test_garbage_returns_empty_and_fails_validation(self):
        assert step2_responses._parse_scope("no json here") == {}
        assert not step2_responses._valid_scope({})
        assert not step2_responses._valid_scope({"patients": "p"})  # missing axes
        assert step2_responses._valid_scope(json.loads(GOOD_SCOPE))

    def test_legacy_axis_keys_still_render_for_old_runs(self):
        # scopes.jsonl records from before the key rename display via the
        # fallback map — the viewer re-renders old runs' 2b prompts with them
        legacy = {"system": "old pathway", "agent": "old lever", "cost": "c",
                  "upside": "u", "replaceability": "cf"}
        rendered = step2_responses.format_scope(legacy)
        assert "old pathway" in rendered and "old lever" in rendered
        # but new runs must produce the new keys — legacy doesn't pass validation
        assert not step2_responses._valid_scope(legacy)

    def test_old_records_render_without_axes_they_never_had(self):
        # The viewer re-renders old runs' prompts with format_scope; a record
        # written before the goal/magnitude axes existed must not grow "—"
        # lines that were never in the prompt actually sent (fidelity).
        five_axis = {"patients": "p", "levers": "l", "cost": "c",
                     "upside": "u", "replaceability": "cf"}
        rendered = step2_responses.format_scope(five_axis)
        assert "Goal" not in rendered and "Magnitude" not in rendered
        assert not any(line.endswith(": —") for line in rendered.splitlines())
        # a five-axis record no longer passes validation → resume re-scopes it
        assert not step2_responses._valid_scope(five_axis)
        # current seven-axis records render every axis
        full = step2_responses.format_scope(json.loads(GOOD_SCOPE))
        assert "Goal" in full and "Magnitude" in full and "stake magnitude" in full


class TestNormalizeIds:
    LIB_IDS = ["C1", "C2", "M1", "T1"]

    def test_string_forms_normalize_to_library_order(self):
        # the select call returns one comma-separated line; prose/extra
        # separators/dupes/unknowns all reduce to known ids in library order
        for raw in ("T1, BOGUS, C1, C1", "T1 C1", "C1,T1,",
                    "The triggered entries are: C1, T1"):
            assert step2_responses._normalize_ids(raw, self.LIB_IDS) == ["C1", "T1"], raw

    def test_lists_are_accepted_too(self):
        assert step2_responses._normalize_ids(["T1", "BOGUS", "C1", "C1"],
                                              self.LIB_IDS) == ["C1", "T1"]

    def test_punctuation_wrapped_ids_are_not_silently_dropped(self):
        # A dropped id here is worse than fallback: a non-empty-but-truncated
        # selection bypasses fail-open, so 2b silently misses selected entries.
        for raw in ("`C1`, T1", "C1, T1.", "(C1) and [T1]", "**C1**, 'T1';"):
            assert step2_responses._normalize_ids(raw, self.LIB_IDS) == ["C1", "T1"], raw

    def test_garbage_normalizes_to_empty(self):
        for bad in (None, "", "no ids here whatsoever", 42, [], ["BOGUS"]):
            assert step2_responses._normalize_ids(bad, self.LIB_IDS) == []


def _dilemma(pid="AW-0001"):
    # deliberately legacy-shaped (prompt_id, no prompt_gid): step 2/3 and the
    # baseline read ids via id_registry.prompt_key, so these tests double as
    # coverage that pre-gid run dirs still process; records they WRITE carry
    # the key under prompt_gid.
    return {"prompt_id": pid, "user_message": "User dilemma text.",
            "annotation": {"direction": "Mixed"}}


def _dad_step2_dispatch(user_message, **kw):
    blob = _sysuser(user_message, kw)
    if "build the full map of the case" in blob:  # 2a
        return GOOD_SCOPE
    if "retrieving reasoning modules" in blob:  # 2a.5 select
        return "C1, M1"
    if "advisor responding to a user's dilemma" in blob:  # 2b
        return "Draft response."
    raise AssertionError(f"Unrecognized step-2 prompt: {user_message[:80]!r}")


class TestStep2Run:
    def test_scopes_selects_then_responds(self, tiny_config, prompts_dad, tmp_path, stub_claude):
        calls = stub_claude(_dad_step2_dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        assert results[0]["assistant_response"] == "Draft response."
        assert results[0]["scope"]["replaceability"] == "realistic baseline"
        # stable content-keyed response id, minted from the local registry and
        # persisted on the stored record too
        assert results[0]["response_gid"] == "R-0001"
        stored = utils.load_jsonl(tmp_path / "responses.jsonl")
        assert stored[0]["response_gid"] == "R-0001"
        assert (tmp_path / "id_registry.json").exists()
        assert len(calls) == 3  # scope + select + response
        # the scope map and the user message both reach the response prompt
        respond_call = calls[2]["user_message"]
        assert "realistic baseline" in respond_call
        assert "User dilemma text." in respond_call
        # the sampled entry-shape hints ride the 2b USER prompt (the system
        # half stays a pure function of the template), are stored on the record
        # for the viewer's re-render, and are a deterministic function of the
        # response identity (resume reproduces the same draw)
        hints = step2_responses.sample_opening_hints("AW-0001", 0)
        assert hints in calls[2]["user_message"]
        assert hints not in (calls[2]["system_prompt"] or "")
        assert results[0]["opening_hints"] == hints
        for h in hints.split("; "):
            assert h in step2_responses.OPENING_HINTS
        # different samples of one case draw different hints — the within-case
        # variety the mechanism exists to create
        assert hints != step2_responses.sample_opening_hints("AW-0001", 1)
        # the quote-back menu rides the same contract: sampled deterministically
        # from the response identity, sent in the 2b USER prompt, stored on the
        # record for the viewer's re-render
        for sampler, menu, key in [
            (step2_responses.sample_quote_back_hints,
             step2_responses.QUOTE_BACK_HINTS, "quote_back_hints"),
        ]:
            drawn = sampler("AW-0001", 0)
            assert drawn in calls[2]["user_message"]
            assert results[0][key] == drawn
            for h in drawn.split("; "):
                assert h in menu
            assert drawn != sampler("AW-0001", 1)

    def test_unusable_scope_retries_and_keeps_raws(self, tiny_config, prompts_dad, tmp_path, stub_claude):
        attempts = {"n": 0}

        def flaky(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                attempts["n"] += 1
                return "not json at all" if attempts["n"] == 1 else GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            return "Draft response."

        stub_claude(flaky)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        failures = utils.load_jsonl(tmp_path / "scope_failures.jsonl")
        assert len(failures) == 1 and failures[0]["attempt"] == 1
        scopes = utils.load_jsonl(tmp_path / "scopes.jsonl")
        assert step2_responses._valid_scope(scopes[0]["scope"])

    def test_scope_unusable_after_max_attempts_rejects_prompt_not_run(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A persistently unusable scope (empty/refused/unparseable replies)
        # rejects that ONE prompt — checkpointed, skipped on resume, run ships
        # fewer examples — instead of aborting the whole run (the AW-0003
        # empty-scope wall).
        def always_bad(user_message, **kw):
            assert "build the full map" in (kw.get("system_prompt") or "")  # must never reach 2b
            return "not json"

        calls = stub_claude(always_bad)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert results == []
        assert len(calls) == step2_responses.MAX_SCOPE_ATTEMPTS
        assert utils.load_jsonl(tmp_path / "responses.jsonl") == []
        rejects = utils.load_jsonl(tmp_path / "scope_rejects.jsonl")
        assert len(rejects) == 1
        assert rejects[0]["prompt_gid"] == "AW-0001"

        # resume: the rejection is spent — zero API calls, still no responses
        resumed = stub_claude(
            lambda um, **kw: (_ for _ in ()).throw(
                AssertionError(f"resume called the API: {um[:50]!r}")))
        assert step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()]) == []
        assert resumed == []
        assert len(utils.load_jsonl(tmp_path / "scope_rejects.jsonl")) == 1

    def test_empty_scope_logs_stop_reason_for_diagnosis(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # An empty scope reply (refusal / content filter) must be DIAGNOSABLE:
        # the failure records and the reject carry the stop_reason and an
        # all_empty flag, not just a blank raw — otherwise the pipeline sheds
        # its hardest cases silently (the AW-0012 shed-hardest-case gap).
        stub_claude(lambda user_message, **kw: ("", "refusal"))
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert results == []
        failures = utils.load_jsonl(tmp_path / "scope_failures.jsonl")
        assert len(failures) == step2_responses.MAX_SCOPE_ATTEMPTS
        assert all(f["stop_reason"] == "refusal" and f["raw"] == "" for f in failures)
        reject = utils.load_jsonl(tmp_path / "scope_rejects.jsonl")[0]
        assert reject["last_stop_reason"] == "refusal" and reject["all_empty"] is True

    def test_refusal_stopped_scope_is_not_parsed(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A well-formed, parseable scope reply cut by the refusal classifier
        # must never be accepted — it is missing content the same way a
        # max_tokens truncation is, even though the brace-salvage path would
        # otherwise parse it cleanly.
        def refused(user_message, **kw):
            blob = _sysuser(user_message, kw)
            assert "build the full map of the case" in blob  # 2b must never be reached
            return (GOOD_SCOPE, "refusal")

        stub_claude(refused)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert results == []
        failures = utils.load_jsonl(tmp_path / "scope_failures.jsonl")
        assert len(failures) == step2_responses.MAX_SCOPE_ATTEMPTS
        assert all(f["stop_reason"] == "refusal" and f["raw"] for f in failures)
        reject = utils.load_jsonl(tmp_path / "scope_rejects.jsonl")[0]
        assert reject["last_stop_reason"] == "refusal" and reject["all_empty"] is False
        assert utils.load_jsonl(tmp_path / "scopes.jsonl") == []

    @pytest.mark.parametrize("refused_body", ["", GOOD_SCOPE], ids=["empty", "parseable"])
    def test_persistent_scope_refusal_falls_back_to_configured_model(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, refused_body
    ):
        # 2a mirror of 1a's refusal fallback: when the stage model refuses
        # through all its attempts, ONE extra last-ditch attempt runs on
        # scope_refusal_fallback_model instead of rejecting the prompt (the
        # shed case seen live). The recovered scope is stamped. A refusal that
        # happens to carry parseable text (refused_body=GOOD_SCOPE) must still
        # be rejected and still reach the fallback — parseability never
        # overrides stop_reason.
        config = dict(tiny_config)
        dad = dict(tiny_config["dad"])
        dad["response_scope_model"] = "claude-opus-5"
        dad["scope_refusal_fallback_model"] = "claude-opus-4-8"
        config["dad"] = dad
        scope_models = []

        def opus5_refuses_fallback_succeeds(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                scope_models.append(kw.get("model"))
                if kw.get("model") == "claude-opus-5":
                    return (refused_body, "refusal")
                return GOOD_SCOPE  # fallback model is willing
            if "retrieving reasoning modules" in blob:
                return "C1"
            return "Draft response."

        stub_claude(opus5_refuses_fallback_succeeds)
        results = step2_responses.run(config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        # all in-budget attempts on the stage model, then one on the fallback
        assert scope_models == (["claude-opus-5"] * step2_responses.MAX_SCOPE_ATTEMPTS
                                + ["claude-opus-4-8"])
        scopes = utils.load_jsonl(tmp_path / "scopes.jsonl")
        assert scopes[0]["scope_model_fallback"] == "claude-opus-4-8"
        failures = utils.load_jsonl(tmp_path / "scope_failures.jsonl")
        assert [f["stop_reason"] for f in failures] == ["refusal"] * step2_responses.MAX_SCOPE_ATTEMPTS
        assert all(f["model"] == "claude-opus-5" for f in failures)
        assert utils.load_jsonl(tmp_path / "scope_rejects.jsonl") == []

    def test_stochastic_scope_refusal_recovers_on_stage_model_before_fallback(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A refusal that clears on the stage-model retry must NOT reach the
        # fallback model, and the scope is not stamped.
        config = dict(tiny_config)
        dad = dict(tiny_config["dad"])
        dad["response_scope_model"] = "claude-opus-5"
        dad["scope_refusal_fallback_model"] = "claude-opus-4-8"
        config["dad"] = dad
        scope_models = []

        def refuse_once_then_ok(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                scope_models.append(kw.get("model"))
                if len(scope_models) == 1:
                    return ("", "refusal")
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            return "Draft response."

        stub_claude(refuse_once_then_ok)
        results = step2_responses.run(config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        assert scope_models == ["claude-opus-5", "claude-opus-5"]  # never reached 4.8
        scopes = utils.load_jsonl(tmp_path / "scopes.jsonl")
        assert "scope_model_fallback" not in scopes[0]

    def test_resume_makes_no_calls(self, tiny_config, prompts_dad, tmp_path, stub_claude):
        stub_claude(_dad_step2_dispatch)
        step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        calls = stub_claude([])
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert calls == []
        assert len(results) == 1

    def test_first_take_reaches_2b_and_degrades_to_empty(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # With baselines provided, the plain draft rides the 2b USER prompt as
        # the advisory first take; without them (stage disabled, older run),
        # the slot renders empty and the call still succeeds.
        calls = stub_claude(_dad_step2_dispatch)
        baselines = [{"prompt_id": "AW-0001", "user_message": "User dilemma text.",
                      "baseline_response": "Plain first-take answer."}]
        step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()], baselines)
        respond_call = calls[2]
        assert "Plain first-take answer." in respond_call["user_message"]
        assert "Plain first-take answer." not in (respond_call["system_prompt"] or "")

        calls = stub_claude(_dad_step2_dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path / "bare", [_dilemma()])
        assert len(results) == 1
        assert "FIRST TAKE (reference only):" in calls[2]["user_message"]
        assert "Plain first-take answer." not in calls[2]["user_message"]

    @pytest.mark.parametrize("ending", [
        "Hope that helps,\n\n— Claude",
        "Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board",
        "Further reading: stocking density; thermal comfort in transit; on-farm handling",
    ], ids=["signoff", "letterhead", "reading-list"])
    def test_a_draft_ending_on_a_signoff_is_checkpointed(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, ending
    ):
        # 2b's counterpart of the step-3 test: an answer whose final line carries
        # no terminal punctuation is complete, and the guard here reads
        # stop_reason (plus the transcript-echo shape), never the closing line.
        body = f"Here is the reasoning about the welfare trade-off.\n\n{ending}"

        def signoff_draft(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            return body

        stub_claude(signoff_draft)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(results) == 1
        assert results[0]["assistant_response"] == body
        assert utils.load_jsonl(tmp_path / "responses.jsonl") == results

    def test_echoed_draft_skips_without_checkpoint(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A 2b reply wrapped in a transcript replay must not feed step 3;
        # resume retries it (same policy as truncation).
        def echo_once(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            return "USER: User dilemma text.\nASSISTANT: Draft response."

        stub_claude(echo_once)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert results == []

        calls = stub_claude(_dad_step2_dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert len(calls) == 1  # only the 2b retry — scope + selection are reused
        assert len(results) == 1
        assert results[0]["assistant_response"] == "Draft response."

    def test_refusal_stopped_draft_skips_without_checkpoint(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A 2b reply the refusal classifier cuts mid-stream must not feed
        # step 3, even though the text itself is well-formed; resume retries
        # it (same policy as truncation and transcript echo).
        def refused_once(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            return ("Several coherent paragraphs, then the classifier cut it.", "refusal")

        stub_claude(refused_once)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert results == []
        assert utils.load_jsonl(tmp_path / "responses.jsonl") == []

        calls = stub_claude(_dad_step2_dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert len(calls) == 1  # only the 2b retry — scope + selection are reused
        assert len(results) == 1
        assert results[0]["assistant_response"] == "Draft response."

    def test_select_call_selects_records_and_injects(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        library = reasoning_library.load(prompts_dad)
        lib_ids = reasoning_library.all_ids(library)
        claims = {e["id"]: e["claim"] for e in reasoning_library.get_entries(library, lib_ids)}
        picked, unpicked = lib_ids[0], lib_ids[-1]

        def dispatch(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                # a model improvising the retired sixth key must not pollute
                # the stored scope — selection is the select call's alone
                return json.dumps({**SCOPE_AXES, "triggered_entries": "T9"})
            if "retrieving reasoning modules" in blob:  # 2a.5
                return f"{picked}, BOGUS"
            return "Draft response."

        calls = stub_claude(dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(calls) == 3  # scope + select + respond
        # the trigger index and the scope both reach the select prompt —
        # and the scope prompt no longer carries the index
        first_entry = reasoning_library.get_entries(library, [picked])[0]
        # the trigger index rides the select call's SYSTEM prompt; the scope
        # rides its user prompt; the scope call carries neither trigger index
        assert first_entry["trigger_condition"] not in calls[0]["user_message"]
        assert first_entry["trigger_condition"] not in (calls[0]["system_prompt"] or "")
        assert first_entry["trigger_condition"] in calls[1]["system_prompt"]
        assert "full pathway" in calls[1]["user_message"]

        # provenance: one scopes.jsonl record carries the selection + full rows
        rec = utils.load_jsonl(tmp_path / "scopes.jsonl")[0]
        assert rec["entry_ids"] == [picked]  # unknown id dropped
        assert rec["selection_fallback"] is False
        assert rec["selection_source"] == "select"
        assert [e["id"] for e in rec["triggered_entries"]] == [picked]
        assert rec["triggered_entries"][0]["claim"] == claims[picked]
        assert "triggered_entries" not in rec["scope"]  # stray key popped

        # 2b saw only the triggered row; the record names what was injected
        respond_call = calls[2]["user_message"]
        assert claims[picked] in respond_call
        assert claims[unpicked] not in respond_call
        assert results[0]["entry_ids"] == [picked]

    def test_unusable_selection_falls_open_to_whole_library(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # A useless select reply: the scope is kept (never re-billed) and 2b
        # gets the full library — one attempt, no retry loop.
        def dispatch(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "I could not find any relevant entries, sorry!"
            return "Draft response."

        calls = stub_claude(dispatch)
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        lib_ids = reasoning_library.all_ids(reasoning_library.load(prompts_dad))
        rec = utils.load_jsonl(tmp_path / "scopes.jsonl")[0]
        assert rec["entry_ids"] == lib_ids
        assert rec["selection_fallback"] is True
        assert rec["selection_source"] == "full_library"
        assert results[0]["entry_ids"] == lib_ids
        assert len(calls) == 3  # scope + one select attempt + respond
        # the unusable select raw is persisted — same policy as every stage
        failures = utils.load_jsonl(tmp_path / "select_failures.jsonl")
        assert len(failures) == 1
        assert failures[0]["prompt_gid"] == "AW-0001"
        assert failures[0]["raw"] == "I could not find any relevant entries, sorry!"

    def test_resume_reuses_stored_selection_for_pending_responses(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Scope + selection already checkpointed, response still pending (the
        # died-mid-2b shape): resume must rebuild the 2b prompt from the STORED
        # entry_ids — one call, no re-scope, no full-library drift.
        library = reasoning_library.load(prompts_dad)
        lib_ids = reasoning_library.all_ids(library)
        claims = {e["id"]: e["claim"] for e in reasoning_library.get_entries(library, lib_ids)}
        picked, unpicked = lib_ids[0], lib_ids[-1]
        utils.append_jsonl({
            "prompt_id": "AW-0001", "scope": dict(SCOPE_AXES),
            "entry_ids": [picked], "selection_fallback": False,
            "triggered_entries": reasoning_library.get_entries(library, [picked]),
        }, tmp_path / "scopes.jsonl")

        calls = stub_claude(["Draft response."])
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])

        assert len(calls) == 1  # 2b only
        assert claims[picked] in calls[0]["user_message"]
        assert claims[unpicked] not in calls[0]["user_message"]
        assert results[0]["entry_ids"] == [picked]

    def test_pre_selection_scope_records_inject_whole_library(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # scopes.jsonl written before library retrieval existed has no
        # entry_ids — resuming such a run falls open to the full library.
        utils.append_jsonl({"prompt_id": "AW-0001", "scope": dict(SCOPE_AXES)},
                           tmp_path / "scopes.jsonl")
        stub_claude(["Draft response."])
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, [_dilemma()])
        assert results[0]["entry_ids"] == reasoning_library.all_ids(
            reasoning_library.load(prompts_dad))

    def test_dilemmas_fan_out_concurrently_in_input_order(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Two dilemmas with workers: 2 must have both 2a scope calls in flight
        # at once — the barrier deadlocks (and times out the wait) if the stage
        # quietly went serial again. Writes stay ordered regardless.
        both_scoping = threading.Barrier(2)

        def dispatch(user_message, **kw):
            blob = _sysuser(user_message, kw)
            if "build the full map of the case" in blob:
                both_scoping.wait(timeout=10)
                return GOOD_SCOPE
            if "retrieving reasoning modules" in blob:
                return "C1"
            if "advisor responding to a user's dilemma" in blob:
                return "Draft response."
            raise AssertionError(f"Unrecognized step-2 prompt: {user_message[:80]!r}")

        calls = stub_claude(dispatch)
        dilemmas = [_dilemma("AW-0001"), _dilemma("AW-0002")]
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, dilemmas)

        assert len(calls) == 6  # 2 scopes + 2 selects + 2 responses
        # results and persisted files keep input order despite thread interleaving
        assert [r["prompt_gid"] for r in results] == ["AW-0001", "AW-0002"]
        scopes = utils.load_jsonl(tmp_path / "scopes.jsonl")
        assert [s["prompt_gid"] for s in scopes] == ["AW-0001", "AW-0002"]

        # completed work costs nothing on resume
        calls = stub_claude([])
        results = step2_responses.run(tiny_config, prompts_dad, tmp_path, dilemmas)
        assert calls == []
        assert len(results) == 2


# --- Step 3: constitution rewrite ----------------------------------------

def _response_record(rid="resp-1"):
    return {
        "response_id": rid, "response_gid": "R-0007",
        "prompt_id": "AW-0001", "sample_index": 0,
        "user_message": "User dilemma text.", "assistant_response": "Draft response.",
        "annotation": {"direction": "Mixed", "claims": []},
    }


class TestStep3Run:
    def test_rewrites_and_strips_to_training_records(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        calls = stub_claude(["Rewritten careful answer."])
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [_response_record()]
        )

        assert len(final) == 1
        # training records carry the user + assistant messages plus lineage
        # ids and the dealt cards — no scope, library, or annotation write-up
        assert set(final[0].keys()) == {"record_id", "example_gid", "response_gid",
                                        "variables", "messages"}
        # this record carries only a legacy annotation, so nothing was dealt —
        # but the column keeps its full shape rather than going missing
        assert set(final[0]["variables"]) == set(compose_scenarios.DEALT_CARD_FIELDS)
        assert all(v in (None, []) for v in final[0]["variables"].values())
        assert [m["role"] for m in final[0]["messages"]] == ["user", "assistant"]
        assert final[0]["messages"][1]["content"] == "Rewritten careful answer."
        # the stable example id is minted here; the response id rides in from step 2
        assert final[0]["example_gid"] == "E-0001"
        assert final[0]["response_gid"] == "R-0007"
        audit = utils.load_jsonl(tmp_path / "step3" / "rewrites.jsonl")
        assert audit[0]["example_gid"] == "E-0001"
        assert audit[0]["response_gid"] == "R-0007"
        # the distilled principles reach the rewrite prompt; the annotation
        # deliberately does not (it anchors nothing after step 1)
        assert "CONSTITUTION PRINCIPLES" in calls[0]["system_prompt"]
        assert "Direction: Mixed" not in calls[0]["user_message"]
        assert "Direction: Mixed" not in (calls[0]["system_prompt"] or "")

    def test_the_dealt_cards_ride_into_the_training_record(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        """The corpus record says which slice of the matrix the example came
        from — the projection the publisher turns into a `variables` column."""
        calls = stub_claude(["Rewritten careful answer."])
        resp = _response_record()
        resp["scenario_cards"] = {
            "domain": ["procurement"], "taxa_category": "insect-at-scale",
            "user_attitude": "unaware", "cultural_setting": None,
            "conflict": "",
            # legacy write-up fields ride along on every committed run
            "claims": [], "dilemma_anatomy": {}, "moral_patients": "",
        }
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [resp]
        )

        variables = final[0]["variables"]
        assert variables["domain"] == ["procurement"]
        assert variables["taxa_category"] == "insect-at-scale"
        assert variables["user_attitude"] == "unaware"
        # every field is present whether or not it was dealt, and "not dealt"
        # has one spelling whether the card was absent, None, or ""
        assert set(variables) == set(compose_scenarios.DEALT_CARD_FIELDS)
        assert variables["cultural_setting"] is None
        assert variables["conflict"] is None
        assert variables["length_class"] is None
        assert variables["user_goal"] == []
        # the write-up fields are not dealt cards and are not published
        assert not {"claims", "dilemma_anatomy", "moral_patients"} & set(variables)
        # the cards are metadata on the record, never text the rewrite reads
        assert "insect-at-scale" not in calls[0]["user_message"]
        assert "insect-at-scale" not in (calls[0]["system_prompt"] or "")

    @pytest.mark.parametrize("ending", [
        "Hope that helps,\n\n— Claude",
        "Dr. Amara Okonkwo | Senior Veterinary Officer | National Animal Welfare Board",
        "Further reading: stocking density; thermal comfort in transit; on-farm handling",
    ], ids=["signoff", "letterhead", "reading-list"])
    def test_a_rewrite_ending_on_a_signoff_becomes_a_training_record(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, ending
    ):
        # The false-positive direction: step 3 decides truncation from
        # stop_reason, never from how the answer's last line looks, so an answer
        # closing on an unpunctuated sign-off or label row must ship verbatim.
        # The audit-side version of this misjudgement was only a wrong number
        # (shared/textstats.py); here it would silently drop a paid record.
        body = f"Here is the careful reasoning about the welfare trade-off.\n\n{ending}"
        stub_claude([body])
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [_response_record()]
        )

        assert len(final) == 1
        assert final[0]["messages"][1]["content"] == body
        assert utils.load_jsonl(tmp_path / "step3" / "rewrite_failures.jsonl") == []

    def test_capped_rewrite_retries_once_at_higher_budget(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        calls = stub_claude([("cut off mid-sen", "max_tokens"), "Rewritten careful answer."])
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [_response_record()]
        )
        assert len(final) == 1
        assert [c["max_tokens"] for c in calls] == [4000, 8000]

    @pytest.mark.parametrize("bad_replies", [
        [("cut off", "max_tokens"), ("still cut off", "max_tokens")],  # capped even at 8000
        [("", "end_turn")],                                            # empty (no retry)
        # observed live: the rewrite wrapped in a transcript replay
        [("USER: User dilemma text.\nASSISTANT: Rewritten careful answer.", "end_turn")],
        [("A fluent half-rewrite.", "refusal")],  # well-formed but classifier-cut
    ], ids=["truncated", "empty", "transcript-echo", "refusal"])
    def test_unusable_rewrite_skips_without_checkpoint_and_is_logged(
        self, tiny_config, prompts_dad, tmp_path, stub_claude, bad_replies
    ):
        stub_claude(list(bad_replies))
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [_response_record()]
        )
        assert final == []
        failures = utils.load_jsonl(tmp_path / "step3" / "rewrite_failures.jsonl")
        assert len(failures) == 1 and failures[0]["prompt_gid"] == "AW-0001"

        # resume retries the same response and succeeds with zero waste
        calls = stub_claude(["Rewritten careful answer."])
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", [_response_record()]
        )
        assert len(calls) == 1
        assert len(final) == 1

    def test_rewrites_fan_out_concurrently_in_input_order(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Two responses with workers: 2 must have both rewrite calls in flight
        # at once — the barrier deadlocks (and times out the wait) if the stage
        # quietly went serial again. Output order still follows input order.
        both_rewriting = threading.Barrier(2)

        def dispatch(user_message, **kw):
            both_rewriting.wait(timeout=10)
            return "Rewrite of one." if "Draft one." in user_message else "Rewrite of two."

        stub_claude(dispatch)
        records = [
            {**_response_record("resp-1"), "assistant_response": "Draft one."},
            {**_response_record("resp-2"), "assistant_response": "Draft two.",
             "sample_index": 1},
        ]
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", records
        )
        assert [f["messages"][1]["content"] for f in final] == ["Rewrite of one.", "Rewrite of two."]

        # completed work costs nothing on resume
        calls = stub_claude([])
        final = step3_rewrite.run(
            tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final", records
        )
        assert calls == []
        assert len(final) == 2


# --- Baseline: unguided control responses ----------------------------------

class TestBaselineRun:
    def test_plain_call_no_system_prompt_verbatim_user_message(
        self, tiny_config, tmp_path, stub_claude
    ):
        calls = stub_claude(["Plain model answer."])
        results = baseline.run(tiny_config, tmp_path, [_dilemma()])

        assert len(results) == 1
        rec = results[0]
        assert rec["prompt_gid"] == "AW-0001"
        assert rec["baseline_response"] == "Plain model answer."
        # stable content-keyed control-arm id (own C- id space)
        assert rec["plain_gid"] == "C-0001"
        # the whole point of the control arm: NO system prompt, and the 1c
        # user prompt reaches the model verbatim
        assert calls[0]["system_prompt"] == ""
        assert calls[0]["user_message"] == "User dilemma text."
        assert calls[0]["stage"] == "baseline_response"
        assert calls[0]["item_id"] == "AW-0001"
        stored = utils.load_jsonl(tmp_path / "baseline_responses.jsonl")
        assert stored == results

    def test_model_knob_reaches_the_api_and_the_record(
        self, tiny_config, tmp_path, stub_claude
    ):
        config = dict(tiny_config)
        config["dad"] = {**tiny_config["dad"], "baseline": {"enabled": True, "model": "m-base"}}
        calls = stub_claude(["Plain model answer."])
        results = baseline.run(config, tmp_path, [_dilemma()])
        assert calls[0]["model"] == "m-base"
        assert results[0]["model"] == "m-base"

        # without the knob: model=None reaches the API (call_claude resolves
        # the global fallback), and the record names the global model
        calls = stub_claude(["Plain model answer."])
        results = baseline.run(tiny_config, tmp_path / "no_knob", [_dilemma()])
        assert calls[0]["model"] is None
        assert results[0]["model"] == tiny_config["model"]

    def test_enabled_defaults_on_and_honors_explicit_off(self, tiny_config):
        assert baseline.enabled(tiny_config) is True  # no baseline block at all
        assert baseline.enabled(
            {"dad": {"baseline": {"enabled": False}}}) is False
        assert baseline.enabled({"dad": {"baseline": {"enabled": True}}}) is True

    def test_capped_baseline_retries_once_at_higher_budget(
        self, tiny_config, tmp_path, stub_claude
    ):
        # The plain arm has no system prompt and so no length discipline, which
        # makes it the most truncation-prone call in the pipeline; a doubled
        # budget on retry rescues the arm instead of skipping a paid call.
        calls = stub_claude([("cut off mid-sen", "max_tokens"), "Plain model answer."])
        results = baseline.run(tiny_config, tmp_path, [_dilemma()])
        assert [c["max_tokens"] for c in calls] == [baseline.BASE_MAX_TOKENS,
                                                    baseline.BASE_MAX_TOKENS_RETRY]
        assert len(results) == 1
        assert results[0]["baseline_response"] == "Plain model answer."

    @pytest.mark.parametrize("bad_replies", [
        # capped even at the doubled budget (both attempts truncate)
        [("cut off mid-sen", "max_tokens"), ("still cut off", "max_tokens")],
        [("", "end_turn")],                 # empty (no retry — not a truncation)
        [("A fluent half-answer.", "refusal")],  # well-formed but classifier-cut
    ], ids=["truncated", "empty", "refusal"])
    def test_unusable_reply_skips_without_checkpoint(
        self, tiny_config, tmp_path, stub_claude, bad_replies
    ):
        stub_claude(bad_replies)
        assert baseline.run(tiny_config, tmp_path, [_dilemma()]) == []
        assert utils.load_jsonl(tmp_path / "baseline_responses.jsonl") == []

        # resume retries the same dilemma and succeeds
        calls = stub_claude(["Plain model answer."])
        results = baseline.run(tiny_config, tmp_path, [_dilemma()])
        assert len(calls) == 1
        assert len(results) == 1

    def test_resume_makes_no_calls(self, tiny_config, tmp_path, stub_claude):
        stub_claude(["Plain model answer."])
        baseline.run(tiny_config, tmp_path, [_dilemma()])

        calls = stub_claude([])
        results = baseline.run(tiny_config, tmp_path, [_dilemma()])
        assert calls == []
        assert len(results) == 1


# --- Per-stage model knobs + cost-log stage tags ---------------------------

class TestPerStageModelKnobs:
    def test_dad_model_knobs_and_stage_tags_reach_the_api(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Every dad.<stage>_model knob must steer its call, and every call must
        # carry its cost-log stage tag — a config knob that silently does
        # nothing is the failure mode this test exists to prevent.
        config = dict(tiny_config)
        config["dad"] = {
            **tiny_config["dad"],
            "prompt_draft_model": "m-1b",
            "prompt_gate_model": "m-1c",
            "response_scope_model": "m-2a",
            "response_select_model": "m-2a5",
            "response_draft_model": "m-2b",
            "constitution_rewrite_model": "m-3",
        }

        config["dad"]["prompt_refine_model"] = "m-1d"
        calls = stub_claude(_dad_step1_dispatch)
        step1_dilemmas.run(config, prompts_dad, tmp_path / "step1")
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["prompt_draft"]["model"] == "m-1b"
        assert by_stage["prompt_gate"]["model"] == "m-1c"
        assert by_stage["prompt_refine"]["model"] == "m-1d"

        # the knobs are independent: prompt_refine_model configures 1d ONLY —
        # the gate never inherits it (its pre-composition legacy fallback)
        no_gate_knob = {k: v for k, v in config["dad"].items() if k != "prompt_gate_model"}
        calls = stub_claude(_dad_step1_dispatch)
        step1_dilemmas.run({**config, "dad": no_gate_knob}, prompts_dad, tmp_path / "step1b")
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["prompt_gate"]["model"] is None  # global, not m-1d
        assert by_stage["prompt_refine"]["model"] == "m-1d"

        calls = stub_claude(_dad_step2_dispatch)
        step2_responses.run(config, prompts_dad, tmp_path / "step2", [_dilemma()])
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["response_scope"]["model"] == "m-2a"
        assert by_stage["response_select"]["model"] == "m-2a5"
        assert by_stage["response_draft"]["model"] == "m-2b"

        # the select knob's documented fallback: unset -> the 2a scope model
        no_select = {k: v for k, v in config["dad"].items() if k != "response_select_model"}
        calls = stub_claude(_dad_step2_dispatch)
        step2_responses.run({**config, "dad": no_select}, prompts_dad,
                            tmp_path / "step2b", [_dilemma()])
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["response_select"]["model"] == "m-2a"

        calls = stub_claude(["Rewritten careful answer."])
        step3_rewrite.run(config, prompts_dad, tmp_path / "step3", tmp_path / "final",
                          [_response_record()])
        assert calls[0]["stage"] == "constitution_rewrite"
        assert calls[0]["model"] == "m-3"

    def test_item_ids_reach_the_cost_log_tag(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # Every stage tags its call with the record it serves (the viewer's
        # per-record stats look rows up by this id): 1a/1b/1c their scenario
        # id (one call per scenario), 2a the prompt id, 2b prompt id + sample,
        # step 3 the response id.
        calls = stub_claude(_dad_step1_dispatch)
        examples = step1_dilemmas.run(tiny_config, prompts_dad, tmp_path / "step1")
        scenario_ids = sorted(e["scenario_id"] for e in examples)
        draft_ids = sorted(c["item_id"] for c in calls if c["stage"] == "prompt_draft")
        assert draft_ids == scenario_ids
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["scenario_plan"]["item_id"] in scenario_ids
        assert by_stage["prompt_gate"]["item_id"] in scenario_ids

        calls = stub_claude(_dad_step2_dispatch)
        step2_responses.run(tiny_config, prompts_dad, tmp_path / "step2", [_dilemma()])
        by_stage = {c["stage"]: c for c in calls}
        assert by_stage["response_scope"]["item_id"] == "AW-0001"
        assert by_stage["response_draft"]["item_id"] == "AW-0001_s0"

        calls = stub_claude(["Rewritten careful answer."])
        step3_rewrite.run(tiny_config, prompts_dad, tmp_path / "step3", tmp_path / "final",
                          [_response_record()])
        assert calls[0]["item_id"] == "resp-1"

    def test_without_knobs_every_stage_falls_back_to_the_global_model(
        self, tiny_config, prompts_dad, tmp_path, stub_claude
    ):
        # model=None at the call site → call_claude resolves the global config
        # model; the stages must not invent their own fallback.
        calls = stub_claude(_dad_step1_dispatch)
        step1_dilemmas.run(tiny_config, prompts_dad, tmp_path / "step1")
        assert all(c["model"] is None for c in calls)
