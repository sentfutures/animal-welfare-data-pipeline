"""Viewer loader: per-stage cost aggregation over run cost logs.

The loader is deliberately streamlit-free (viewer/loader.py docstring), so it
is testable like any other module.
"""

import json

from viewer import loader


def _write_log(tmp_path, records):
    path = tmp_path / "cost_log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return tmp_path


class TestCostByStage:
    def test_aggregates_calls_cost_and_models_per_stage(self, tmp_path):
        run = _write_log(tmp_path, [
            {"stage": "prompt_draft", "cost_usd": 0.10, "model": "m1"},
            {"stage": "prompt_draft", "cost_usd": 0.20, "model": "m1"},
            {"stage": "constitution_rewrite", "cost_usd": 0.30, "model": "m2"},
        ])
        by = loader.cost_by_stage(run)
        assert by["prompt_draft"] == {"calls": 2, "cost_usd": 0.3, "models": ["m1"]}
        assert by["constitution_rewrite"] == {"calls": 1, "cost_usd": 0.3, "models": ["m2"]}

    def test_legacy_records_without_stage_group_as_untagged(self, tmp_path):
        # every committed run predates stage tags — the viewer must render them
        run = _write_log(tmp_path, [{"cost_usd": 0.5, "model": "m"}])
        assert loader.cost_by_stage(run) == {
            "(untagged)": {"calls": 1, "cost_usd": 0.5, "models": ["m"]}
        }

    def test_missing_log_is_empty(self, tmp_path):
        assert loader.cost_by_stage(tmp_path) == {}


class TestCallStats:
    def test_per_item_rows_are_aggregated(self, tmp_path):
        # two calls served AW-0001 (caller-level retry) — sum cost/time,
        # count API retries across both, report the call count
        run = _write_log(tmp_path, [
            {"stage": "response_scope", "item_id": "AW-0001", "cost_usd": 0.10,
             "model": "m1", "duration_s": 4.0, "attempts": 1},
            {"stage": "response_scope", "item_id": "AW-0001", "cost_usd": 0.20,
             "model": "m1", "duration_s": 6.5, "attempts": 3},
            {"stage": "response_scope", "item_id": "AW-0002", "cost_usd": 9.0,
             "model": "m1", "duration_s": 9.0, "attempts": 1},
        ])
        s = loader.call_stats(run, "response_scope", "AW-0001")
        assert s == {"per_item": True, "calls": 2, "models": ["m1"],
                     "cost_usd": 0.3, "duration_s": 10.5, "retries": 2,
                     "batch_size": 1}

    def test_batched_item_id_matches_by_membership(self, tmp_path):
        run = _write_log(tmp_path, [
            {"stage": "prompt_draft", "item_id": "S-01,S-02,S-03", "cost_usd": 0.5,
             "model": "m1", "duration_s": 20.0, "attempts": 1},
        ])
        s = loader.call_stats(run, "prompt_draft", "S-02")
        assert s["per_item"] is True and s["batch_size"] == 3
        assert loader.call_stats(run, "prompt_draft", "S-99") is None

    def test_legacy_rows_fall_back_to_stage_wide(self, tmp_path):
        # rows logged before item_id/duration/attempts existed: report the
        # stage's models but flag that the numbers aren't this record's
        run = _write_log(tmp_path, [
            {"stage": "constitution_rewrite", "cost_usd": 0.30, "model": "m2"},
        ])
        s = loader.call_stats(run, "constitution_rewrite", "some-response-id")
        assert s["per_item"] is False and s["models"] == ["m2"]
        assert "duration_s" not in s and "retries" not in s

    def test_unknown_stage_or_missing_log_is_none(self, tmp_path):
        assert loader.call_stats(tmp_path, "response_scope", "AW-0001") is None
        run = _write_log(tmp_path, [{"stage": "prompt_draft", "cost_usd": 0.1, "model": "m"}])
        assert loader.call_stats(run, "response_scope", "AW-0001") is None


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _write_dad_run(root, name, examples):
    """Minimal DAD run: step-1 dilemmas + step-3 rewrites + final corpus.
    Each example dict: prompt_id, user_message, and optional goal/scenario_id/response."""
    run = root / name
    dil, rew, fin = [], [], []
    for i, e in enumerate(examples):
        rid = f"{name}-rec-{i}"
        ann = {"dilemma_anatomy": {"goal": e.get("goal", "")}}
        resp = e.get("response", f"response-{i}")
        dil.append({"prompt_id": e["prompt_id"], "user_message": e["user_message"],
                    "annotation": ann, "scenario_gid": e.get("scenario_gid"),
                    "prompt_gid": e.get("prompt_gid")})
        rew.append({"record_id": rid, "prompt_id": e["prompt_id"], "sample_index": 0,
                    "example_gid": e.get("example_gid"),
                    "user_message": e["user_message"], "annotation": ann,
                    "rewritten_response": resp, "draft_response": "draft"})
        fin.append({"record_id": rid,
                    "messages": [{"role": "user", "content": e["user_message"]},
                                 {"role": "assistant", "content": resp}]})
    _write_jsonl(run / "step1" / "dilemmas.jsonl", dil)
    _write_jsonl(run / "step3" / "rewrites.jsonl", rew)
    _write_jsonl(run / "final" / "dad_corpus.jsonl", fin)
    return run


class TestDadGoalLabel:
    def test_goal_then_fallback_to_first_line(self):
        assert loader.dad_goal_label({"dilemma_anatomy": {"goal": "decide X"}}, "ignored") == "decide X"
        assert loader.dad_goal_label({}, "# Title\nbody") == "Title"
        assert loader.dad_goal_label(None, "") == "(untitled)"


class TestMatchDad:
    def test_user_message_pairs_across_different_awids(self, tmp_path):
        # The core fix: same prompt text under different AW-#### ids still pairs.
        a = _write_dad_run(tmp_path, "A", [
            {"prompt_id": "AW-0001", "user_message": "Same text", "goal": "gA", "response": "respA"}])
        b = _write_dad_run(tmp_path, "B", [
            {"prompt_id": "AW-0999", "user_message": "Same text", "goal": "gB", "response": "respB"}])
        matched, only_a, only_b = loader.match_dad(a, b, "user_message")
        assert len(matched) == 1 and not only_a and not only_b
        m = matched[0]
        assert m.same_prompt is True
        assert (m.a.prompt_id, m.b.prompt_id) == ("AW-0001", "AW-0999")
        assert (m.a.response, m.b.response) == ("respA", "respB")
        assert m.label == "gA"

    def test_user_message_does_not_pair_awid_collisions(self, tmp_path):
        # Same AW-#### id but different prompt text must NOT pair (the bug we fix).
        a = _write_dad_run(tmp_path, "A", [{"prompt_id": "AW-0001", "user_message": "Prompt Alpha"}])
        b = _write_dad_run(tmp_path, "B", [{"prompt_id": "AW-0001", "user_message": "Prompt Beta"}])
        matched, only_a, only_b = loader.match_dad(a, b, "user_message")
        assert matched == []
        assert len(only_a) == 1 and len(only_b) == 1

    def test_prompt_id_mode_pairs_by_awid_and_flags_prompt_diff(self, tmp_path):
        a = _write_dad_run(tmp_path, "A", [{"prompt_id": "AW-0001", "user_message": "Prompt Alpha"}])
        b = _write_dad_run(tmp_path, "B", [{"prompt_id": "AW-0001", "user_message": "Prompt Beta"}])
        matched, _, _ = loader.match_dad(a, b, "prompt_id")
        assert len(matched) == 1 and matched[0].same_prompt is False

    def test_scenario_id_mode_pairs_differing_prompts(self, tmp_path):
        # Prompt-tuning mode: same scenario, different prompt text → pairs, flagged.
        a = _write_dad_run(tmp_path, "A", [
            {"prompt_id": "AW-0001", "scenario_gid": "S-0001", "user_message": "A phrasing"}])
        b = _write_dad_run(tmp_path, "B", [
            {"prompt_id": "AW-0042", "scenario_gid": "S-0001", "user_message": "B phrasing"}])
        matched, _, _ = loader.match_dad(a, b, "scenario_id")
        assert len(matched) == 1 and matched[0].same_prompt is False
        assert loader.run_has_scenario_ids(a) and loader.run_has_scenario_ids(b)

    def test_run_without_scenario_ids(self, tmp_path):
        a = _write_dad_run(tmp_path, "A", [{"prompt_id": "AW-0001", "user_message": "x"}])
        assert loader.run_has_scenario_ids(a) is False


class TestDadLineageBaseline:
    def test_baseline_joins_by_prompt_id_in_both_lineage_shapes(self, tmp_path):
        run = _write_dad_run(tmp_path, "A", [{"prompt_id": "AW-0001", "user_message": "U"}])
        _write_jsonl(run / "baseline" / "baseline_responses.jsonl",
                     [{"prompt_id": "AW-0001", "user_message": "U",
                       "baseline_response": "plain answer", "model": "m-base"}])
        lin = loader.dad_lineage(run, "A-rec-0")
        assert lin["baseline"]["baseline_response"] == "plain answer"
        lin = loader.dad_lineage_by_prompt(run, "AW-0001")
        assert lin["baseline"]["baseline_response"] == "plain answer"

    def test_runs_without_baseline_records_get_none(self, tmp_path):
        run = _write_dad_run(tmp_path, "A", [{"prompt_id": "AW-0001", "user_message": "U"}])
        assert loader.dad_lineage(run, "A-rec-0")["baseline"] is None


class TestDadExampleLabels:
    def test_maps_prompt_ids_to_example_gids_skipping_unlabeled(self, tmp_path):
        run = _write_dad_run(tmp_path, "A", [
            {"prompt_id": "AW-0001", "user_message": "u1", "example_gid": "E-0042"},
            {"prompt_id": "AW-0002", "user_message": "u2"},  # pre-gid record
        ])
        assert loader.dad_example_labels(run) == {"AW-0001": "E-0042"}

    def test_run_without_rewrites_gives_empty_map(self, tmp_path):
        assert loader.dad_example_labels(tmp_path / "nothing") == {}


class TestLoadAudit:
    def test_missing_report_returns_none(self, tmp_path):
        assert loader.load_audit(tmp_path) is None

    def test_reads_report_json(self, tmp_path):
        (tmp_path / "audit").mkdir()
        (tmp_path / "audit" / "audit_report.json").write_text(
            json.dumps({"n_prompts": 3, "sections": []}), encoding="utf-8")
        assert loader.load_audit(tmp_path) == {"n_prompts": 3, "sections": []}


class TestLoadDiversity:
    def test_missing_returns_none(self, tmp_path):
        assert loader.load_diversity(tmp_path) is None

    def test_reads_report_json(self, tmp_path):
        (tmp_path / "audit").mkdir()
        (tmp_path / "audit" / "diversity_report.json").write_text(
            json.dumps({"embed_model": "m", "sections": []}), encoding="utf-8")
        assert loader.load_diversity(tmp_path)["embed_model"] == "m"




class TestListRunsAcrossProfiles:
    def _run(self, root, subdir, run_id, profile):
        d = root / subdir / "runs" / run_id
        d.mkdir(parents=True)
        (d / "run_manifest.json").write_text(json.dumps({"label": run_id, "config": {"dad": {"profile": profile}}}))
        return d

    def test_human_profile_runs_list_as_dad_runs_newest_first(self, tmp_path):
        self._run(tmp_path, "dad", "2026-10-01_10-00_animal", "animal")
        human = self._run(tmp_path, "dad_human", "2026-10-20_10-00_local_human", "human")
        runs = loader.list_runs(tmp_path)
        assert [(r.pipeline, r.run_id) for r in runs] == [
            ("dad", "2026-10-20_10-00_local_human"), ("dad", "2026-10-01_10-00_animal")]
        assert runs[0].run_dir == human
        assert loader.get_run("dad", "2026-10-20_10-00_local_human", tmp_path).run_dir == human
