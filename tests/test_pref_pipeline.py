"""Money-path tests for the preference pipeline: pair generation with per-arm
checkpointing, and the Streamlit-free rating/export logic in prefdata."""

import json

import pytest

import pref_pipeline.run as pref_run
from pref_pipeline import prefdata
from shared import utils


@pytest.fixture
def outputs_root(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(root))
    return root


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [{"prompt_id": f"P-{i}", "user_message": f"Question {i}?"} for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture
def pref_config_file(tiny_config, tmp_path):
    tiny_config["pref"] = {
        "prompts_path": None,
        "arms": {
            "a": {"name": "baseline", "system_prompt": ""},
            "b": {"name": "candidate", "system_prompt": "Reason with care."},
        },
    }
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path


def _run_main(monkeypatch, config_file, prompts_file, *extra):
    monkeypatch.setattr("sys.argv", ["run.py", "--config", str(config_file),
                                     "--prompts", str(prompts_file), "--label", "t", *extra])
    pref_run.main()


def _arm_dispatch(user_message, **kw):
    # arm b carries the only non-empty system prompt
    return f"{'B' if kw['system_prompt'] else 'A'} answer to {user_message}"


class TestPairGeneration:
    def test_two_arms_per_prompt(self, pref_config_file, prompts_file, outputs_root,
                                 stub_claude, monkeypatch):
        calls = stub_claude(_arm_dispatch)
        _run_main(monkeypatch, pref_config_file, prompts_file)

        run_dir = next((outputs_root / "pref" / "runs").iterdir())
        pairs = utils.load_jsonl(run_dir / "pairs" / "pairs.jsonl")
        assert len(pairs) == 2 and len(calls) == 4
        for p in pairs:
            assert p["response_a"].startswith("A answer")
            assert p["response_b"].startswith("B answer")
            assert p["left_arm"] in ("a", "b")
        # arms frozen for resume
        assert (run_dir / "inputs" / "arm_prompts.yaml").exists()

    def test_per_arm_effort_reaches_the_call(self, tiny_config, prompts_file, outputs_root,
                                             stub_claude, monkeypatch, tmp_path):
        # effort is a per-arm knob like model/max_tokens: sent for the arm that
        # sets it, left unset (backend default) for the one that doesn't, and
        # frozen into the run's arm_prompts.yaml so --resume replays it.
        import yaml
        tiny_config["pref"] = {"arms": {
            "a": {"name": "opus", "model": "claude-opus-5-5", "effort": "medium"},
            "b": {"name": "sonnet", "model": "claude-sonnet-5-5"},
        }}
        cfg = tmp_path / "config_effort.yaml"
        cfg.write_text(yaml.safe_dump(tiny_config))
        calls = stub_claude(lambda user_message, **kw: f"{kw['model']} answer")
        _run_main(monkeypatch, cfg, prompts_file, "--limit", "1")

        by_model = {c["model"]: c for c in calls}
        assert by_model["claude-opus-5-5"]["effort"] == "medium"
        assert by_model["claude-sonnet-5-5"]["effort"] is None
        run_dir = next((outputs_root / "pref" / "runs").iterdir())
        frozen = yaml.safe_load((run_dir / "inputs" / "arm_prompts.yaml").read_text())
        assert frozen["a"]["effort"] == "medium" and frozen["b"]["effort"] is None

    def test_unknown_effort_level_is_refused_before_any_call(self, tiny_config, prompts_file,
                                                             outputs_root, stub_claude,
                                                             monkeypatch, tmp_path):
        import yaml
        tiny_config["pref"] = {"arms": {
            "a": {"name": "a", "effort": "maximum"},
            "b": {"name": "b"},
        }}
        cfg = tmp_path / "config_bad_effort.yaml"
        cfg.write_text(yaml.safe_dump(tiny_config))
        calls = stub_claude(_arm_dispatch)
        with pytest.raises(SystemExit, match="effort must be one of"):
            _run_main(monkeypatch, cfg, prompts_file)
        assert calls == []

    def test_failed_arm_keeps_sibling_and_resume_retries_only_it(
        self, pref_config_file, prompts_file, outputs_root, stub_claude, monkeypatch
    ):
        state = {"b_fails": True}

        def flaky(user_message, **kw):
            is_b = bool(kw["system_prompt"])
            if is_b and state["b_fails"]:
                raise RuntimeError("arm b exploded")
            return f"{'B' if is_b else 'A'} answer"

        stub_claude(flaky)
        _run_main(monkeypatch, pref_config_file, prompts_file, "--limit", "1")
        run_dir = next((outputs_root / "pref" / "runs").iterdir())
        # no pair written, but arm A's paid response is cached
        assert utils.load_jsonl(run_dir / "pairs" / "pairs.jsonl") == []
        cache = utils.load_jsonl(run_dir / "pairs" / "arm_responses.jsonl")
        assert [c["arm"] for c in cache] == ["a"]

        # resume: arm b recovers; arm a must NOT be re-billed
        state["b_fails"] = False
        calls = stub_claude(flaky)
        _run_main(monkeypatch, pref_config_file, prompts_file, "--limit", "1", "--resume")
        assert len(calls) == 1  # only arm b
        pairs = utils.load_jsonl(run_dir / "pairs" / "pairs.jsonl")
        assert len(pairs) == 1 and pairs[0]["response_a"] == "A answer"

    def test_resume_warns_when_backend_changed(self, pref_config_file, prompts_file,
                                               outputs_root, stub_claude, monkeypatch,
                                               capsys, tmp_path):
        # Same resume convention as the SDF/DAD orchestrators: flipping
        # `backend` between start and --resume must be surfaced, not silent.
        import yaml
        stub_claude(_arm_dispatch)
        _run_main(monkeypatch, pref_config_file, prompts_file)
        assert "different backend" not in capsys.readouterr().err

        cfg = yaml.safe_load(pref_config_file.read_text())
        cfg["backend"] = "claude_code"
        flipped = tmp_path / "config_flipped.yaml"
        flipped.write_text(yaml.safe_dump(cfg))
        _run_main(monkeypatch, flipped, prompts_file, "--resume")
        assert "different backend" in capsys.readouterr().err

    def test_truncated_arm_defers_pair(self, pref_config_file, prompts_file, outputs_root,
                                       stub_claude, monkeypatch):
        def truncating(user_message, **kw):
            return ("B cut off", "max_tokens") if kw["system_prompt"] else "A answer"

        stub_claude(truncating)
        _run_main(monkeypatch, pref_config_file, prompts_file, "--limit", "1")
        run_dir = next((outputs_root / "pref" / "runs").iterdir())
        assert utils.load_jsonl(run_dir / "pairs" / "pairs.jsonl") == []


class TestPrefData:
    @pytest.fixture
    def rated_run(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "pairs").mkdir(parents=True)
        pairs = [
            {"pair_id": "pair_P-0", "prompt_id": "P-0", "user_message": "Q0?",
             "arm_names": {"a": "baseline", "b": "candidate"},
             "response_a": "A0", "response_b": "B0", "left_arm": "a"},
            {"pair_id": "pair_P-1", "prompt_id": "P-1", "user_message": "Q1?",
             "arm_names": {"a": "baseline", "b": "candidate"},
             "response_a": "A1", "response_b": "B1", "left_arm": "b"},
        ]
        utils.save_jsonl(pairs, run_dir / "pairs" / "pairs.jsonl")
        return run_dir, pairs

    def test_decisive_rating_deblinds_correctly(self, rated_run):
        run_dir, pairs = rated_run
        # pair 1: left is arm b — choosing "left" must credit arm b
        rating = prefdata.record_rating(run_dir, pairs[1], rater="r1", choice="left")
        assert rating["chosen_arm"] == "b"
        prefs = utils.load_jsonl(run_dir / "final" / "preferences.jsonl")
        assert len(prefs) == 1
        assert prefs[0]["chosen"] == "B1" and prefs[0]["rejected"] == "A1"

    def test_ties_and_both_bad_are_excluded_from_export(self, rated_run):
        run_dir, pairs = rated_run
        prefdata.record_rating(run_dir, pairs[0], rater="r1", choice="tie")
        prefdata.record_rating(run_dir, pairs[1], rater="r1", choice="both_bad")
        assert utils.load_jsonl(run_dir / "final" / "preferences.jsonl") == []

    def test_invalid_choice_rejected(self, rated_run):
        run_dir, pairs = rated_run
        with pytest.raises(ValueError):
            prefdata.record_rating(run_dir, pairs[0], rater="r1", choice="middle")

    def test_side_assignment_is_deterministic_and_salted(self):
        a = pref_run._left_arm("pair_X", salt="run1")
        assert a == pref_run._left_arm("pair_X", salt="run1")  # stable across reloads
        differs = any(pref_run._left_arm(f"pair_{i}", salt="run1")
                      != pref_run._left_arm(f"pair_{i}", salt="run2") for i in range(16))
        assert differs, "salt has no effect on side assignment"
