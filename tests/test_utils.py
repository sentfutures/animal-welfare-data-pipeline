"""Tests for shared/utils.py: JSONL I/O, prompts, RNG, run dirs, checkpoints,
and the parallel_map helper the SDF layers fan out on."""

import json
import random
import re
import time

import pytest

from shared import utils


class TestParallelMap:
    @pytest.mark.parametrize("workers", [1, 4])
    def test_maps_all_items(self, workers):
        assert list(utils.parallel_map(lambda x: x * 2, [1, 2, 3], workers)) == [2, 4, 6]

    def test_results_come_back_in_input_order(self):
        # First item finishes last; order must still follow the input, because
        # callers zip() results with items to write files and mark checkpoints.
        def slow_first(x):
            if x == 0:
                time.sleep(0.02)
            return x

        assert list(utils.parallel_map(slow_first, [0, 1, 2, 3], workers=4)) == [0, 1, 2, 3]

    def test_worker_exception_propagates(self):
        def boom(x):
            if x == 2:
                raise ValueError("worker failed")
            return x

        with pytest.raises(ValueError, match="worker failed"):
            list(utils.parallel_map(boom, [1, 2, 3], workers=2))


class TestJsonl:
    def test_save_load_roundtrip_preserves_records(self, tmp_path):
        records = [{"id": 1, "text": "héllo wörld — 🐙"}, {"id": 2, "nested": {"a": [1, 2]}}]
        path = tmp_path / "out" / "data.jsonl"
        utils.save_jsonl(records, path)
        assert utils.load_jsonl(path) == records

    def test_save_jsonl_does_not_escape_unicode(self, tmp_path):
        path = tmp_path / "data.jsonl"
        utils.save_jsonl([{"text": "🐟"}], path)
        assert "🐟" in path.read_text(encoding="utf-8")

    def test_append_jsonl_extends_existing_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        utils.save_jsonl([{"id": 1}], path)
        utils.append_jsonl({"id": 2}, path)
        assert utils.load_jsonl(path) == [{"id": 1}, {"id": 2}]

    def test_load_jsonl_missing_file_returns_empty_list(self, tmp_path):
        assert utils.load_jsonl(tmp_path / "nope.jsonl") == []

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"id": 1}\n\n{"id": 2}\n\n')
        assert utils.load_jsonl(path) == [{"id": 1}, {"id": 2}]

    def test_load_jsonl_reads_utf8_regardless_of_locale(self, tmp_path):
        # Regression: pipeline files are UTF-8 by construction (ensure_ascii=False),
        # but reads without an explicit encoding= used the locale codec — cp1252 on
        # Windows — and crashed on the first non-cp1252 byte (curly quotes, emoji,
        # non-English text). Write raw UTF-8 bytes so no locale is involved.
        path = tmp_path / "data.jsonl"
        path.write_bytes('{"text": "curly “quotes” — 🐟"}\n'.encode("utf-8"))
        assert utils.load_jsonl(path) == [{"text": "curly “quotes” — 🐟"}]


PAYLOAD = [{"subtype_name": "River survey"}, {"subtype_name": "Coastal survey"}]


class TestExtractJson:
    """Model responses wrap JSON in fences or surround it with prose; a paid
    run must not crash on an otherwise usable response (a live claude_code run
    died at layer 2 with "Extra data" from a trailing sentence)."""

    def test_clean_json_passes_through(self):
        assert utils.extract_json(json.dumps(PAYLOAD)) == PAYLOAD
        assert utils.extract_json('{"alignment": 9}') == {"alignment": 9}

    def test_markdown_fences_tolerated(self):
        assert utils.extract_json("```json\n" + json.dumps(PAYLOAD) + "\n```") == PAYLOAD

    def test_trailing_prose_tolerated(self):
        text = json.dumps(PAYLOAD) + "\n\nLet me know if you'd like more subtypes."
        assert utils.extract_json(text) == PAYLOAD

    def test_leading_preamble_tolerated(self):
        text = "Here are the subtypes you asked for:\n" + json.dumps(PAYLOAD)
        assert utils.extract_json(text) == PAYLOAD

    def test_short_bracketed_aside_does_not_shadow_payload(self):
        # "[2]" is itself valid JSON; the longest parse must win.
        text = "I generated [2] subtypes:\n" + json.dumps(PAYLOAD) + "\nDone."
        assert utils.extract_json(text) == PAYLOAD

    def test_no_json_raises_jsondecodeerror(self):
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json("garbage")

    def test_control_characters_inside_strings_tolerated(self):
        # temperature-1 prose JSON often carries literal newlines inside values
        raw = '{"notes": "line one\nline two"}'
        assert utils.extract_json(raw)["notes"] == "line one\nline two"

    def test_shape_narrowed_helpers_raise_on_wrong_shape(self):
        # extract_json_object/_array turn shape mismatches into the same
        # JSONDecodeError as a parse failure, keeping callers on one error path
        assert utils.extract_json_object('{"a": 1}') == {"a": 1}
        assert utils.extract_json_array("[1, 2]") == [1, 2]
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_object("[1, 2]")
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_array('{"a": 1}')

    def test_truncated_json_raises_jsondecodeerror(self):
        # A cut-off payload contains complete inner objects; salvaging one
        # would hand the caller a wrong-shaped fragment, so this must raise.
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json(json.dumps(PAYLOAD)[:-10])

    def test_complete_payload_survives_truncated_trailing_chatter(self):
        # max_tokens can cut the response mid-sentence *after* the payload has
        # already closed; a dangling bracket in that chatter must not discard
        # the good parse (review finding on #59).
        assert utils.extract_json('[{"a": 1}]\n\nThanks for the {') == [{"a": 1}]
        assert utils.extract_json(json.dumps(PAYLOAD) + '\nAlso note ["unclosed') == PAYLOAD

    def test_malformed_array_raises_instead_of_returning_fragment(self):
        # Missing/trailing commas are common LLM slip-ups; the broken array
        # must raise like plain json.loads did, not silently return its first
        # inner object as a dict (second review finding on #59).
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json('[{"a": 1} {"b": 2}]')  # missing comma
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json('[{"a": 1}, {"b": 2},]')  # trailing comma

    def test_benign_bracket_blip_in_prose_does_not_block_payload(self):
        # A failed parse that never contained a complete value (a brace used
        # as prose punctuation) is not a broken payload — the real JSON wins.
        text = "Wrap it in {curly} braces:\n" + json.dumps(PAYLOAD)
        assert utils.extract_json(text) == PAYLOAD

    # --- recover=True: the shared JSON salvage the eval + step-1 both use ---

    def test_strict_default_still_refuses_wrong_shape(self):
        # recover defaults off; the strict no-coerce guarantee is unchanged
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_array('{"reasons": ["a", "b"]}')
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_object("[1, 2]")

    def test_recover_unwraps_object_wrapped_array(self):
        # the observed judge slip: {"reasons": [...]} instead of a bare array
        assert utils.extract_json_array('{"reasons": ["a", "b"]}', recover=True) == ["a", "b"]
        assert utils.extract_json_array('{"items": [1, 2, 3]}', recover=True) == [1, 2, 3]

    def test_recover_array_refuses_ambiguous_object(self):
        # never guess: no list value, or more than one, still raises
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_array('{"a": 1}', recover=True)
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_array('{"a": [1], "b": [2]}', recover=True)

    def test_recover_array_salvages_truncated_object_array(self):
        # a broken container still yields its complete objects
        assert utils.extract_json_array(
            '[{"x": 1}, {"y": 2}, {"z"', recover=True) == [{"x": 1}, {"y": 2}]

    def test_recover_array_salvages_bracketless_object_stream(self):
        # The live judge slip (11/79 calls on one eval pass): the objects come
        # back WITHOUT the opening '[' — sometimes without commas, sometimes
        # with a stray trailing ']'. extract_json parses one object here, so the
        # broken-container branch never fires; recovery must still read them all.
        assert utils.extract_json_array(
            '{"a": 1},\n{"b": 2}', recover=True) == [{"a": 1}, {"b": 2}]
        assert utils.extract_json_array(
            '{"a": 1}\n{"b": 2}', recover=True) == [{"a": 1}, {"b": 2}]   # commas dropped too
        assert utils.extract_json_array(
            '{"a": 1}, {"b": 2}]', recover=True) == [{"a": 1}, {"b": 2}]  # stray closer
        # a lone object is still ambiguous (could be a wrapper) -> still raises
        with pytest.raises(json.JSONDecodeError):
            utils.extract_json_array('{"a": 1}', recover=True)

    def test_recover_object_salvages_first_complete_object(self):
        # wrong-shape (list) or broken container -> first salvaged object
        assert utils.extract_json_object('[{"a": 1}, {"b": 2}]', recover=True) == {"a": 1}
        assert utils.extract_json_object('{"a": 1} then junk {', recover=True) == {"a": 1}

    def test_salvage_json_objects_brace_matches_complete_objects(self):
        assert utils.salvage_json_objects(
            '[{"a": 1}, {"b": 2}, {trunc') == [{"a": 1}, {"b": 2}]
        # a brace inside a string is not a container boundary
        assert utils.salvage_json_objects('[{"note": "a } brace"}]') == [{"note": "a } brace"}]
        assert utils.salvage_json_objects("no objects here") == []


class TestRepoRelative:
    """Paths written into reports/manifests must not carry the machine they ran
    on: audit reports are committed and published, so an absolute path would
    leak a home directory and username."""

    def test_path_inside_the_repo_becomes_relative(self):
        p = utils.REPO_ROOT / "outputs" / "dad" / "runs" / "2026-01-01_00-00_x"
        assert utils.repo_relative(p) == "outputs/dad/runs/2026-01-01_00-00_x"

    def test_relative_input_is_relativized_from_the_repo_root(self):
        assert utils.repo_relative("evals/audit_dad.py") == "evals/audit_dad.py"

    def test_symlink_is_resolved_to_its_target(self, tmp_path, monkeypatch):
        # `--input outputs/dad/latest` should record WHICH run was audited, not
        # the moving pointer, so the field still means something months later.
        target = utils.REPO_ROOT / "outputs" / "dad" / "runs" / "2026-01-01_00-00_x"
        link = tmp_path / "latest"
        link.symlink_to(target)
        assert utils.repo_relative(link) == "outputs/dad/runs/2026-01-01_00-00_x"

    def test_path_outside_the_repo_keeps_only_its_name(self, tmp_path):
        # a seed file or run dir elsewhere on disk: the parent directories say
        # nothing about the run and would carry the username
        outside = tmp_path / "scratchpad" / "seeds.jsonl"
        assert utils.repo_relative(outside) == "seeds.jsonl"

    def test_no_home_directory_survives(self):
        assert "/Users/" not in utils.repo_relative("~/Documents/elsewhere/run")


class TestLoadPrompt:
    def test_renders_placeholders(self, tmp_path):
        tpl = tmp_path / "t.txt"
        tpl.write_text("Hello {name}, count={count}")
        assert utils.load_prompt(tpl, name="world", count=3) == "Hello world, count=3"

    def test_without_kwargs_returns_verbatim(self, tmp_path):
        tpl = tmp_path / "t.txt"
        tpl.write_text("Literal {braces} untouched")
        assert utils.load_prompt(tpl) == "Literal {braces} untouched"

    def test_missing_placeholder_raises(self, tmp_path):
        tpl = tmp_path / "t.txt"
        tpl.write_text("Hello {name}")
        with pytest.raises(KeyError):
            utils.load_prompt(tpl, other="x")


class TestSampleLanguage:
    def test_certain_distribution_always_returns_that_language(self):
        assert all(utils.sample_language({"en": 1.0}) == "en" for _ in range(20))

    def test_repeatable_under_global_seed(self):
        dist = {"en": 0.5, "de": 0.5}
        random.seed(123)
        first = [utils.sample_language(dist) for _ in range(20)]
        random.seed(123)
        second = [utils.sample_language(dist) for _ in range(20)]
        assert first == second

    def test_injected_rng_gives_reproducible_sequence(self):
        dist = {"en": 0.5, "de": 0.5}
        seq1 = [utils.sample_language(dist, rng=random.Random(42)) for _ in range(5)]
        seq2 = [utils.sample_language(dist, rng=random.Random(42)) for _ in range(5)]
        assert seq1 == seq2


class TestRunDirs:
    def test_new_run_id_sanitizes_label(self):
        rid = utils.new_run_id("my run!/v2 ")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_my-run--v2", rid)

    def test_create_run_dir_writes_manifest(self, tmp_path):
        runs_root = tmp_path / "runs"
        config = {"model": "test-model", "foo": "bar"}
        run_dir = utils.create_run_dir(runs_root, label="dev", config=config)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["run_id"] == run_dir.name
        assert manifest["label"] == "dev"
        assert manifest["model"] == "test-model"
        assert manifest["config"] == config
        assert manifest["git_commit"] is None or isinstance(manifest["git_commit"], str)

    def test_create_run_dir_points_latest_symlink_at_run(self, tmp_path):
        runs_root = tmp_path / "runs"
        run_dir = utils.create_run_dir(runs_root, label="dev", config={})
        link = tmp_path / "latest"
        assert link.is_symlink()
        assert link.resolve() == run_dir.resolve()

    def test_create_run_dir_collision_appends_suffix(self, tmp_path, monkeypatch):
        # Pin the minted id so both calls collide regardless of wall clock
        monkeypatch.setattr(utils, "new_run_id", lambda label: "2026-01-01_00-00_dev")
        runs_root = tmp_path / "runs"
        first = utils.create_run_dir(runs_root, label="dev", config={})
        second = utils.create_run_dir(runs_root, label="dev", config={})
        assert first.name == "2026-01-01_00-00_dev"
        assert second.name == "2026-01-01_00-00_dev-2"
        assert (tmp_path / "latest").resolve() == second.resolve()

    def test_create_run_dir_manifest_records_git_state(self, tmp_path):
        run_dir = utils.create_run_dir(tmp_path / "runs", label="dev", config={})
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["manifest_version"] == 3
        assert manifest["inputs_snapshot"] is False
        assert isinstance(manifest["git_dirty"], bool)
        assert isinstance(manifest["git_dirty_files"], list)
        # Shape only: CI's actions/checkout leaves a detached HEAD, so the value
        # is the literal "HEAD" there and a branch name locally.
        assert manifest["git_branch"] is None or isinstance(manifest["git_branch"], str)

    def test_create_run_dir_snapshots_input_dirs(self, tmp_path):
        src = tmp_path / "src_prompts"
        src.mkdir()
        (src / "t.txt").write_text("template")
        run_dir = utils.create_run_dir(
            tmp_path / "runs", label="dev", config={}, snapshot_dirs={"prompts": src}
        )
        assert (run_dir / "inputs" / "prompts" / "t.txt").read_text() == "template"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["inputs_snapshot"] is True

    def test_resolve_run_dir_by_id(self, tmp_path):
        (tmp_path / "run_a").mkdir()
        assert utils.resolve_run_dir(tmp_path, "run_a") == tmp_path / "run_a"

    def test_resolve_run_dir_unknown_id_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            utils.resolve_run_dir(tmp_path, "missing")

    def test_resolve_run_dir_picks_latest_by_name(self, tmp_path):
        for name in ["2026-01-01_10-00_dev", "2026-01-02_09-00_dev", "2026-01-01_23-59_dev"]:
            (tmp_path / name).mkdir()
        (tmp_path / "stray.txt").write_text("not a dir")
        assert utils.resolve_run_dir(tmp_path).name == "2026-01-02_09-00_dev"

    def test_resolve_run_dir_empty_root_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            utils.resolve_run_dir(tmp_path / "does-not-exist")


class TestWarnIfBackendChanged:
    def test_warns_when_live_backend_differs_from_manifest(self, tmp_path, capsys):
        run_dir = utils.create_run_dir(tmp_path / "runs", label="dev", config={"backend": "claude_code"})
        utils.warn_if_backend_changed(run_dir, {"backend": "api"})
        assert "different backend" in capsys.readouterr().err

    def test_silent_when_backend_matches(self, tmp_path, capsys):
        run_dir = utils.create_run_dir(tmp_path / "runs", label="dev", config={"backend": "claude_code"})
        utils.warn_if_backend_changed(run_dir, {"backend": "claude_code"})
        assert capsys.readouterr().err == ""

    def test_manifest_without_backend_key_treated_as_api(self, tmp_path, capsys):
        # A run created before the backend key existed defaults to api.
        run_dir = utils.create_run_dir(tmp_path / "runs", label="dev", config={})
        utils.warn_if_backend_changed(run_dir, {"backend": "api"})
        assert capsys.readouterr().err == ""
        utils.warn_if_backend_changed(run_dir, {"backend": "claude_code"})
        assert "different backend" in capsys.readouterr().err

    def test_missing_manifest_is_silent(self, tmp_path, capsys):
        utils.warn_if_backend_changed(tmp_path / "no-such-run", {"backend": "api"})
        assert capsys.readouterr().err == ""


class TestResolveConstitutionDir:
    def test_returns_sibling_constitution_for_snapshot_prompts(self, tmp_path):
        prompts = tmp_path / "inputs" / "prompts"
        constitution = tmp_path / "inputs" / "constitution"
        prompts.mkdir(parents=True)
        constitution.mkdir()
        assert utils.resolve_constitution_dir(prompts) == constitution

    def test_returns_none_for_live_prompts_dir(self, tmp_path):
        live = tmp_path / "prompts" / "dad"
        live.mkdir(parents=True)
        assert utils.resolve_constitution_dir(live) is None

    def test_returns_none_when_snapshot_has_no_constitution(self, tmp_path):
        prompts = tmp_path / "inputs" / "prompts"
        prompts.mkdir(parents=True)
        assert utils.resolve_constitution_dir(prompts) is None


class TestCheckpoint:
    def test_starts_empty_without_file(self, tmp_path):
        cp = utils.Checkpoint(tmp_path / "_checkpoint.json")
        assert not cp.is_done("x")
        assert cp.done_count == 0

    def test_mark_done_persists_across_instances(self, tmp_path):
        path = tmp_path / "_checkpoint.json"
        utils.Checkpoint(path).mark_done("layer1")
        assert utils.Checkpoint(path).is_done("layer1")

    def test_ids_are_stringified(self, tmp_path):
        cp = utils.Checkpoint(tmp_path / "_checkpoint.json")
        cp.mark_done(3)
        assert cp.is_done(3)
        assert cp.is_done("3")

    def test_done_count_ignores_duplicate_marks(self, tmp_path):
        cp = utils.Checkpoint(tmp_path / "_checkpoint.json")
        cp.mark_done("a")
        cp.mark_done("b")
        cp.mark_done("a")
        assert cp.done_count == 2

    def test_creates_parent_dirs_on_first_mark(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "_checkpoint.json"
        utils.Checkpoint(path).mark_done("x")
        assert path.exists()


class TestLooksLikeTranscriptEcho:
    def test_flags_role_markers_at_start(self):
        assert utils.looks_like_transcript_echo("USER: hi\nASSISTANT: reply")
        assert utils.looks_like_transcript_echo("  ASSISTANT: reply text")
        assert utils.looks_like_transcript_echo("HUMAN: question")

    def test_normal_replies_pass(self):
        assert not utils.looks_like_transcript_echo("Lead with the honest version.")
        # a role marker mentioned mid-reply is not an echo
        assert not utils.looks_like_transcript_echo(
            "The form has a field labeled USER: fill it in truthfully.")
        assert not utils.looks_like_transcript_echo("")

def test_resolve_run_dir_ignores_handmade_local_dirs(tmp_path):
    """A local_* scratch dir sorts after every timestamp name; bare --resume
    must still pick the newest pipeline-created run (the 2026-07-11 incident)."""
    (tmp_path / "2026-07-10_09-00_dev").mkdir()
    (tmp_path / "2026-07-11_20-06_matrix100").mkdir()
    (tmp_path / "local_2026-07-11_scratch").mkdir()
    picked = utils.resolve_run_dir(tmp_path)
    assert picked.name == "2026-07-11_20-06_matrix100"
    # explicit --run-id still reaches hand-made dirs
    assert utils.resolve_run_dir(tmp_path, "local_2026-07-11_scratch").name == "local_2026-07-11_scratch"


# --- SDF layer renumber compatibility ----------------------------------
#
# The layers were renumbered 1-2/3/4/5 -> 1/2/3/4. The layouts OVERLAP: old
# layer3 holds drafts, new layer3 holds rewrites. These pin that a pre-renumber
# run is never read through the new meaning of a name it shares.

class TestSdfStageFile:
    def _old_run(self, tmp_path):
        """A run dir in the pre-renumber layout."""
        for d, f in (("layer12", "plans.jsonl"), ("layer12", "prompts.jsonl"),
                     ("layer3", "drafts.jsonl"), ("layer4", "rewrites.jsonl"),
                     ("layer5", "scores.jsonl")):
            (tmp_path / d).mkdir(exist_ok=True)
            (tmp_path / d / f).write_text('{"a": 1}\n', encoding="utf-8")
        return tmp_path

    def _new_run(self, tmp_path):
        for d, f in (("layer1", "plans.jsonl"), ("layer1", "prompts.jsonl"),
                     ("layer2", "drafts.jsonl"), ("layer3", "rewrites.jsonl"),
                     ("layer4", "scores.jsonl")):
            (tmp_path / d).mkdir(exist_ok=True)
            (tmp_path / d / f).write_text('{"a": 1}\n', encoding="utf-8")
        return tmp_path

    def test_a_pre_renumber_run_resolves_to_its_own_directories(self, tmp_path):
        run = self._old_run(tmp_path)
        assert utils.sdf_stage_file(run, "plan") == run / "layer12" / "plans.jsonl"
        assert utils.sdf_stage_file(run, "draft") == run / "layer3" / "drafts.jsonl"
        assert utils.sdf_stage_file(run, "rewrite") == run / "layer4" / "rewrites.jsonl"
        assert utils.sdf_stage_file(run, "score") == run / "layer5" / "scores.jsonl"

    def test_an_old_runs_drafts_are_never_read_as_its_rewrites(self, tmp_path):
        """layer3/ is drafts in an old run and rewrites in a new one. A resolver
        that matched on directory alone would return the drafts here."""
        run = self._old_run(tmp_path)
        assert utils.sdf_stage_file(run, "rewrite").name == "rewrites.jsonl"
        assert "layer3" not in utils.sdf_stage_file(run, "rewrite").parts

    def test_a_current_run_resolves_to_the_new_directories(self, tmp_path):
        run = self._new_run(tmp_path)
        assert utils.sdf_stage_file(run, "draft") == run / "layer2" / "drafts.jsonl"
        assert utils.sdf_stage_file(run, "rewrite") == run / "layer3" / "rewrites.jsonl"
        assert utils.sdf_stage_file(run, "score") == run / "layer4" / "scores.jsonl"

    def test_an_empty_run_reports_through_the_current_layout(self, tmp_path):
        assert utils.sdf_stage_file(tmp_path, "score") == tmp_path / "layer4" / "scores.jsonl"

    def test_stage_dir_writes_beside_a_resumed_old_runs_own_dirs(self, tmp_path):
        run = self._old_run(tmp_path)
        assert utils.sdf_stage_dir(run, 1) == run / "layer12"
        assert utils.sdf_stage_dir(run, 4) == run / "layer5"

    def test_stage_dir_gives_a_fresh_run_the_new_names(self, tmp_path):
        assert utils.sdf_stage_dir(tmp_path, 1) == tmp_path / "layer1"
        assert utils.sdf_stage_dir(tmp_path, 4) == tmp_path / "layer4"


class TestSdfTemplatePath:
    def test_a_pre_renumber_snapshot_reads_its_own_template_names(self, tmp_path):
        for n in ("layers1-2.txt", "layer3.txt", "layer4.txt", "layer5.txt"):
            (tmp_path / n).write_text("x", encoding="utf-8")
        assert utils.sdf_template_path(tmp_path, 1).name == "layers1-2.txt"
        assert utils.sdf_template_path(tmp_path, 2).name == "layer3.txt"
        assert utils.sdf_template_path(tmp_path, 3).name == "layer4.txt"
        assert utils.sdf_template_path(tmp_path, 4).name == "layer5.txt"

    def test_a_current_snapshot_reads_the_new_names(self, tmp_path):
        for n in ("layer1.txt", "layer2.txt", "layer3.txt", "layer4.txt"):
            (tmp_path / n).write_text("x", encoding="utf-8")
        assert utils.sdf_template_path(tmp_path, 1).name == "layer1.txt"
        assert utils.sdf_template_path(tmp_path, 2).name == "layer2.txt"
        assert utils.sdf_template_path(tmp_path, 4).name == "layer4.txt"
