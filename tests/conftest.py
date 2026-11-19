"""Shared fixtures for the offline test suite. Tests can never call the API or spawn the Claude Code CLI: sockets are disabled, a fake key overrides the real one, and
``_call_with_retry`` and ``_call_claude_code_with_retry`` raise. Stub at ``shared.api.call_claude`` (``stub_claude``), never below it, or tenacity sleeps; embeddings are stubbed at ``shared.embeddings.embed_texts``."""

import json
import random
import re
import threading
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from shared import api, embeddings, gemini_api, openai_api

REPO_ROOT = Path(__file__).resolve().parent.parent


def dad_scenario_plan_reply(user_message: str) -> str:
    """Echo a conforming step-1a plan reply for a rendered step1a_scenario.txt
    prompt: working notes plus a tagged self-contained scenario description."""
    return ("<scenario_planning>Considered five situations; chose the second."
            "</scenario_planning>\n"
            "<scenario_description>A person faces a concrete decision where the "
            "tempting option quietly runs its cost through the animals involved."
            "</scenario_description>")


def dad_scenario_reply(user_message: str) -> str:
    """A conforming step-1b reply for a rendered step1b_dilemmas.txt prompt: the drafted user message inside <user_prompt> tags."""
    return ("<user_prompt>Drafted user message from the scenario description. "
            "The situation keeps going with believable texture.</user_prompt>")


@pytest.fixture(autouse=True)
def _seed_rng():
    """Reseed the global RNG before every test (draws are still not deterministic across worker threads; inject an rng for that)."""
    random.seed(0)


@pytest.fixture(autouse=True)
def _no_real_dotenv_and_fresh_key_announcements(monkeypatch, tmp_path_factory):
    """Point the key loader's repo root at an empty folder so no test ever reads a real .env,
    and reset its once-per-variable printing."""
    from shared import api_keys
    monkeypatch.setattr(api_keys, "REPO_ROOT", tmp_path_factory.mktemp("repo_root_without_dotenv"))
    monkeypatch.setattr(api_keys, "_sources_already_announced", set())
    # Tests supply fake keys through the environment (the repo root has no .env), so they use the mode that falls back to it.
    monkeypatch.setattr(api_keys, "_api_key_source_mode", api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT)
    # That mode searches upward; confine it to the empty folder so a .env above the base temporary folder is never read.
    def _dotenv_path_in_repo_root_only():
        candidate = api_keys.REPO_ROOT / ".env"
        return candidate.resolve() if candidate.is_file() else None
    monkeypatch.setattr(api_keys, "_nearest_dotenv_path_searching_upward", _dotenv_path_in_repo_root_only)
    # Orchestrator tests call main(), whose flag handling would reset that mode to the default.
    monkeypatch.setattr(api_keys, "set_api_key_source_mode", lambda mode: None)


@pytest.fixture(autouse=True)
def _api_guard(monkeypatch):
    """Fake credentials, reset shared.api globals, and block the API seam."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.setattr(api, "_config", {})
    monkeypatch.setattr(api, "_client", None)
    monkeypatch.setattr(api, "_cost_log_path", None)
    monkeypatch.setattr(api, "_record_model_calls", True)
    monkeypatch.setattr(api, "_UNPRICED_WARNED", set())
    monkeypatch.setattr(api, "_backend", "api")
    monkeypatch.setattr(api, "_cc_demoted", None)
    monkeypatch.setattr(api, "_claude_code_oauth_token", None)
    monkeypatch.setattr(api, "_neutral_system_warned", False)
    monkeypatch.setattr(api, "_temperature_warned", False)

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "Anthropic API call attempted during tests — stub shared.api.call_claude"
        )

    monkeypatch.setattr(api, "_call_with_retry", _blocked)

    # The claude_code seam spawns the real `claude` CLI, which --disable-socket cannot block; block it here so a test that flips _backend without stubbing this fails in-process.
    def _blocked_cc(*args, **kwargs):
        raise AssertionError(
            "claude_code backend invoked during tests — "
            "stub shared.api._call_claude_code_with_retry"
        )

    monkeypatch.setattr(api, "_call_claude_code_with_retry", _blocked_cc)


@pytest.fixture(autouse=True)
def _reset_dad_profile():
    """The active DAD profile is process state (run.py sets it): start and end
    every test on the default (animal) profile."""
    from dad_pipeline import profiles
    profiles.reset()
    yield
    profiles.reset()


@pytest.fixture(autouse=True)
def _reset_sdf_profile():
    """The active SDF profile is process state (run.py sets it): start and end
    every test on the default (animal) profile."""
    from sdf_pipeline import profiles
    profiles.reset()
    yield
    profiles.reset()


@pytest.fixture(autouse=True)
def _openai_guard(monkeypatch):
    """Fake embedding-provider credentials (both legs), reset shared.embeddings
    globals, block the provider-routing seam."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-gemini-key")
    monkeypatch.delenv("EMBEDDINGS_MODEL", raising=False)
    monkeypatch.setattr(embeddings, "_config", {})
    monkeypatch.setattr(embeddings, "_client", None)
    monkeypatch.setattr(embeddings, "_cost_log_path", None)
    monkeypatch.setattr(embeddings, "_UNPRICED_WARNED", set())

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "Embedding API call attempted during tests — stub shared.embeddings.embed_texts"
        )

    monkeypatch.setattr(embeddings, "_embed_with_retry", _blocked)


@pytest.fixture(autouse=True)
def _openai_responses_guard(monkeypatch):
    """Reset shared.openai_api's client and block its network seam, so an
    OpenAI-routed stage can never reach the API from a test."""
    monkeypatch.setattr(openai_api, "_client", None)

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "OpenAI Responses call attempted during tests — use stub_openai_responses"
        )

    monkeypatch.setattr(openai_api, "_create_response", _blocked)


@pytest.fixture(autouse=True)
def _gemini_guard(monkeypatch):
    """Block shared.gemini_api's network seam, so a Gemini-routed stage can
    never reach the API from a test."""
    def _blocked(*args, **kwargs):
        raise AssertionError(
            "Gemini generateContent call attempted during tests — stub shared.gemini_api._generate"
        )

    monkeypatch.setattr(gemini_api, "_generate", _blocked)


@pytest.fixture
def stub_openai_responses(monkeypatch):
    """Factory replacing ``shared.openai_api._create_response`` with a recording stub: ``install(responses)`` takes a callable or a list; returns the recorded request arguments."""

    def install(responses):
        calls = []
        queue = list(responses) if isinstance(responses, list) else None

        def fake(client, **request_arguments):
            calls.append(request_arguments)
            return queue.pop(0) if queue is not None else responses(request_arguments)

        monkeypatch.setattr(openai_api, "_create_response", fake)
        return calls

    return install


@pytest.fixture
def openai_response():
    """Factory for objects shaped like an OpenAI Responses API response."""

    def make(text="ok", input_tokens=1000, cached_tokens=0, output_tokens=500, reasoning_tokens=300,
             status="completed", incomplete_reason=None, refusal=None):
        part = (SimpleNamespace(type="refusal", refusal=refusal) if refusal is not None
                else SimpleNamespace(type="output_text", text=text))
        return SimpleNamespace(
            status=status,
            incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
            output=[SimpleNamespace(type="message", content=[part])],
            usage=SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens,
                input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
                output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens)),
        )

    return make


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Factory replacing ``shared.embeddings.embed_texts`` with a recording stub: ``install()`` gives each distinct text a deterministic unit vector (``vectors`` pins exact geometry); returns the recorded calls."""

    def install(dim: int = 8, vectors: dict | None = None):
        calls = []

        def fake(texts, model=embeddings.DEFAULT_MODEL):
            # mirror the real embed_texts contract: empties must never reach the API
            for i, t in enumerate(texts):
                if not t or not t.strip():
                    raise ValueError(f"embed_texts got an empty text at index {i}")
            calls.append({"texts": list(texts), "model": model})
            rows = []
            for t in texts:
                if vectors is not None:
                    v = np.asarray(vectors[t], dtype=np.float32)
                else:
                    rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))
                    v = rng.standard_normal(dim).astype(np.float32)
                rows.append(v / np.linalg.norm(v))
            return np.stack(rows)

        monkeypatch.setattr(embeddings, "embed_texts", fake)
        return calls

    return install


@pytest.fixture
def stub_claude(monkeypatch):
    """Factory replacing ``shared.api.call_claude`` with a recording stub: ``install(responses)`` takes a list (FIFO) or ``(user_message, **kwargs) -> str``; returns the recorded calls."""

    def install(responses):
        calls = []
        queue = list(responses) if isinstance(responses, list) else None
        busy = threading.Lock()

        def fake(user_message, system_prompt="", injection="", model=None, max_tokens=None,
                 return_stop_reason=False, stage=None, temperature=None, item_id=None,
                 cache_system=False, json_schema=None):
            calls.append({
                "user_message": user_message,
                "system_prompt": system_prompt,
                "injection": injection,
                "model": model,
                "max_tokens": max_tokens,
                "stage": stage,
                "temperature": temperature,
                "item_id": item_id,
                "cache_system": cache_system,
                "json_schema": json_schema,
            })
            if queue is None:
                # json_schema is passed only by schema-constrained callers.
                extra = {"json_schema": json_schema} if json_schema is not None else {}
                result = responses(
                    user_message,
                    system_prompt=system_prompt,
                    injection=injection,
                    model=model,
                    max_tokens=max_tokens,
                    stage=stage,
                    temperature=temperature,
                    item_id=item_id,
                    **extra,
                )
            else:
                # FIFO queues assume serial calls: a parallel stage (workers > 1
                # with 2+ pending items) interleaves pops and maps responses to
                # the wrong items nondeterministically. Fail loudly instead.
                assert busy.acquire(blocking=False), (
                    "queue-based stub_claude called concurrently — use a callable "
                    "dispatcher for stages that fan out via parallel_map"
                )
                try:
                    assert queue, "stub_claude queue exhausted — more API calls than canned responses"
                    result = queue.pop(0)
                finally:
                    busy.release()
            # Dispatchers/queues may return (text, stop_reason) to exercise
            # truncation guards; plain strings imply a clean end_turn.
            text, stop = result if isinstance(result, tuple) else (result, "end_turn")
            return (text, stop) if return_stop_reason else text

        monkeypatch.setattr(api, "call_claude", fake)
        return calls

    return install


@pytest.fixture
def fake_message():
    """Factory for objects shaped like an Anthropic Message response."""

    def make(text="ok", input_tokens=10, output_tokens=5, stop_reason="end_turn"):
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            content=[SimpleNamespace(text=text, type="text")],
            stop_reason=stop_reason,
        )

    return make


@pytest.fixture
def stub_hf(monkeypatch):
    """Factory replacing ``evals.publish_hf``'s three Hub-API chokepoints with recording stubs; ``install(raise_on_call=True)`` makes every call raise (to assert --dry-run makes none)."""

    def install(raise_on_call: bool = False):
        from evals import publish_hf
        calls = []

        def _blocked(name):
            def fn(*args, **kwargs):
                raise AssertionError(f"{name} called during a supposed --dry-run")
            return fn

        def _create_repo(repo_id):
            calls.append({"fn": "create_repo", "repo_id": repo_id})

        def _upload_folder(folder_path, repo_id, commit_message, delete_patterns):
            calls.append({
                "fn": "upload_folder", "folder_path": folder_path,
                "repo_id": repo_id, "commit_message": commit_message,
                "delete_patterns": delete_patterns,
            })
            return "fake-commit-sha"

        def _create_tag(repo_id, tag):
            calls.append({"fn": "create_tag", "repo_id": repo_id, "tag": tag})

        stubs = {
            "_create_repo": _create_repo, "_upload_folder": _upload_folder,
            "_create_tag": _create_tag,
        }
        for name, impl in stubs.items():
            monkeypatch.setattr(publish_hf, name,
                                _blocked(name) if raise_on_call else impl)
        return calls

    return install


@pytest.fixture(scope="session")
def prompts_sdf():
    return REPO_ROOT / "prompts" / "sdf"


@pytest.fixture(scope="session")
def prompts_dad():
    return REPO_ROOT / "prompts" / "dad"


@pytest.fixture
def tiny_config(tmp_path):
    """config.yaml shape with minimal scale knobs; all paths under tmp_path."""
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 4000,
        "temperature": 1.0,
        "workers": 2,
        "sdf": {
            "n_prompts": 2,
            "seed": 0,
            "entity_pool_seed": 137,
            "min_score_threshold": 7,
            # stub documents are near-identical by construction; the cull has
            # its own unit test with the threshold on
            "near_dup_threshold": None,
        },
        "dad": {
            "dilemmas": {
                "count": 2,
                "batch_size": 2,
                "id_start": 1,
                "seed_path": None,
                # pinned so the sampled scenarios (and the fields the stub
                # echoes back) are identical run to run
                "scenario_seed": 7,
                # both 1c/1d toggles explicit: the composer refuses configs
                # that set exactly one
                "gate": True,
                "refine": True,
            },
            "responses": {"per_prompt": 1},
            # default-on in real configs, but the auto evals are subprocesses —
            # outside pytest-socket's reach — so the test config opts out; the
            # knob itself is covered in test_e2e_smoke.py with subprocess mocked
            "evals": {"auto": False},
        },
        "language_distribution": {"en": 1.0},
        "outputs": {"cost_log": str(tmp_path / "cost_log.jsonl")},
    }


@pytest.fixture
def stage_dir(tmp_path):
    """An empty directory standing in for one run's layer1 stage folder."""
    directory = tmp_path / "layer1"
    directory.mkdir()
    return directory


@pytest.fixture
def tiny_config_file(tiny_config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path




def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run tests marked @pytest.mark.slow (opt-in: they need local model files or are slow)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: opt-in test, skipped unless --run-slow is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: pass --run-slow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
