"""Tests for shared/api.py: the safety net itself, call assembly, cost tracking.

Nothing here (or anywhere in the suite) reaches the network: see conftest.py.
"""

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import pytest
import pytest_socket
import yaml

from shared import api, utils

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestSafetyNet:
    def test_unstubbed_call_claude_is_blocked(self, tiny_config_file, tmp_path):
        api.init(str(tiny_config_file), cost_log_path=tmp_path / "cost.jsonl")
        with pytest.raises(AssertionError, match="API call attempted"):
            api.call_claude("hello")

    def test_unstubbed_claude_code_call_is_blocked(self, tiny_config, tmp_path):
        # The claude_code seam spawns a real CLI subprocess, which pytest-socket
        # can't block — so _api_guard must fail it fast in-process instead.
        tiny_config["backend"] = "claude_code"
        path = tmp_path / "cc.yaml"
        path.write_text(yaml.safe_dump(tiny_config))
        api.init(str(path), cost_log_path=tmp_path / "cost.jsonl")
        with pytest.raises(AssertionError, match="claude_code backend invoked"):
            api.call_claude("hello")

    # pytest-socket warns as well as raises when it blocks name resolution.
    # The warning IS the guard working, so silence exactly it (and nothing
    # else) to keep the suite's warning summary clean.
    @pytest.mark.filterwarnings("ignore:A test tried to use socket.getaddrinfo:UserWarning")
    def test_sockets_are_blocked(self):
        with pytest.raises(pytest_socket.SocketBlockedError):
            socket.create_connection(("127.0.0.1", 9), timeout=0.1)

    def test_init_without_key_fails_loudly(self, tiny_config_file, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        with pytest.raises(KeyError):
            api.init(str(tiny_config_file))


@pytest.fixture
def recorded_api(monkeypatch, tmp_path, fake_message):
    """Exercise call_claude itself: record what reaches the transport layer.

    Replaces _call_with_retry (below call_claude, above the network) and gives
    the module a dummy client + tmp cost log so no init()/anthropic is needed.
    """
    calls = []
    canned = {"message": fake_message(text="response-text")}

    def record(client, model, max_tokens, system, messages, temperature):
        calls.append({"model": model, "max_tokens": max_tokens, "system": system,
                      "messages": messages, "temperature": temperature})
        return canned["message"]

    monkeypatch.setattr(api, "_call_with_retry", record)
    monkeypatch.setattr(api, "_client", object())
    monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
    canned["calls"] = calls
    return canned


class TestCallClaude:
    def test_returns_response_text(self, recorded_api):
        assert api.call_claude("hi") == "response-text"

    def test_injection_appended_to_system_prompt(self, recorded_api):
        api.call_claude("hi", system_prompt="sys", injection="inj")
        assert recorded_api["calls"][0]["system"] == "sys\n\ninj"

    def test_injection_without_system_prompt_stands_alone(self, recorded_api):
        api.call_claude("hi", injection="inj")
        assert recorded_api["calls"][0]["system"] == "inj"

    def test_user_message_sent_as_single_user_turn(self, recorded_api):
        api.call_claude("hi")
        assert recorded_api["calls"][0]["messages"] == [{"role": "user", "content": "hi"}]

    def test_cache_system_wraps_system_in_ephemeral_block(self, recorded_api):
        api.call_claude("hi", system_prompt="big system", cache_system=True)
        sysarg = recorded_api["calls"][0]["system"]
        assert isinstance(sysarg, list)
        assert sysarg[0]["text"] == "big system"
        assert sysarg[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_system_off_sends_plain_string(self, recorded_api):
        api.call_claude("hi", system_prompt="big system")
        assert recorded_api["calls"][0]["system"] == "big system"

    def test_cache_system_ignored_for_empty_system(self, recorded_api):
        api.call_claude("hi", cache_system=True)  # nothing to cache
        assert recorded_api["calls"][0]["system"] == ""

    def test_model_and_max_tokens_fall_back_to_config(self, recorded_api, monkeypatch):
        monkeypatch.setattr(api, "_config", {"model": "cfg-model", "max_tokens": 123})
        api.call_claude("hi")
        assert recorded_api["calls"][0]["model"] == "cfg-model"
        assert recorded_api["calls"][0]["max_tokens"] == 123

    def test_explicit_model_and_max_tokens_win(self, recorded_api, monkeypatch):
        monkeypatch.setattr(api, "_config", {"model": "cfg-model", "max_tokens": 123})
        api.call_claude("hi", model="override", max_tokens=9)
        assert recorded_api["calls"][0]["model"] == "override"
        assert recorded_api["calls"][0]["max_tokens"] == 9

    def test_temperature_falls_back_to_config(self, recorded_api, monkeypatch):
        monkeypatch.setattr(api, "_config", {"temperature": 0.3})
        api.call_claude("hi")
        assert recorded_api["calls"][0]["temperature"] == 0.3

    def test_temperature_defaults_to_one_without_config(self, recorded_api, monkeypatch):
        monkeypatch.setattr(api, "_config", {})
        api.call_claude("hi")
        assert recorded_api["calls"][0]["temperature"] == 1.0

    def test_explicit_temperature_wins_even_at_zero(self, recorded_api, monkeypatch):
        # 0.0 is falsy — the override must use an is-None check, not `or`
        monkeypatch.setattr(api, "_config", {"temperature": 0.3})
        api.call_claude("hi", temperature=0.0)
        assert recorded_api["calls"][0]["temperature"] == 0.0

    def test_return_stop_reason_tuple_contract(self, recorded_api, fake_message):
        recorded_api["message"] = fake_message(text="partial", stop_reason="max_tokens")
        text, stop = api.call_claude("hi", return_stop_reason=True)
        assert (text, stop) == ("partial", "max_tokens")
        # default return stays a bare string for backward compatibility
        assert api.call_claude("hi") == "partial"

    def test_suspect_stop_reason_warns(self, recorded_api, fake_message, capsys):
        recorded_api["message"] = fake_message(text="cut", stop_reason="max_tokens")
        api.call_claude("hi")
        assert "max_tokens" in capsys.readouterr().err

    def test_clean_stop_reason_is_silent(self, recorded_api, capsys):
        api.call_claude("hi")
        assert capsys.readouterr().err == ""


class TestSamplingParams:
    """temperature/top_p/top_k are removed on the Claude 5 family and Opus 4.7+
    (the API 400s if sent). call_claude resolves temperature unconditionally;
    _call_with_retry gates whether it reaches the transport."""

    def test_predicate_matches_current_model_families(self):
        for m in ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                  "claude-sonnet-5", "claude-fable-5", "claude-mythos-5"):
            assert not api._accepts_sampling_params(m), m
        for m in ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6",
                  "claude-opus-4-5", "claude-sonnet-4-5"):
            assert api._accepts_sampling_params(m), m

    def _capture(self, monkeypatch, model):
        """Drive the REAL _call_with_retry with a fake client; return the kwargs
        that reached client.messages.create."""
        seen = {}

        class FakeMessages:
            def create(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                    content=[SimpleNamespace(text="ok", type="text")],
                    stop_reason="end_turn")

        class FakeClient:
            messages = FakeMessages()

        import importlib
        real = importlib.reload(api)._call_with_retry
        real(client=FakeClient(), model=model, max_tokens=5, system="", messages=[],
             temperature=0.0)
        return seen

    def test_temperature_omitted_for_rejecting_model(self, monkeypatch):
        seen = self._capture(monkeypatch, "claude-opus-4-8")
        assert "temperature" not in seen

    def test_temperature_sent_for_accepting_model(self, monkeypatch):
        seen = self._capture(monkeypatch, "claude-haiku-4-5")
        assert seen["temperature"] == 0.0

    def test_nondefault_temperature_on_rejecting_model_warns_once(
        self, recorded_api, monkeypatch, capsys
    ):
        monkeypatch.setattr(api, "_config", {"model": "claude-opus-4-8", "temperature": 1.0})
        api.call_claude("hi", temperature=0.0)
        assert "does not accept sampling parameters" in capsys.readouterr().err
        api.call_claude("hi", temperature=0.0)  # warned-once
        assert capsys.readouterr().err == ""


class TestRetryPredicate:
    def test_bad_request_is_not_retried(self, monkeypatch):
        """A 400 (e.g. exhausted credit balance) is deterministic: it must
        surface immediately, not after 8 exponential-backoff attempts. This
        drives the REAL tenacity-wrapped _call_with_retry — safe offline
        because a non-retryable error never sleeps."""
        attempts = []

        class FakeMessages:
            def create(self, **kwargs):
                attempts.append(1)
                raise anthropic.BadRequestError(
                    message="credit balance is too low",
                    response=httpx.Response(400, request=httpx.Request("POST", "https://api.invalid")),
                    body=None,
                )

        class FakeClient:
            messages = FakeMessages()

        # the autouse guard replaces _call_with_retry; restore the real one
        import importlib
        real = importlib.reload(api)._call_with_retry
        with pytest.raises(anthropic.BadRequestError):
            real(client=FakeClient(), model="m", max_tokens=5, system="", messages=[],
                 temperature=1.0)
        assert len(attempts) == 1  # exactly one attempt, no retries


class TestCostTracking:
    def test_cost_logged_with_known_model_pricing(self, recorded_api, tmp_path, fake_message):
        recorded_api["message"] = fake_message(input_tokens=1_000_000, output_tokens=0)
        api.call_claude("hi", model="claude-haiku-4-5-20251001")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["cost_usd"] == pytest.approx(1.00)  # $1.00 / 1M input tokens
        assert record["model"] == "claude-haiku-4-5-20251001"
        assert record["input_tokens"] == 1_000_000

    def test_cache_read_tokens_priced_at_ten_percent(self, recorded_api, tmp_path, fake_message):
        msg = fake_message(input_tokens=0, output_tokens=0)
        msg.usage.cache_read_input_tokens = 1_000_000
        recorded_api["message"] = msg
        api.call_claude("hi", model="claude-sonnet-5", cache_system=True)
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["cache_read_tokens"] == 1_000_000
        assert record["cost_usd"] == pytest.approx(0.30)  # 1M * $3/M * 0.10

    def test_cache_write_tokens_priced_at_125_percent(self, recorded_api, tmp_path, fake_message):
        msg = fake_message(input_tokens=0, output_tokens=0)
        msg.usage.cache_creation_input_tokens = 1_000_000
        recorded_api["message"] = msg
        api.call_claude("hi", model="claude-sonnet-5", cache_system=True)
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["cache_creation_tokens"] == 1_000_000
        assert record["cost_usd"] == pytest.approx(3.75)  # 1M * $3/M * 1.25

    def test_config_model_is_priced_without_warning(self, recorded_api, tmp_path, fake_message, capsys):
        # Contract from shared/api.py: _PRICING must cover every model id that
        # can appear in config.yaml. A warning here means the tables drifted.
        config_model = utils.load_config(str(REPO_ROOT / "config.yaml"))["model"]
        recorded_api["message"] = fake_message(input_tokens=1_000_000, output_tokens=0)
        api.call_claude("hi", model=config_model)
        assert "not in shared/api.py _PRICING" not in capsys.readouterr().err

    def test_every_configured_model_knob_is_priced(self):
        # Same contract as above, extended to the per-stage overrides
        # (sdf.rewrite_model, dad.constitution_rewrite_model, …). A stage knob
        # naming an unpriced model bills the whole stage at Sonnet rates behind
        # a single warning — on a 500-doc run with an Opus rewrite that
        # understates the run by more than half.
        config = utils.load_config(str(REPO_ROOT / "config.yaml"))

        def model_values(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "model" or key.endswith("_model"):
                        if isinstance(value, str) and value:
                            yield key, value
                    else:
                        yield from model_values(value)

        seen = list(model_values(config))
        assert seen, "config.yaml declares no model knobs — did the schema change?"
        for key, model in seen:
            assert model in api._PRICING, f"config {key}={model} is not in _PRICING"

    def test_unknown_model_falls_back_to_sonnet_rates_with_warning(self, recorded_api, tmp_path, fake_message, capsys):
        recorded_api["message"] = fake_message(input_tokens=1_000_000, output_tokens=0)
        api.call_claude("hi", model="claude-nonexistent-model")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["cost_usd"] == pytest.approx(3.00)
        assert "not in shared/api.py _PRICING" in capsys.readouterr().err

    def test_unpriced_model_warns_only_once(self, recorded_api, fake_message, capsys):
        recorded_api["message"] = fake_message(input_tokens=1, output_tokens=1)
        api.call_claude("a", model="claude-nonexistent-model")
        api.call_claude("b", model="claude-nonexistent-model")
        assert capsys.readouterr().err.count("not in shared/api.py _PRICING") == 1

    def test_stage_written_to_cost_record(self, recorded_api, tmp_path):
        api.call_claude("hi", stage="prompt_draft")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["stage"] == "prompt_draft"

    def test_untagged_call_writes_no_stage_key(self, recorded_api, tmp_path):
        # pre-tag consumers (and jq one-liners) must not meet a null field
        api.call_claude("hi")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert "stage" not in record

    def test_item_id_written_to_cost_record(self, recorded_api, tmp_path):
        api.call_claude("hi", stage="response_scope", item_id="AW-0001")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["item_id"] == "AW-0001"

    def test_call_without_item_id_writes_no_item_key(self, recorded_api, tmp_path):
        api.call_claude("hi")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert "item_id" not in record

    def test_duration_and_attempts_written_to_cost_record(self, recorded_api, tmp_path):
        # _call_with_retry is stubbed here, so no tenacity attempt fires and
        # the attempt counter keeps its per-call reset value of 1
        api.call_claude("hi")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["attempts"] == 1
        assert isinstance(record["duration_s"], float) and record["duration_s"] >= 0

    def test_attempt_counter_resets_between_calls(self, recorded_api, tmp_path):
        # a retried call must not leak its attempt count into the next call
        # on the same thread
        api._attempt_state.n = 5  # as if a previous call retried 4 times
        api.call_claude("hi")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["attempts"] == 1

    def test_get_total_cost_sums_all_calls(self, recorded_api, fake_message):
        recorded_api["message"] = fake_message(input_tokens=1_000_000, output_tokens=1_000_000)
        api.call_claude("a", model="claude-sonnet-4-6")  # $3 + $15
        api.call_claude("b", model="claude-sonnet-4-6")
        assert api.get_total_cost() == pytest.approx(36.0)

    def test_get_total_cost_without_log_is_zero(self):
        # autouse fixture resets _cost_log_path to None
        assert api.get_total_cost() == 0.0

    def test_get_total_cost_missing_file_is_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "never-written.jsonl")
        assert api.get_total_cost() == 0.0


class TestBackendSelection:
    def test_init_defaults_to_api_backend(self, tiny_config_file):
        api.init(str(tiny_config_file))  # tiny_config has no `backend` key
        assert api._backend == "api"

    def test_init_reads_claude_code_backend(self, tiny_config, tmp_path):
        tiny_config["backend"] = "claude_code"
        path = tmp_path / "cc.yaml"
        path.write_text(yaml.safe_dump(tiny_config))
        api.init(str(path))
        assert api._backend == "claude_code"

    def test_init_rejects_unknown_backend(self, tiny_config, tmp_path):
        tiny_config["backend"] = "bogus"
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(tiny_config))
        with pytest.raises(ValueError, match="backend"):
            api.init(str(path))

    def test_claude_code_backend_needs_no_key(self, tiny_config, tmp_path, monkeypatch):
        # The whole point of claude_code: it authenticates via the CLI, not a key.
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        tiny_config["backend"] = "claude_code"
        path = tmp_path / "cc.yaml"
        path.write_text(yaml.safe_dump(tiny_config))
        api.init(str(path))  # must NOT raise, unlike the api backend
        assert api._backend == "claude_code"


class TestClassifyClaudeCodeError:
    # Window exhaustion must abort the run (non-retryable); a transient CLI
    # hiccup must fall through to the retried ClaudeCodeError path.
    @pytest.mark.parametrize("message", [
        "Claude AI usage limit reached|1751400000",
        "You have reached your usage limit",
        # What the CLI reports instead of the window message when the org has
        # usage-billing overflow disabled — must pause-and-resume, not retry.
        "You've hit your org's monthly spend limit \u00b7 run /usage-credits to ask your admin for a higher limit",
    ])
    def test_usage_limit_is_non_retryable(self, message):
        err = api._classify_claude_code_error(message)
        assert isinstance(err, api.UsageLimitExceeded)
        assert not isinstance(err, api.ClaudeCodeError)  # tenacity won't retry it
        assert "--resume" in str(err)

    @pytest.mark.parametrize("message", [
        "rate limit exceeded, retry shortly",
        "429 rate limit reached",  # 'limit reached' must NOT be caught as window exhaustion
        "spawn ENOENT",
        "unknown claude_code error",
    ])
    def test_transient_error_is_retryable(self, message):
        assert isinstance(api._classify_claude_code_error(message), api.ClaudeCodeError)

    def test_content_policy_refusal_is_a_refusal_not_an_error(self):
        # The AW-0027-era wall: a per-item policy refusal must be neither
        # retried (same content, same verdict) nor treated as backend death.
        message = ("API Error: Claude Code is unable to respond to this request, "
                   "which appears to violate our Usage Policy "
                   "(https://www.anthropic.com/legal/aup). Try rephrasing.")
        err = api._classify_claude_code_error(message)
        assert isinstance(err, api.ClaudeCodeRefusal)
        assert not isinstance(err, api.ClaudeCodeError)      # tenacity won't retry it
        assert not isinstance(err, api.UsageLimitExceeded)   # and it isn't the window


class TestClaudeCodeRefusalSurface:
    def _init_cc(self, tiny_config, tmp_path, backend):
        import yaml
        tiny_config["backend"] = backend
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump(tiny_config))
        api.init(str(cfg), cost_log_path=tmp_path / "cost.jsonl")

    def test_refusal_surfaces_as_refusal_stop_reason(self, tiny_config, tmp_path, monkeypatch):
        self._init_cc(tiny_config, tmp_path, "claude_code")

        def refuse(**kw):
            raise api.ClaudeCodeRefusal("violates our Usage Policy")

        monkeypatch.setattr(api, "_call_claude_code_with_retry", refuse)
        text, stop = api.call_claude(user_message="u", system_prompt="s",
                                     max_tokens=100, return_stop_reason=True)
        assert (text, stop) == ("", "refusal")

    def test_refusal_never_demotes_the_auto_backend(self, tiny_config, tmp_path, monkeypatch):
        self._init_cc(tiny_config, tmp_path, "auto")

        def refuse(**kw):
            raise api.ClaudeCodeRefusal("violates our Usage Policy")

        monkeypatch.setattr(api, "_call_claude_code_with_retry", refuse)
        text, stop = api.call_claude(user_message="u", system_prompt="s",
                                     max_tokens=100, return_stop_reason=True)
        assert (text, stop) == ("", "refusal")
        assert api._cc_demoted is None  # per-item verdict, not backend health


class TestResolveClaudeCodeSystem:
    def test_nonempty_system_passes_through_silently(self, capsys):
        assert api._resolve_cc_system("real system") == "real system"
        assert capsys.readouterr().err == ""

    def test_empty_system_gets_neutral_stand_in(self, capsys):
        assert api._resolve_cc_system("") == api._NEUTRAL_SYSTEM

    def test_empty_system_warns_only_once(self, capsys):
        api._resolve_cc_system("")
        api._resolve_cc_system("")
        assert capsys.readouterr().err.count("neutral system prompt") == 1


@pytest.mark.enable_socket
class TestRunClaudeCodeQuery:
    """Exercise the claude_code money path (`_run_claude_code_query`) at the true
    external boundary — a stubbed `claude_agent_sdk.query` yielding real SDK
    message objects — so a wrong attribute name or an SDK schema change is caught,
    unlike tests that stub the wrapper away. `_run_claude_code_query` is the
    un-retried inner, so error cases raise immediately (no tenacity backoff).

    `enable_socket`: `query` is stubbed, so no network I/O occurs — the exemption
    is only for the localhost socketpair asyncio's event loop (via `anyio.run`)
    uses for its internal self-pipe, which `--disable-socket` would otherwise block.
    """

    @staticmethod
    def _install_query(monkeypatch, messages):
        pytest.importorskip("claude_agent_sdk")

        async def fake_query(*, prompt, options):
            for m in messages:
                yield m

        monkeypatch.setattr("claude_agent_sdk.query", fake_query)

    @staticmethod
    def _result(**kw):
        types = pytest.importorskip("claude_agent_sdk.types")
        fields = dict(
            subtype="success", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="sesn_test",
        )
        fields.update(kw)
        return types.ResultMessage(**fields)

    @staticmethod
    def _assistant(text):
        types = pytest.importorskip("claude_agent_sdk.types")
        return types.AssistantMessage(
            content=[types.TextBlock(text=text)], model="claude-haiku-4-5"
        )

    def test_happy_path_parses_text_tokens_cost(self, monkeypatch):
        self._install_query(monkeypatch, [
            self._assistant("draft answer"),
            self._result(result="draft answer",
                         usage={"input_tokens": 120, "output_tokens": 34},
                         total_cost_usd=0.0021, stop_reason="end_turn"),
        ])
        text, in_tok, out_tok, cost, stop = api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")
        assert text == "draft answer"
        assert (in_tok, out_tok) == (120, 34)
        assert cost == pytest.approx(0.0021)
        assert stop == "end_turn"

    def test_result_none_falls_back_to_joined_assistant_text(self, monkeypatch):
        self._install_query(monkeypatch, [
            self._assistant("part1 "),
            self._assistant("part2"),
            self._result(result=None, usage={"input_tokens": 5, "output_tokens": 1}),
        ])
        text, in_tok, out_tok, cost, stop = api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")
        assert text == "part1 part2"
        assert (in_tok, out_tok, cost) == (5, 1, None)

    def test_missing_usage_defaults_tokens_to_zero(self, monkeypatch):
        self._install_query(monkeypatch, [
            self._result(result="x", usage=None, total_cost_usd=None),
        ])
        assert api._run_claude_code_query("claude-haiku-4-5", "sys", "hi") == ("x", 0, 0, None, None)

    def test_is_error_usage_limit_raises_non_retryable(self, monkeypatch):
        self._install_query(monkeypatch, [
            self._result(is_error=True, result="Claude AI usage limit reached|1751400000"),
        ])
        with pytest.raises(api.UsageLimitExceeded):
            api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")

    def test_is_error_transient_raises_retryable(self, monkeypatch):
        self._install_query(monkeypatch, [
            self._result(is_error=True, result="temporary backend failure, try again"),
        ])
        with pytest.raises(api.ClaudeCodeError):
            api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")

    def test_no_result_message_raises(self, monkeypatch):
        self._install_query(monkeypatch, [self._assistant("orphan text")])
        with pytest.raises(api.ClaudeCodeError, match="no result message"):
            api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")

    def test_stops_reading_at_result_so_trailing_stream_error_cannot_mask_it(self, monkeypatch):
        """After an is_error result the real CLI exits non-zero; if the reader
        keeps consuming the stream, that exit surfaces as an opaque ProcessError
        (the SDK reports only the result subtype, e.g. "error result: success")
        which masks result_msg.result — the CLI's actual error — and bypasses
        usage-limit classification. The reader must stop at the ResultMessage."""
        pytest.importorskip("claude_agent_sdk")
        result = self._result(is_error=True, result="API Error: 500 upstream failure")

        async def fake_query(*, prompt, options):
            yield result
            raise AssertionError("stream read past the result message")

        monkeypatch.setattr("claude_agent_sdk.query", fake_query)
        with pytest.raises(api.ClaudeCodeError, match="500 upstream failure"):
            api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")

    def test_normal_system_prompt_stays_inline(self, monkeypatch):
        pytest.importorskip("claude_agent_sdk")
        seen = {}

        async def fake_query(*, prompt, options):
            seen["system_prompt"] = options.system_prompt
            yield self._result(result="ok", usage={"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr("claude_agent_sdk.query", fake_query)
        api._run_claude_code_query("claude-haiku-4-5", "sys", "hi")
        assert seen["system_prompt"] == "sys"

    def test_oversized_system_prompt_travels_by_file_not_argv(self, monkeypatch):
        """Linux caps a single argv string at 128 KiB (MAX_ARG_STRLEN), and the
        SDF layer-4/5 constitution system prompt (~185 KB) exceeds it — spawning
        the CLI fails with E2BIG unless the prompt is handed over as
        --system-prompt-file. The temp file must also not outlive the call."""
        pytest.importorskip("claude_agent_sdk")
        big_system = "x" * (api._CC_SYSTEM_ARG_MAX_BYTES + 1)
        seen = {}

        async def fake_query(*, prompt, options):
            seen["system_prompt"] = options.system_prompt
            if isinstance(options.system_prompt, dict):
                # Read while the call is in flight — the file is deleted after.
                seen["content"] = Path(options.system_prompt["path"]).read_text()
            yield self._result(result="ok", usage={"input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr("claude_agent_sdk.query", fake_query)
        text, *_ = api._run_claude_code_query("claude-haiku-4-5", big_system, "hi")
        assert text == "ok"
        assert isinstance(seen["system_prompt"], dict)
        assert seen["system_prompt"]["type"] == "file"
        assert seen["content"] == big_system
        assert not Path(seen["system_prompt"]["path"]).exists()

    def test_oversized_system_temp_file_cleaned_up_on_error(self, monkeypatch):
        pytest.importorskip("claude_agent_sdk")
        big_system = "x" * (api._CC_SYSTEM_ARG_MAX_BYTES + 1)
        seen = {}

        async def fake_query(*, prompt, options):
            seen["path"] = options.system_prompt["path"]
            raise RuntimeError("spawn failed")
            yield  # pragma: no cover — makes this an async generator

        monkeypatch.setattr("claude_agent_sdk.query", fake_query)
        with pytest.raises(api.ClaudeCodeError, match="spawn failed"):
            api._run_claude_code_query("claude-haiku-4-5", big_system, "hi")
        assert not Path(seen["path"]).exists()


class TestClaudeCodeDispatch:
    """call_claude routes to the claude_code path and logs its reported cost.

    _call_claude_code_with_retry is stubbed so no CLI subprocess is spawned;
    the api seam (_call_with_retry) stays the _blocked raiser from _api_guard,
    so this also proves the claude_code path never touches the API client.
    """

    def test_dispatches_and_logs_notional_cost(self, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        seen = {}

        def fake_cc(model, system, user_message):
            seen.update(model=model, system=system, user_message=user_message)
            return ("cc-response", 111, 22, 0.004, "end_turn")

        monkeypatch.setattr(api, "_call_claude_code_with_retry", fake_cc)

        out = api.call_claude("hi", system_prompt="sys", injection="inj", model="claude-haiku-4-5")

        assert out == "cc-response"
        # system prompt + injection assembled the same way as the api path
        assert seen == {"model": "claude-haiku-4-5", "system": "sys\n\ninj", "user_message": "hi"}
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["backend"] == "claude_code"
        assert record["cost_usd"] == pytest.approx(0.004)  # Claude Code's own cost, logged verbatim
        assert record["input_tokens"] == 111 and record["output_tokens"] == 22

    def test_stage_tag_logged_on_claude_code(self, monkeypatch, tmp_path):
        # The per-stage cost breakdown must not be lost on the claude_code path.
        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_call_claude_code_with_retry",
                            lambda model, system, user_message: ("t", 1, 1, 0.001, "end_turn"))
        api.call_claude("hi", stage="layer3_rewrite")
        record = json.loads((tmp_path / "cost.jsonl").read_text().strip())
        assert record["stage"] == "layer3_rewrite"

    def test_non_default_temperature_warns_once_on_claude_code(self, monkeypatch, tmp_path, capsys):
        # The CLI exposes no temperature control; a deliberate override must be
        # surfaced (once), and the default 1.0 must stay silent.
        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_call_claude_code_with_retry",
                            lambda model, system, user_message: ("t", 1, 1, None, "end_turn"))
        api.call_claude("hi")  # config default 1.0 — no warning
        assert "temperature" not in capsys.readouterr().err
        api.call_claude("hi", temperature=0.3)
        api.call_claude("hi", temperature=0.3)
        assert capsys.readouterr().err.count("cannot set sampling temperature") == 1

    def test_return_stop_reason_honored_on_claude_code(self, monkeypatch, tmp_path):
        # regression: the claude_code branch must honor return_stop_reason like
        # the api path, or callers doing `text, stop = call_claude(...)` mis-unpack.
        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_call_claude_code_with_retry",
                            lambda model, system, user_message: ("t", 1, 1, None, "max_tokens"))
        assert api.call_claude("hi", return_stop_reason=True) == ("t", "max_tokens")
        assert api.call_claude("hi") == "t"  # bare string without the flag

    def test_suspect_stop_reason_warns_on_claude_code(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_call_claude_code_with_retry",
                            lambda model, system, user_message: ("cut", 1, 1, None, "max_tokens"))
        api.call_claude("hi")
        assert "max_tokens" in capsys.readouterr().err


class TestAdaptiveThinkingModels:
    class _FakeMessages:
        def __init__(self):
            self.kwargs = None
        def create(self, **kwargs):
            self.kwargs = kwargs
            return "response"

    class _FakeClient:
        def __init__(self):
            self.messages = TestAdaptiveThinkingModels._FakeMessages()

    def test_thinking_disabled_for_standard_models(self):
        # the autouse guard replaces _call_with_retry; restore the real one
        # (same pattern as TestRetryPredicate)
        import importlib
        real = importlib.reload(api)._call_with_retry
        client = self._FakeClient()
        real(client=client, model="claude-sonnet-5", max_tokens=100,
             system="s", messages=[], temperature=1.0)
        assert client.messages.kwargs["thinking"] == {"type": "disabled"}

    def test_thinking_flag_omitted_for_mythos_class(self):
        # fable/mythos 400 on thinking-disabled; the flag is omitted and
        # _response_text strips the thinking blocks downstream
        import importlib
        real = importlib.reload(api)._call_with_retry
        client = self._FakeClient()
        real(client=client, model="claude-fable-5", max_tokens=100,
             system="s", messages=[], temperature=1.0)
        assert "thinking" not in client.messages.kwargs

    def test_predicate_covers_the_tier(self):
        assert api._requires_adaptive_thinking("claude-fable-5")
        assert api._requires_adaptive_thinking("claude-mythos-5")
        assert not api._requires_adaptive_thinking("claude-sonnet-5")
        assert not api._requires_adaptive_thinking("claude-haiku-4-5-20251001")


class TestAutoBackend:
    """backend 'auto': subscription-first per-call routing with api fallback.

    The cc seam is stubbed per test; recorded_api provides the api leg. Routing
    contract: system-prompt calls prefer claude_code, empty-system calls (the
    DAD baseline arm) always take the api leg, and any subscription failure
    demotes the rest of the run to the api — re-serving the failed call."""

    def _arm(self, monkeypatch, tmp_path, cc=None):
        monkeypatch.setattr(api, "_backend", "auto")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_sdk_importable", lambda: True)
        if cc is not None:
            monkeypatch.setattr(api, "_call_claude_code_with_retry", cc)

    def test_routes_by_system_prompt(self, recorded_api, monkeypatch, tmp_path):
        cc_calls = []

        def fake_cc(model, system, user_message):
            cc_calls.append(system)
            return ("cc-text", 1, 1, 0.001, "end_turn")

        self._arm(monkeypatch, tmp_path, fake_cc)
        # a system-prompt call takes the subscription leg...
        assert api.call_claude("hi", system_prompt="sys") == "cc-text"
        # ...an empty-system call (the baseline arm) takes the api leg exactly
        assert api.call_claude("hi") == "response-text"
        assert cc_calls == ["sys"]
        assert len(recorded_api["calls"]) == 1
        # each cost-log record names the backend that actually served it
        records = [json.loads(l) for l in (tmp_path / "cost.jsonl").read_text().splitlines()]
        assert [r["backend"] for r in records] == ["claude_code", "api"]

    def test_window_exhaustion_falls_back_and_stays_demoted(
        self, recorded_api, monkeypatch, tmp_path, capsys
    ):
        cc_calls = {"n": 0}

        def limited_cc(model, system, user_message):
            cc_calls["n"] += 1
            raise api.UsageLimitExceeded("Claude AI usage limit reached|1234567890")

        self._arm(monkeypatch, tmp_path, limited_cc)
        # the failed call is re-served via the api, not lost
        assert api.call_claude("hi", system_prompt="sys") == "response-text"
        # demotion is sticky: later calls skip the subscription entirely
        assert api.call_claude("again", system_prompt="sys") == "response-text"
        assert cc_calls["n"] == 1
        assert "Falling back to the Anthropic API" in capsys.readouterr().err
        records = [json.loads(l) for l in (tmp_path / "cost.jsonl").read_text().splitlines()]
        assert [r["backend"] for r in records] == ["api", "api"]

    def test_missing_sdk_serves_run_via_api(self, recorded_api, monkeypatch, tmp_path, capsys):
        # cc seam stays the _api_guard raiser — it must never be reached
        monkeypatch.setattr(api, "_backend", "auto")
        monkeypatch.setattr(api, "_sdk_importable", lambda: False)
        assert api.call_claude("hi", system_prompt="sys") == "response-text"
        assert "claude-agent-sdk is not installed" in capsys.readouterr().err

    def test_auto_requires_api_key(self, monkeypatch, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(f"backend: auto\noutputs:\n  cost_log: {tmp_path / 'log.jsonl'}\n")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(KeyError, match="ANTHROPIC_API_KEY"):
            api.init(str(cfg))

    def test_pure_claude_code_backend_still_propagates_limit(self, monkeypatch, tmp_path):
        # the fallback is an auto-mode behavior; explicit claude_code keeps the
        # stop-and---resume contract (checkpointed progress, window resets later)
        def limited_cc(model, system, user_message):
            raise api.UsageLimitExceeded("usage limit reached")

        monkeypatch.setattr(api, "_backend", "claude_code")
        monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
        monkeypatch.setattr(api, "_call_claude_code_with_retry", limited_cc)
        with pytest.raises(api.UsageLimitExceeded):
            api.call_claude("hi", system_prompt="sys")
