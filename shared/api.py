"""Anthropic API wrapper with retry logic and cost tracking.

Three backends behind the same call_claude() contract:
- "api" (default): the anthropic SDK, billed to the shared ANTHROPIC_API_KEY.
- "claude_code": the Claude Agent SDK driving the Claude Code CLI, billed to
  the contributor's own Claude subscription (Claude Code login, or a
  CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`).
- "auto": prefer the subscription, fall back to the api key. Per-call routing:
  empty-system calls (the DAD baseline arm) always go to the api so the
  plain-model condition stays exact; everything else goes to claude_code until
  it can't serve the run (sdk/CLI missing, usage window exhausted, or a
  persistently failing CLI), at which point the rest of the run is served by
  the api — announced loudly, and each cost-log record names the backend that
  actually served it. Requires ANTHROPIC_API_KEY (the fallback leg).

Select via the `backend` key in config.yaml. See README "Authentication".
"""

import contextlib
import os
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, UTC

import anthropic
import yaml
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

_config: dict = {}
_client: anthropic.Anthropic | None = None
_cost_log_path: Path | None = None
_backend: str = "api"
# call_claude may run from worker threads (utils.parallel_map); the Anthropic
# client is thread-safe, but appends to the cost log must be serialized.
_cost_log_lock = threading.Lock()

# Pricing per million tokens (input, output) for known models
# Prices per million tokens (input, output). Keys must cover every model id
# (or alias) that can appear in config.yaml — unknown models fall back to
# Sonnet rates WITH A WARNING, which can badly misstate real spend (a Haiku
# run was overreported 3x this way). The Anthropic console is the source of
# truth for billing; this log is for per-stage breakdowns.
_PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
_UNPRICED_WARNED: set = set()

_BACKENDS = ("api", "claude_code", "auto")

# auto backend: why the subscription path is currently out of play (None =
# still in play). Set once per run, loudly — after a demotion every remaining
# call is served by the api backend.
_cc_demoted: str | None = None

# Matches subscription-limit exhaustion, which must abort rather than retry.
# Two message families qualify: "Claude AI usage limit reached|<reset-timestamp>"
# (the 5-hour window), and "You've hit your org's monthly spend limit" — which
# orgs that DISABLE usage-billing overflow receive in place of the window
# message, so it too usually means the window resets in a few hours (observed
# live on an overnight run, where it burned 8 tenacity retries per call and
# killed the run instead of pausing for --resume). Deliberately
# narrow otherwise: a transient CLI "rate limit" hiccup should fall through to
# the retried ClaudeCodeError path, so we don't match bare "rate limit" /
# "limit reached" here.
_LIMIT_PATTERN = re.compile(r"usage limit|spend limit", re.IGNORECASE)
# The CLI's content-policy refusal (observed: "Claude Code is unable to respond
# to this request, which appears to violate our Usage Policy"). Distinct from
# the window-limit class: per-item, non-retryable, never a backend demotion.
_REFUSAL_PATTERN = re.compile(r"usage policy|unable to respond to this request",
                              re.IGNORECASE)

# Claude Code treats an empty --system-prompt as unset and substitutes its own
# agentic CLI prompt, which leaks tool/codebase behavior into generated text.
# Stages that send no system prompt get this neutral stand-in instead. Note this
# means a genuinely empty system prompt is not reproducible on this backend — the
# DAD pipeline's response steps (which send no system prompt) are therefore not
# reproduced exactly here; use backend: api for runs where that matters.
_NEUTRAL_SYSTEM = "You are Claude, a helpful AI assistant. Respond directly to the user's message."
_neutral_system_warned = False
_temperature_warned = False

# Linux caps each argv string at 128 KiB (MAX_ARG_STRLEN), and a str
# system_prompt reaches the CLI as a single --system-prompt argument — so the
# ~185 KB constitution (SDF layers 3-5) aborts the spawn with E2BIG on Linux.
# System prompts over this many UTF-8 bytes travel via --system-prompt-file.
_CC_SYSTEM_ARG_MAX_BYTES = 100_000


class UsageLimitExceeded(Exception):
    """Claude subscription usage window exhausted (claude_code backend).

    Not retried: the 5-hour window can take hours to reset. Checkpoints are
    written after every call, so the run can continue later with --resume.
    """


class ClaudeCodeError(Exception):
    """Transient claude_code backend failure; retried by tenacity."""


class ClaudeCodeRefusal(Exception):
    """The subscription path refused this request on content/policy grounds.

    Not transient — retrying re-serves the same content — and not the
    backend's health, so it must never demote an auto run. call_claude
    converts it to the api leg's refusal shape (empty text,
    stop_reason='refusal') so the callers' per-item rejection machinery
    handles it like any other refusal."""


def init(config_path: str = "config.yaml", cost_log_path: str | Path | None = None) -> None:
    global _config, _client, _cost_log_path, _backend, _cc_demoted
    with open(config_path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    _backend = _config.get("backend", "api")
    _cc_demoted = None
    if _backend not in _BACKENDS:
        raise ValueError(f"config backend must be one of {_BACKENDS}, got {_backend!r}")
    # The api backend needs the key; fail loudly here rather than deep in a run.
    # The claude_code backend authenticates via the Claude Code CLI, so no key.
    # auto needs the key too: it is the fallback leg AND serves the empty-system
    # calls (the DAD baseline arm) that claude_code cannot reproduce exactly.
    if _backend in ("api", "auto") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise KeyError("ANTHROPIC_API_KEY")
    _client = None  # constructed lazily; the claude_code backend needs no API key
    _cost_log_path = Path(cost_log_path or _config["outputs"]["cost_log"])
    _cost_log_path.parent.mkdir(parents=True, exist_ok=True)


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "backend 'api' requires ANTHROPIC_API_KEY in .env; "
                "set it, or switch config.yaml to backend: claude_code"
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _log_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
    stage: str | None = None,
    item_id: str | None = None,
    duration_s: float | None = None,
    attempts: int | None = None,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    backend: str | None = None,
) -> None:
    # claude_code passes Claude Code's own reported cost; the api backend leaves
    # cost_usd=None, so we price it from _PRICING with a loud fallback on unknown
    # models (a mispriced run shouldn't hide in the log).
    if cost_usd is None:
        prices = _PRICING.get(model)
        if prices is None:
            if model not in _UNPRICED_WARNED:
                _UNPRICED_WARNED.add(model)
                print(
                    f"  WARNING: model {model!r} is not in shared/api.py _PRICING — "
                    "estimating cost at Sonnet rates ($3/$15 per MTok). Add the model "
                    "to _PRICING for accurate cost logs.",
                    file=sys.stderr,
                )
            prices = (3.00, 15.00)
        # Prompt-caching prices (Anthropic): a cache WRITE is 1.25x the input
        # rate, a cache READ is 0.1x. Plain (uncached) calls pass 0 for both.
        cost_usd = (
            (input_tokens / 1_000_000) * prices[0]
            + (output_tokens / 1_000_000) * prices[1]
            + (cache_creation_tokens / 1_000_000) * prices[0] * 1.25
            + (cache_read_tokens / 1_000_000) * prices[0] * 0.10
        )
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        # For claude_code, cost_usd is notional (what the call would have cost
        # at API prices) — actual billing is the contributor's subscription.
        # The EFFECTIVE backend that served the call (auto runs log which leg
        # each call actually took); falls back to the configured backend.
        "backend": backend or _backend,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
    }
    if cache_creation_tokens:
        record["cache_creation_tokens"] = cache_creation_tokens
    if cache_read_tokens:
        record["cache_read_tokens"] = cache_read_tokens
    if stage:
        record["stage"] = stage
    if item_id:
        record["item_id"] = item_id
    if duration_s is not None:
        record["duration_s"] = round(duration_s, 2)
    if attempts is not None:
        record["attempts"] = attempts
    with _cost_log_lock, open(_cost_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# Only transient failures are worth retrying. APIStatusError is the base for
# 4xx too, and retrying a non-retryable 4xx (bad request, auth, not-found) just
# burns 8 exponential-backoff attempts before surfacing the real error — so
# retry only rate limits, 5xx, and connection/timeout (APITimeoutError
# subclasses APIConnectionError).
_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)

# Attempt counter for the cost log. Thread-local because call_claude runs from
# parallel_map worker threads; tenacity's own .statistics is shared across
# threads and would misattribute counts.
_attempt_state = threading.local()


def _note_attempt(retry_state) -> None:
    _attempt_state.n = retry_state.attempt_number


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(8),
    before=_note_attempt,
)
def _call_with_retry(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    system: str | list[dict],
    messages: list[dict],
    temperature: float,
) -> anthropic.types.Message:
    # Extended thinking OFF everywhere — training data should show user-facing
    # reasoning, not internal scratchpads (see CLAUDE.md). Models in the Claude 5
    # family emit a thinking block by default, so disable it explicitly rather
    # than parse around it. Exception: the Mythos-class models (fable/mythos)
    # REQUIRE adaptive thinking — the API 400s on `thinking: disabled` — so for
    # those the flag is omitted, thinking runs, and _response_text strips the
    # thinking blocks. Their outputs are therefore generated WITH hidden
    # reasoning: comparison/eval arms only, never corpus generation, unless the
    # no-scratchpads design decision is deliberately revisited.
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    # temperature/top_p/top_k are REMOVED on the Claude 5 family and Opus 4.7+ —
    # the API 400s if any is sent. Only send temperature to models that still
    # accept it; the rest run at default sampling (steer them via prompting).
    if _accepts_sampling_params(model):
        kwargs["temperature"] = temperature
    if not _requires_adaptive_thinking(model):
        kwargs["thinking"] = {"type": "disabled"}
    return client.messages.create(**kwargs)


_ADAPTIVE_THINKING_PREFIXES = ("claude-fable", "claude-mythos")


def _requires_adaptive_thinking(model: str) -> bool:
    """Mythos-class models cannot run with thinking disabled (400); callers
    omit the flag for them and rely on _response_text to strip the blocks."""
    return model.startswith(_ADAPTIVE_THINKING_PREFIXES)


# Models that reject sampling parameters (temperature/top_p/top_k) with a 400:
# the Claude 5 family (Fable/Mythos/Sonnet 5) and Opus 4.7+. Older models
# (Opus 4.6/4.5, Sonnet 4.6/4.5, Haiku 4.5, …) still accept temperature.
_NO_SAMPLING_PREFIXES = (
    "claude-fable", "claude-mythos",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-5",
    "claude-sonnet-5",
)


def _accepts_sampling_params(model: str) -> bool:
    """False for models that 400 on temperature/top_p/top_k (Claude 5 family,
    Opus 4.7+). Callers pass temperature unconditionally; this gates whether it
    reaches the API."""
    return not model.startswith(_NO_SAMPLING_PREFIXES)


def _response_text(response: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response, skipping any non-text blocks
    (e.g. a thinking block that slips through). Returns '' if there is no text."""
    return "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )


def _classify_claude_code_error(message: str) -> Exception:
    if _LIMIT_PATTERN.search(message):
        return UsageLimitExceeded(
            f"Claude subscription usage limit reached: {message}\n"
            "Progress is checkpointed — wait for your usage window to reset, "
            "then continue this run with --resume."
        )
    if _REFUSAL_PATTERN.search(message):
        return ClaudeCodeRefusal(message)
    return ClaudeCodeError(message)


def _sdk_importable() -> bool:
    """True when the claude-agent-sdk package is importable (the subscription
    path's in-process requirement; the CLI itself is probed by the first call)."""
    import importlib.util
    return importlib.util.find_spec("claude_agent_sdk") is not None


def _demote_cc(reason: str) -> None:
    """auto backend: take the subscription path out of play for the rest of the
    run and say so loudly — every remaining call is served by the api backend
    (billed to ANTHROPIC_API_KEY). Progress made so far is unaffected."""
    global _cc_demoted
    if _cc_demoted is not None:
        return
    _cc_demoted = reason
    print(
        f"  NOTICE: backend 'auto' — subscription path unavailable ({reason}). "
        "Falling back to the Anthropic API (billed to ANTHROPIC_API_KEY) for the "
        "rest of this run. Each cost-log record names the backend that served it.",
        file=sys.stderr,
    )


def _resolve_cc_system(system: str) -> str:
    """Effective system prompt for the claude_code backend.

    Claude Code injects its own agentic prompt when the system is empty, so an
    empty system gets a neutral stand-in. Warn once so the substitution — which
    notably changes the DAD `plain` condition — isn't silent.
    """
    global _neutral_system_warned
    if system:
        return system
    if not _neutral_system_warned:
        _neutral_system_warned = True
        print(
            "  WARNING: backend 'claude_code' substitutes a neutral system prompt for "
            "empty-system calls (Claude Code injects its own agentic prompt otherwise). "
            "Stages that rely on a truly empty system prompt are not reproduced faithfully "
            "here — notably the DAD response steps. Use backend: api for those.",
            file=sys.stderr,
        )
    return _NEUTRAL_SYSTEM


def _run_claude_code_query(
    model: str,
    system: str,
    user_message: str,
) -> tuple[str, int, int, float | None, str | None]:
    """One Claude Code CLI turn: run the query and parse the result into
    (text, input_tokens, output_tokens, notional_cost_usd, stop_reason).

    Raises UsageLimitExceeded / ClaudeCodeError on failure; the retry wrapper
    (_call_claude_code_with_retry) decides whether to retry. Kept separate from
    that wrapper so this parse — the money path — is unit-testable by stubbing
    claude_agent_sdk.query, without triggering tenacity's backoff.
    """
    try:
        import anyio
        from claude_agent_sdk import CLINotFoundError, ClaudeAgentOptions, query
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
    except ImportError as e:
        raise RuntimeError(
            "backend 'claude_code' requires the claude-agent-sdk package; "
            "run: pip install -r requirements.txt"
        ) from e

    resolved_system = _resolve_cc_system(system)
    system_file: str | None = None
    if len(resolved_system.encode("utf-8")) > _CC_SYSTEM_ARG_MAX_BYTES:
        fd, system_file = tempfile.mkstemp(prefix="claude_code_system_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(resolved_system)

    option_kwargs = dict(
        model=model,
        system_prompt=(
            {"type": "file", "path": system_file}
            if system_file is not None
            else resolved_system
        ),
        tools=[],  # pure text generation: no file/bash/web access
        max_turns=1,
        # Thinking disabled so training data shows user-facing reasoning only.
        # Mythos-class models (fable/mythos) require adaptive thinking and 400
        # on disabled, so the flag is omitted for them (mirrors the api
        # backend's _call_with_retry); their thinking blocks are stripped by
        # the TextBlock filter below.
        # Hermetic run: without these, the CLI loads the contributor's own
        # ~/.claude settings (custom agents, plan-by-default permission modes,
        # hooks), which leaks agentic scaffolding into generated text.
        setting_sources=[],
        permission_mode="default",
        # Blank out any key loaded from .env so the subprocess can't silently
        # bill the shared API key — Claude Code treats an empty value as unset
        # and falls back to its own login / CLAUDE_CODE_OAUTH_TOKEN.
        env={"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""},
    )
    if not _requires_adaptive_thinking(model):
        option_kwargs["thinking"] = {"type": "disabled"}
    options = ClaudeAgentOptions(**option_kwargs)

    async def _run() -> tuple[list[str], object | None]:
        text_parts: list[str] = []
        result_msg = None
        stream = query(prompt=user_message, options=options)
        try:
            async for msg in stream:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    # The result is final for our single-turn calls — stop here.
                    # After an is_error result the CLI exits non-zero; reading
                    # past the result turns that exit into a ProcessError whose
                    # text drops result_msg.result (the CLI's actual error),
                    # masking the is_error handling below and its usage-limit
                    # classification.
                    result_msg = msg
                    break
        finally:
            await stream.aclose()
        return text_parts, result_msg

    try:
        text_parts, result_msg = anyio.run(_run)
    except CLINotFoundError as e:
        raise RuntimeError(
            "backend 'claude_code' requires the Claude Code CLI "
            "(https://claude.com/claude-code); install it, then log in "
            "or set CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`."
        ) from e
    except Exception as e:  # CLI failures surface as assorted exception types
        raise _classify_claude_code_error(str(e)) from e
    finally:
        if system_file is not None:
            with contextlib.suppress(OSError):
                os.unlink(system_file)

    if result_msg is None:
        raise ClaudeCodeError("claude_code backend returned no result message")
    if result_msg.is_error:
        raise _classify_claude_code_error(
            result_msg.result or result_msg.subtype or "unknown claude_code error"
        )

    text = result_msg.result if result_msg.result is not None else "".join(text_parts)
    usage = result_msg.usage or {}
    return (
        text,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        result_msg.total_cost_usd,
        result_msg.stop_reason,
    )


@retry(
    retry=retry_if_exception_type(ClaudeCodeError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(8),
    before=_note_attempt,
)
def _call_claude_code_with_retry(
    model: str,
    system: str,
    user_message: str,
) -> tuple[str, int, int, float | None, str | None]:
    """Retry wrapper around _run_claude_code_query (transient ClaudeCodeError
    only; UsageLimitExceeded is not retried). This is the seam call_claude uses
    and that the test suite blocks."""
    return _run_claude_code_query(model, system, user_message)


def call_claude(
    user_message: str,
    system_prompt: str = "",
    injection: str = "",
    model: str | None = None,
    max_tokens: int | None = None,
    return_stop_reason: bool = False,
    stage: str | None = None,
    temperature: float | None = None,
    item_id: str | None = None,
    cache_system: bool = False,
) -> str | tuple[str, str | None]:
    """Call Claude and return the response text.

    Args:
        user_message: The user turn content.
        system_prompt: Optional system prompt.
        injection: Optional text appended to the system prompt.
        model: Model override; falls back to config value.
        max_tokens: Token limit override; falls back to config value. Enforced
            on the api backend only — Claude Code applies its own output cap,
            and the pipeline's caps exist to bound per-token API cost, which
            does not apply to subscription usage.
        return_stop_reason: if True, return (text, stop_reason) so the caller can
            reject truncated/refused completions instead of storing them.
        stage: Pipeline-stage tag written into the cost-log record (e.g.
            "prompt_draft", "layer4") so spend can be broken down per stage.
        temperature: Sampling temperature override for this call; falls back to
            the config `temperature` (1.0 — corpus generation wants diversity).
        item_id: Id of the pipeline record this call serves (e.g. a prompt_id
            or response_id; comma-joined ids for a batched call), written into
            the cost-log record so per-record stats can be looked up later.
        cache_system: if True (api backend only), send the system prompt as an
            ephemeral prompt-cache block so repeated calls that share it (e.g. an
            SDF layer's constitution-laden system prompt) are billed at the ~0.1x
            cache-read rate after the first. Ignored on claude_code (the CLI
            manages its own caching) and when the system prompt is empty.

    Returns:
        The assistant's response text, or (text, stop_reason) when
        return_stop_reason is True.
    """
    resolved_model = model or _config.get("model", "claude-sonnet-5")
    resolved_max = max_tokens or _config.get("max_tokens", 4000)
    resolved_temp = temperature if temperature is not None else _config.get("temperature", 1.0)

    full_system = system_prompt
    if injection:
        full_system = (full_system + "\n\n" + injection).strip()

    _attempt_state.n = 1  # _note_attempt overwrites this on every real attempt
    started = time.monotonic()

    # Backend routing. "auto" prefers the subscription (claude_code) and falls
    # back to the api key when it can't serve the call:
    #   - empty-system calls (the DAD baseline arm) always go to the api, so the
    #     plain-model condition stays exact (claude_code would substitute a
    #     neutral system prompt);
    #   - once demoted (sdk missing, CLI missing, usage window exhausted, or a
    #     persistently failing CLI), the rest of the run is served by the api.
    use_cc = _backend == "claude_code"
    if _backend == "auto":
        if _cc_demoted is None and not _sdk_importable():
            _demote_cc("claude-agent-sdk is not installed — pip install -r requirements.txt")
        use_cc = _cc_demoted is None and bool(full_system)

    if use_cc:
        # The Claude Code CLI exposes no sampling-temperature control, so a
        # non-default temperature cannot be honored on this backend. The config
        # default (1.0) matches normal sampling, so only deliberate overrides
        # warrant the warning.
        global _temperature_warned
        if resolved_temp != 1.0 and not _temperature_warned:
            _temperature_warned = True
            print(
                f"  WARNING: temperature={resolved_temp} requested, but backend "
                "'claude_code' cannot set sampling temperature — calls run at the "
                "CLI's default sampling. Use backend: api for temperature-sensitive runs.",
                file=sys.stderr,
            )
        try:
            text, input_tokens, output_tokens, cost, stop_reason = _call_claude_code_with_retry(
                model=resolved_model,
                system=full_system,
                user_message=user_message,
            )
        except ClaudeCodeRefusal as e:
            # A content/policy refusal is per-item, not the backend's health:
            # surface it exactly like the api leg's refusal (empty text,
            # stop_reason='refusal') so the callers' per-item rejection
            # machinery handles it. One refused item must never abort a run —
            # or, under auto, demote the whole backend.
            print(f"  WARNING: claude_code refused this request on policy grounds "
                  f"(model {resolved_model}) — surfaced as stop_reason='refusal'. "
                  f"{str(e).splitlines()[0][:160]}", file=sys.stderr)
            _log_usage(resolved_model, 0, 0, cost_usd=0.0, stage=stage,
                       item_id=item_id, duration_s=time.monotonic() - started,
                       attempts=_attempt_state.n, backend="claude_code")
            return ("", "refusal") if return_stop_reason else ""
        except (UsageLimitExceeded, ClaudeCodeError, RuntimeError) as e:
            if _backend != "auto":
                raise
            # auto: the subscription can't serve this run any more (window
            # exhausted, CLI missing, or 8 straight transient failures) —
            # demote and re-serve THIS call via the api path below.
            _demote_cc(str(e).split("\n")[0])
            use_cc = False
        else:
            _log_usage(resolved_model, input_tokens, output_tokens, cost_usd=cost, stage=stage,
                       item_id=item_id, duration_s=time.monotonic() - started,
                       attempts=_attempt_state.n, backend="claude_code")
            # Same suspect-stop-reason guard as the api path below; Claude Code
            # reports stop_reason on its ResultMessage (e.g. "end_turn").
            if stop_reason not in ("end_turn", "stop_sequence"):
                print(f"  WARNING: response stop_reason={stop_reason!r} "
                      f"(model {resolved_model}, backend claude_code) — output may be "
                      "truncated or refused.", file=sys.stderr)
            return (text, stop_reason) if return_stop_reason else text

    # A non-default temperature can't be honored on models that drop sampling
    # params — surface it once (mirrors the claude_code temperature warning).
    # _temperature_warned is already declared global earlier in this function.
    if resolved_temp != 1.0 and not _accepts_sampling_params(resolved_model) and not _temperature_warned:
        _temperature_warned = True
        print(f"  WARNING: temperature={resolved_temp} requested, but model "
              f"{resolved_model!r} does not accept sampling parameters (removed on the "
              "Claude 5 family / Opus 4.7+) — the call runs at default sampling.",
              file=sys.stderr)

    # Cache the (static, often constitution-sized) system prompt when asked, so
    # calls sharing it pay the cache-read rate after the first. An empty system
    # can't be cached.
    system_arg: str | list[dict] = full_system
    if cache_system and full_system:
        system_arg = [{
            "type": "text",
            "text": full_system,
            "cache_control": {"type": "ephemeral"},
        }]

    response = _call_with_retry(
        client=_get_client(),
        model=resolved_model,
        max_tokens=resolved_max,
        system=system_arg,
        messages=[{"role": "user", "content": user_message}],
        temperature=resolved_temp,
    )

    usage = response.usage
    _log_usage(resolved_model, usage.input_tokens, usage.output_tokens,
               stage=stage, item_id=item_id,
               duration_s=time.monotonic() - started, attempts=_attempt_state.n,
               cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
               cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
               # auto runs demoted to this path are still served by the api
               backend="api")
    # A completion that stopped for any reason other than end_turn/stop_sequence
    # is suspect — max_tokens truncates mid-text, refusal yields little or none.
    # Warn loudly so it isn't silently written into a corpus; callers that build
    # training records should also reject on stop_reason via return_stop_reason.
    if response.stop_reason not in ("end_turn", "stop_sequence"):
        print(f"  WARNING: response stop_reason={response.stop_reason!r} "
              f"(model {resolved_model}, max_tokens {resolved_max}) — output may be "
              "truncated or refused.", file=sys.stderr)
    text = _response_text(response)
    return (text, response.stop_reason) if return_stop_reason else text


def get_total_cost() -> float:
    """Sum cost_usd from the cost log and return total."""
    if _cost_log_path is None or not _cost_log_path.exists():
        return 0.0
    total = 0.0
    with open(_cost_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += json.loads(line).get("cost_usd", 0.0)
    return round(total, 4)
