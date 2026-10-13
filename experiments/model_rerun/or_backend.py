#!/usr/bin/env python3
"""OpenRouter drop-in for shared.api.call_claude (experiment use only).

install() replaces shared.api.call_claude with a function of the same contract
that serves every call via OpenRouter's /chat/completions, so the whole DAD
response side (2a scope, 2a.5 select, 2b respond, step-3 rewrite) runs on a
non-Anthropic model without touching pipeline code. Usage is logged through
shared.api._log_usage (backend "openrouter"; cost_usd is OpenRouter's own
reported cost when present).

stop_reason mapping (the pipeline's reject-truncated/refused machinery keys on
these): stop -> end_turn, length -> max_tokens, content_filter / a bare
refusal message / a 403 moderation error -> refusal.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared import api  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 8
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 522, 524}


def _verify():
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
    return bundle if Path(bundle).exists() else True


def _post_chat(payload: dict) -> dict:
    """One POST to OpenRouter, returned as parsed JSON. Seam for the offline
    smoke test. Raises for non-retryable HTTP errors."""
    import requests

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=600,
        verify=_verify(),
    )
    if resp.status_code in RETRY_STATUS:
        raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code == 403:
        # OpenRouter surfaces provider moderation as 403 — a refusal, not an error.
        return {"_moderation_refusal": True, "_body": resp.text[:500]}
    resp.raise_for_status()
    return resp.json()


class _Retryable(Exception):
    pass


def call_openrouter(
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
):
    """Same contract as shared.api.call_claude (cache_system is accepted and
    ignored — OpenRouter manages its own caching)."""
    resolved_model = model or api._config.get("model")
    resolved_max = max_tokens or api._config.get("max_tokens", 4000)
    resolved_temp = temperature if temperature is not None else api._config.get("temperature", 1.0)

    full_system = system_prompt
    if injection:
        full_system = (full_system + "\n\n" + injection).strip()
    messages = []
    if full_system:
        messages.append({"role": "system", "content": full_system})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": resolved_max,
        "temperature": resolved_temp,
        "usage": {"include": True},
    }

    started = time.monotonic()
    data = None
    attempts = 0
    for attempt in range(MAX_ATTEMPTS):
        attempts = attempt + 1
        try:
            data = _post_chat(payload)
        except _Retryable as e:
            wait = min(4 * (2 ** attempt), 60)
            print(f"  openrouter retry {attempts}/{MAX_ATTEMPTS} in {wait}s ({e})",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        except Exception as e:
            # connection-level errors are retryable too
            wait = min(4 * (2 ** attempt), 60)
            print(f"  openrouter retry {attempts}/{MAX_ATTEMPTS} in {wait}s "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            time.sleep(wait)
            continue
        # OpenRouter can return 200 with an error object (provider failures)
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code")
            if code in RETRY_STATUS or (isinstance(code, int) and code >= 500):
                wait = min(4 * (2 ** attempt), 60)
                print(f"  openrouter retry {attempts}/{MAX_ATTEMPTS} in {wait}s "
                      f"(provider error {err})", file=sys.stderr)
                time.sleep(wait)
                data = None
                continue
            if code == 403:
                data = {"_moderation_refusal": True, "_body": str(err)[:500]}
        break
    if data is None:
        raise RuntimeError(f"OpenRouter call failed after {MAX_ATTEMPTS} attempts "
                           f"(stage {stage}, item {item_id})")

    if data.get("_moderation_refusal"):
        print(f"  WARNING: openrouter moderation refused this request "
              f"(model {resolved_model}, item {item_id}) — surfaced as "
              f"stop_reason='refusal'. {data.get('_body', '')[:160]}", file=sys.stderr)
        api._log_usage(resolved_model, 0, 0, cost_usd=0.0, stage=stage,
                       item_id=item_id, duration_s=time.monotonic() - started,
                       attempts=attempts, backend="openrouter")
        return ("", "refusal") if return_stop_reason else ""

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    refusal_field = message.get("refusal")
    finish = choice.get("finish_reason")

    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "refusal",
        "tool_calls": "tool_use",
    }.get(finish, finish)
    if refusal_field and not text.strip():
        # OpenAI-style structured refusal: the model declined in a dedicated
        # field with no usable content.
        text = ""
        stop_reason = "refusal"

    usage = data.get("usage") or {}
    api._log_usage(
        resolved_model,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        cost_usd=usage.get("cost", 0.0),
        stage=stage,
        item_id=item_id,
        duration_s=time.monotonic() - started,
        attempts=attempts,
        backend="openrouter",
    )
    if stop_reason not in ("end_turn", "stop_sequence"):
        print(f"  WARNING: response stop_reason={stop_reason!r} "
              f"(model {resolved_model}, backend openrouter) — output may be "
              "truncated or refused.", file=sys.stderr)
    return (text, stop_reason) if return_stop_reason else text


def install():
    """Route every shared.api.call_claude call through OpenRouter. Returns the
    original function for restoring."""
    original = api.call_claude
    api.call_claude = call_openrouter
    return original
