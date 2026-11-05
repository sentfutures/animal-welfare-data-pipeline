"""Gemini API wrapper (generateContent over REST), with retry and cost tracking.

Used by any call whose model name starts with ``gemini-`` (see ``is_gemini_model``);
``shared.api.call_claude`` routes those calls here, so a DAD stage can be pointed at a
Gemini model through its usual model knob. The request is one user turn plus an optional
``systemInstruction``, at temperature 1, with ``thinkingConfig.thinkingLevel`` set from the
stage's effort. Thought parts are never requested (``includeThoughts`` is left off), so the
returned text is the answer alone.

The key (``GEMINI_API_KEY``) comes from shared/api_keys.py and is loaded only when a Gemini
model is actually called. httpx is already a dependency (openai's, and the embeddings leg).

Every call is appended to the pipeline's cost log (``shared.api._log_usage``) in the same
record shape as Claude calls, with backend ``gemini``. Thinking tokens are billed as output
and are included in ``output_tokens``; they are also recorded as ``reasoning_tokens``.
"""

import re
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from shared import api
from shared.api_keys import load_api_key

MODEL_NAME_PREFIX = "gemini-"
SAMPLING_TEMPERATURE = 1.0
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# USD per million tokens: (input, cached input, output). Thinking tokens are billed as output.
# Prompts here are far below the 200K-token tier boundary, so only the lower tier is listed.
# Prices as given by the team on 2026-10-29.
_PRICING = {
    "gemini-4-argon": (2.00, 0.20, 10.00),
    "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
}

# finishReason values that mean the answer is usable, cut off, or withheld. Anything else
# (OTHER, MALFORMED_FUNCTION_CALL, an unknown value) is passed through as-is, which callers
# treat as unusable.
_FINISHED = {"STOP"}
_TRUNCATED = {"MAX_TOKENS"}
_REFUSED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
            "IMAGE_SAFETY", "LANGUAGE"}


# A 429 for a per-day quota names it as e.g. "GenerateRequestsPerDayPerProjectPerModel"
# (quotaId) or "..._per_day" (metric); either spelling.
_DAILY_QUOTA = re.compile(r"per_?day", re.IGNORECASE)


class GeminiTransientError(Exception):
    """A rate limit, server error or dropped connection; retried."""


def is_gemini_model(model: str | None) -> bool:
    return bool(model) and model.startswith(MODEL_NAME_PREFIX)


def thinking_level(effort: str | None) -> str | None:
    """The pipeline's effort levels on Gemini's thinkingLevel scale. Gemini has no level above
    high, so xhigh and max are sent as high; None leaves the model's default."""
    if effort is None:
        return None
    return "high" if effort in ("xhigh", "max") else effort


@retry(
    retry=retry_if_exception_type(GeminiTransientError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(8),
    before=api._note_attempt,
)
def _generate(model: str, body: dict) -> dict:
    """The single network seam: tests replace this function."""
    key = load_api_key("GEMINI_API_KEY")   # raises MissingApiKeyError if absent
    try:
        response = httpx.post(_URL.format(model=model), headers={"x-goog-api-key": key},
                              json=body, timeout=450)
    except httpx.TransportError as e:
        raise GeminiTransientError(str(e)) from e
    if response.status_code == 429 and _DAILY_QUOTA.search(response.text):
        # A daily quota does not clear on any retry schedule; fail now rather than wait.
        raise RuntimeError(f"Gemini daily quota exhausted for {model}: {response.text[:600]}")
    if response.status_code == 429 or response.status_code >= 500:
        raise GeminiTransientError(f"HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini API HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def build_request_body(system_prompt: str, user_message: str, max_output_tokens: int,
                       level: str | None, json_schema: dict | None = None) -> dict:
    generation_config = {"temperature": SAMPLING_TEMPERATURE, "maxOutputTokens": max_output_tokens}
    if level is not None:
        generation_config["thinkingConfig"] = {"thinkingLevel": level}
    if json_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseJsonSchema"] = json_schema
    body = {"contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": generation_config}
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    return body


def parse_response(data: dict) -> tuple[str, str]:
    """(answer text, stop_reason) with stop reasons in the names the pipeline checks: a
    finished answer is 'end_turn', a length cut 'max_tokens', a safety block 'refusal'."""
    candidates = data.get("candidates") or []
    if not candidates:
        # The prompt itself was blocked (promptFeedback.blockReason), or nothing came back.
        return "", "refusal" if (data.get("promptFeedback") or {}).get("blockReason") else "empty"
    candidate = candidates[0]
    text = "".join(part.get("text", "") for part in (candidate.get("content") or {}).get("parts") or []
                   if not part.get("thought"))
    reason = candidate.get("finishReason") or "STOP"
    if reason in _FINISHED:
        return text, "end_turn"
    if reason in _TRUNCATED:
        return text, "max_tokens"
    if reason in _REFUSED:
        return "", "refusal"
    return text, reason.lower()


def compute_cost_usd(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    input_price, cached_price, output_price = _PRICING[model]
    return ((input_tokens - cached_input_tokens) * input_price
            + cached_input_tokens * cached_price
            + output_tokens * output_price) / 1_000_000


def call_gemini(
    user_message: str,
    system_prompt: str = "",
    *,
    model: str,
    max_output_tokens: int,
    effort: str | None = None,
    stage: str | None = None,
    item_id: str | None = None,
    json_schema: dict | None = None,
) -> tuple[str, str]:
    """One Gemini call. Returns (text, stop_reason); stop_reason is 'end_turn' only for a
    complete answer, so callers reject truncated or refused ones exactly as they do for
    ``api.call_claude(return_stop_reason=True)``. ``max_output_tokens`` includes thinking."""
    if model not in _PRICING:
        raise ValueError(f"model {model!r} has no price in shared/gemini_api.py _PRICING; "
                         "add it so the cost log is correct")
    api._attempt_state.n = 1
    started = time.monotonic()
    body = build_request_body(system_prompt, user_message, max_output_tokens,
                              thinking_level(effort), json_schema)
    try:
        data = _generate(model, body)
    except Exception as e:
        api._log_model_call(model, "gemini", system_prompt, [user_message], response_text=None,
                            stage=stage, item_id=item_id, attempts=api._attempt_state.n,
                            error=f"{type(e).__name__}: {str(e)[:500]}")
        raise

    usage = data.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount", 0) or 0
    cached_tokens = usage.get("cachedContentTokenCount", 0) or 0
    thinking_tokens = usage.get("thoughtsTokenCount", 0) or 0
    output_tokens = (usage.get("candidatesTokenCount", 0) or 0) + thinking_tokens
    api._log_usage(
        model, input_tokens, output_tokens,
        cost_usd=compute_cost_usd(model, input_tokens, cached_tokens, output_tokens),
        stage=stage, item_id=item_id, duration_s=time.monotonic() - started,
        attempts=api._attempt_state.n, cache_read_tokens=cached_tokens,
        reasoning_tokens=thinking_tokens, backend="gemini")

    text, stop_reason = parse_response(data)
    api._log_model_call(
        model, "gemini", system_prompt, [user_message], response_text=text,
        stop_reason=stop_reason,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens,
               "cache_read_tokens": cached_tokens, "reasoning_tokens": thinking_tokens},
        stage=stage, item_id=item_id, attempts=api._attempt_state.n)
    if stop_reason != "end_turn":
        print(f"  WARNING: response stop_reason={stop_reason!r} (model {model}, "
              f"max_output_tokens {max_output_tokens}) — output may be truncated or refused.",
              flush=True)
    return text, stop_reason
