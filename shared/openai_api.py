"""OpenAI Responses API wrapper for schema-constrained calls, with retry and cost tracking.

Used by stages whose configured model name starts with ``gpt-`` (see
``is_openai_model``). The call shape is the Responses API with ``instructions`` +
``input``, ``reasoning.effort``, and strict ``json_schema`` structured output.
Temperature is 1; ``top_p`` is not sent (gpt-6.1-sol rejects it, and its default is 1).

The key (``OPENAI_API_KEY``) comes from shared/api_keys.py and is loaded only when an
OpenAI model is actually called. By default that is the repo root's own ``.env`` alone (the
operating system environment is not consulted); with ``--find-dotenv`` a ``.env`` found
searching upward is used, then the environment.

Every call is appended to the pipeline's cost log (``shared.api._log_usage``) in the same
record shape as Claude calls, with backend ``openai`` plus a ``reasoning_tokens`` field.
``input_tokens`` is the full input count and ``cache_read_tokens`` the cached part of it.
"""

import os
import threading
import time

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from shared import api
from shared.api_keys import load_api_key

MODEL_NAME_PREFIX = "gpt-"
DEFAULT_REASONING_EFFORT = "medium"
SAMPLING_TEMPERATURE = 1.0
DEFAULT_SCHEMA_NAME = "structured_answer"

# USD per million tokens: (input, cached input, output). Reasoning tokens are billed as output.
# Standard tier, from developers.openai.com/api/docs/pricing on 2026-10-21.
_PRICING = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
}

_client: openai.OpenAI | None = None
_client_lock = threading.Lock()


def is_openai_model(model: str | None) -> bool:
    return bool(model) and model.startswith(MODEL_NAME_PREFIX)


def _get_client() -> openai.OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            api_key = load_api_key("OPENAI_API_KEY")   # raises MissingApiKeyError if absent
            # Accept-Encoding gzip works around a decompression bug in some httpx versions.
            # max_retries=0 because tenacity below does the retries, so every attempt is counted.
            _client = openai.OpenAI(api_key=api_key, default_headers={"Accept-Encoding": "gzip"},
                                    max_retries=0, timeout=600)
        return _client


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(8),
    before=api._note_attempt,
)
def _create_response(client: openai.OpenAI, **request_arguments):
    """The single network seam: tests replace this function."""
    return client.responses.create(**request_arguments)


def build_request_arguments(model, system_prompt, user_message, max_output_tokens, reasoning_effort,
                            json_schema, schema_name):
    arguments = {
        "model": model,
        "input": user_message,
        "max_output_tokens": max_output_tokens,
        "temperature": SAMPLING_TEMPERATURE,
        "reasoning": {"effort": reasoning_effort},
    }
    if system_prompt:
        arguments["instructions"] = system_prompt
    if json_schema is not None:
        arguments["text"] = {"format": {"type": "json_schema", "name": schema_name,
                                        "schema": json_schema, "strict": True}}
    return arguments


def _stop_reason(response) -> str:
    """'end_turn' for a complete answer; otherwise why it is not usable
    ('refusal', the incomplete reason such as 'max_output_tokens', or the status)."""
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return "refusal"
    status = getattr(response, "status", "completed")
    if status == "completed":
        return "end_turn"
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) or status


def _output_text(response) -> str:
    return "".join(
        part.text
        for item in getattr(response, "output", None) or []
        if getattr(item, "type", None) == "message"
        for part in getattr(item, "content", None) or []
        if getattr(part, "type", None) == "output_text"
    )


def compute_cost_usd(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    input_price, cached_price, output_price = _PRICING[model]
    return ((input_tokens - cached_input_tokens) * input_price
            + cached_input_tokens * cached_price
            + output_tokens * output_price) / 1_000_000


def call_openai(
    user_message: str,
    system_prompt: str = "",
    *,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    stage: str | None = None,
    item_id: str | None = None,
    json_schema: dict | None = None,
    schema_name: str = DEFAULT_SCHEMA_NAME,
) -> tuple[str, str]:
    """One schema-constrained OpenAI call. Returns (text, stop_reason); stop_reason is
    'end_turn' only for a complete answer, so callers can reject truncated or refused
    ones the same way they do for ``api.call_claude(return_stop_reason=True)``.
    ``max_output_tokens`` includes reasoning tokens."""
    if model not in _PRICING:
        raise ValueError(f"model {model!r} has no price in shared/openai_api.py _PRICING; "
                         "add it so the cost log is correct")
    client = _get_client()
    api._attempt_state.n = 1
    started = time.monotonic()
    try:
        response = _create_response(client, **build_request_arguments(
            model, system_prompt, user_message, max_output_tokens, reasoning_effort, json_schema, schema_name))
    except Exception as e:
        api._log_model_call(model, "openai", system_prompt, [user_message], response_text=None,
                            stage=stage, item_id=item_id, attempts=api._attempt_state.n,
                            error=f"{type(e).__name__}: {str(e)[:500]}")
        raise

    usage = response.usage
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
    reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
    api._log_usage(
        model, usage.input_tokens, usage.output_tokens,
        cost_usd=compute_cost_usd(model, usage.input_tokens, cached_tokens, usage.output_tokens),
        stage=stage, item_id=item_id, duration_s=time.monotonic() - started,
        attempts=api._attempt_state.n, cache_read_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens, backend="openai")

    stop_reason = _stop_reason(response)
    output_text = _output_text(response)
    api._log_model_call(
        model, "openai", system_prompt, [user_message], response_text=output_text,
        stop_reason=stop_reason,
        usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
               "cache_read_tokens": cached_tokens, "reasoning_tokens": reasoning_tokens},
        stage=stage, item_id=item_id, attempts=api._attempt_state.n)
    if stop_reason != "end_turn":
        print(f"  WARNING: response stop_reason={stop_reason!r} (model {model}, "
              f"max_output_tokens {max_output_tokens}) — output may be truncated or refused.",
              flush=True)
    return output_text, stop_reason
