"""Embeddings wrapper (OpenAI or Gemini) with retry logic and cost tracking.

Structured exactly like shared/api.py: a lazily-initialized module client, a
tenacity-wrapped transport call that retries only transient failures, and
per-call records appended to the same cost-log JSONL schema (output_tokens is
always 0 — embeddings bill input only). ``embed_texts`` is the single
chokepoint every caller uses; tests stub it the same way they stub
``shared.api.call_claude`` (see tests/conftest.py).

Two provider legs behind the one chokepoint, routed by model name: OpenAI
(``text-embedding-*``, needs OPENAI_API_KEY) and Gemini (``gemini-*``, needs
GEMINI_API_KEY; plain REST via httpx — openai's own pinned dependency — so no
new package). ``resolve_default_model()`` picks whichever provider has a key,
OpenAI first. IMPORTANT: diversity numbers are only comparable within one
embedding model — pick one and freeze it; reports refuse cross-model compares.

This is the embedding-based complement to the lexical word-shingle scan in
shared/textstats.py, whose docstring defers paraphrase-level semantic
duplication to embeddings. The Anthropic-driven pipelines never import this
module.
"""

import os
import json
import functools
import threading
import sys
from pathlib import Path
from datetime import datetime, UTC

import numpy as np
import openai
import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from shared.api_keys import load_api_key, load_optional_api_key

DEFAULT_MODEL = "text-embedding-3-small"

# The embeddings endpoint caps each request at 2048 inputs and ~300k total
# tokens, and each individual input at MAX_INPUT_TOKENS tokens. A batch closes
# at MAX_BATCH items or MAX_BATCH_CHARS characters, whichever comes first: item
# count alone can bust the token cap (128 full SDF documents at ~4k tokens each
# is ~512k tokens). Chars bound tokens, but NOT at 1 token/char — CJK runs
# ~1.25-1.5 tokens/char in cl100k, so 180k chars (worst case ~270k tokens)
# stays under the ~300k request cap for any script.
MAX_BATCH = 128
MAX_BATCH_CHARS = 180_000

# text-embedding-3 rejects any single input over this many tokens, 400-ing the
# WHOLE request. embed_texts truncates to it as a safety net (see below).
MAX_INPUT_TOKENS = 8192

# tiktoken downloads its BPE ranks from a remote blob store on first use and
# caches them under TIKTOKEN_CACHE_DIR; the wheel ships no ranks. The offline
# test suite (pytest-socket --disable-socket) and a fresh CI runner have no
# warmed cache, so that first fetch would fail the required smoke check. We ship
# the cl100k_base ranks (the encoding every text-embedding-3 model uses) under
# vendor/tiktoken, keyed by tiktoken's own cache name, and point tiktoken there
# unless the caller already set the dir.
_VENDORED_TIKTOKEN_CACHE = Path(__file__).resolve().parent.parent / "vendor" / "tiktoken"

GEMINI_DEFAULT_MODEL = "gemini-embedding-001"
_GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:batchEmbedContents")
# batchEmbedContents caps at 100 requests; gemini-embedding-001 caps each
# input at 2048 tokens (we bound with the cl100k estimate, close enough as a
# safety net — the API truncates silently rather than 400ing on overflow).
GEMINI_MAX_BATCH = 100
GEMINI_MAX_INPUT_TOKENS = 2048

_config: dict = {}
_client: openai.OpenAI | None = None
_cost_log_path: Path | None = None
_cost_log_lock = threading.Lock()

# $ per million input tokens. Unknown models fall back to the 3-small rate
# WITH A WARNING — embedding spend is cents, but the log should stay honest.
_PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "gemini-embedding-001": 0.15,
}
_UNPRICED_WARNED: set = set()


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def resolve_default_model() -> str:
    """The embedding model to use when the caller doesn't name one: an explicit
    EMBEDDINGS_MODEL env var wins; otherwise whichever provider has a key,
    OpenAI first (the repo's original stack), then Gemini."""
    explicit = os.environ.get("EMBEDDINGS_MODEL")
    if explicit:
        return explicit
    if load_optional_api_key("OPENAI_API_KEY"):
        return DEFAULT_MODEL
    if load_optional_api_key("GEMINI_API_KEY") or load_optional_api_key("GOOGLE_API_KEY"):
        return GEMINI_DEFAULT_MODEL
    return DEFAULT_MODEL  # no key at all: fail later with the provider's error


def init(config_path: str = "config.yaml", cost_log_path: str | Path | None = None) -> None:
    global _config, _client, _cost_log_path
    with open(config_path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    _client = None  # constructed lazily; Gemini-only environments never need it
    _cost_log_path = Path(cost_log_path or _config["outputs"]["cost_log"])
    _cost_log_path.parent.mkdir(parents=True, exist_ok=True)


def _get_client() -> openai.OpenAI:
    global _client
    if _cost_log_path is None:
        init()
    if _client is None:
        key = load_api_key("OPENAI_API_KEY")   # raises MissingApiKeyError if absent
        _client = openai.OpenAI(api_key=key)
    return _client


def _log_usage(model: str, input_tokens: int) -> None:
    price = _PRICING.get(model)
    if price is None:
        if model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(model)
            print(
                f"  WARNING: model {model!r} is not in shared/embeddings.py _PRICING — "
                f"estimating cost at {DEFAULT_MODEL} rates ($0.02 per MTok). Add the "
                "model to _PRICING for accurate cost logs.",
                file=sys.stderr,
            )
        price = _PRICING[DEFAULT_MODEL]
    cost = (input_tokens / 1_000_000) * price
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "cost_usd": round(cost, 6),
    }
    with _cost_log_lock, open(_cost_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class GeminiTransientError(Exception):
    """Gemini embeddings 429/5xx/connection failure; retried by tenacity."""


# Same predicate rationale as shared/api.py: retrying a deterministic 4xx (bad
# request, auth, over-context input) burns 8 exponential-backoff attempts
# before surfacing the real error — retry only rate limits, 5xx, and
# connection/timeout.
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    GeminiTransientError,
)


def _gemini_transport(model: str, batch: list[str]) -> tuple[list[list[float]], int]:
    """One batchEmbedContents call. The response carries no usage block, so
    input tokens are estimated with the cl100k encoding (cost-log honesty at
    cents scale). httpx is openai's own pinned dependency — no new package."""
    import httpx

    key = load_optional_api_key("GEMINI_API_KEY") or load_optional_api_key("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "gemini-* embedding models need GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "in <repo root>/.env or the environment; set it, or use an OpenAI text-embedding-* model."
        )
    try:
        resp = httpx.post(
            _GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": key},
            json={"requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
                for t in batch]},
            timeout=120,
        )
    except httpx.TransportError as e:
        raise GeminiTransientError(str(e)) from e
    if resp.status_code == 429 or resp.status_code >= 500:
        raise GeminiTransientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    vectors = [e["values"] for e in resp.json()["embeddings"]]
    enc = _encoding_for(DEFAULT_MODEL)
    return vectors, sum(len(enc.encode(t)) for t in batch)


def _openai_normalize(response) -> tuple[list[list[float]], int]:
    # the API preserves order, but each item carries its index — trust that
    ordered = sorted(response.data, key=lambda d: d.index)
    return [d.embedding for d in ordered], response.usage.prompt_tokens


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(8),
)
def _embed_with_retry(client: openai.OpenAI | None, model: str,
                      batch: list[str]) -> tuple[list[list[float]], int]:
    """Provider-normalized transport: (vectors in input order, input tokens).
    The single seam tests block/stub — both provider legs live below it."""
    if _is_gemini(model):
        return _gemini_transport(model, batch)
    return _openai_normalize(client.embeddings.create(model=model, input=batch))


@functools.lru_cache(maxsize=8)
def _encoding_for(model: str):
    """tiktoken encoding for a model; falls back to cl100k_base (what the
    text-embedding-3 models use). Imported lazily so the module loads without
    tiktoken for callers that never truncate."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_VENDORED_TIKTOKEN_CACHE))
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def truncate_to_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS,
                       model: str = DEFAULT_MODEL) -> str:
    """Truncate text to at most max_tokens tokens under the model's encoding.

    Character-based truncation cannot bound tokens across scripts: CJK runs well
    over one token per character (~1.25-1.5 in cl100k), so a char cap that is
    safe for English still exceeds the model's input window and 400s the whole
    embedding request. Returns text unchanged when already within the cap."""
    enc = _encoding_for(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _batches(texts: list[str], max_items: int = MAX_BATCH):
    """Split texts into API batches: at most max_items items and (except for a
    single oversized text, which still goes alone) MAX_BATCH_CHARS chars each."""
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (len(batch) >= max_items or chars + len(t) > MAX_BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        yield batch


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """Embed texts and return an L2-normalized float32 matrix, one row per text.

    model=None resolves via resolve_default_model() (explicit EMBEDDINGS_MODEL,
    else whichever provider has a key, OpenAI first).

    Batching happens here (per-provider item cap / MAX_BATCH_CHARS chars per
    request); callers that want to checkpoint paid work between requests
    should chunk their own input and call this per chunk.

    Raises ValueError on empty/whitespace-only texts — the API rejects them
    mid-batch, so the caller must filter (and report) empties first.

    Inputs over the model's token window are truncated to it as a safety net:
    an over-window input 400s the whole batch, and callers that pre-truncate by
    characters cannot bound tokens for CJK/other dense scripts. Callers that
    care about how much was cut should truncate (and report) first.
    """
    model = model or resolve_default_model()
    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(
                f"embed_texts got an empty text at index {i} — filter empties out "
                "(and report them) before embedding."
            )
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    gemini = _is_gemini(model)
    cap = GEMINI_MAX_INPUT_TOKENS if gemini else MAX_INPUT_TOKENS
    texts = [truncate_to_tokens(t, cap, model) for t in texts]

    if _cost_log_path is None:
        init()
    client = None if gemini else _get_client()
    rows: list[list[float]] = []
    for batch in _batches(texts, GEMINI_MAX_BATCH if gemini else MAX_BATCH):
        vectors, input_tokens = _embed_with_retry(client, model, batch)
        _log_usage(model, input_tokens)
        rows.extend(vectors)

    X = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
