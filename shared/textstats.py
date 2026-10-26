"""Corpus text statistics: truncation repair and near-duplicate detection.

Deliberately embedding-free: near-duplicate detection uses cosine similarity
over hashed word-shingle count vectors (crc32 feature hashing), which is
deterministic across sessions, needs no GPU and no new heavy dependencies, and
catches the duplication that matters most in synthetic training data — same
skeleton, same phrasing. It will not catch pure paraphrase-level semantic
duplication; the audit tool's LLM pattern pass covers that angle.

Scale envelope: the pairwise scan is O(n²) in documents with hashed vectors of
dimension 16384 (float32), comfortable to ~10k documents on a laptop. Beyond
that, subsample (see evals/audit_sdf.py --dup-sample) or move to embeddings.
"""

from __future__ import annotations

import re
import unicodedata
import zlib

import numpy as np

# Sentence-final characters: a document ending on one of these is treated as
# complete. Includes closing quotes/brackets/ellipsis so quoted or parenthesized
# endings don't count as truncation, plus the terminal punctuation of the
# corpus's non-Latin scripts (CJK fullwidth stops/quotes, Devanagari danda,
# Arabic-script full stop and question mark) so multilingual documents aren't
# false-flagged as token-cap artifacts.
_TERMINAL_CHARS = '.!?"\'”’)…]:' + '。！？」』】）׃।॥۔؟'

# Closing delimiters that legitimately follow terminal punctuation, stripped
# before the terminal-punctuation test so a quoted or emphasised ending reads as
# finished: `...correctement. »` (French/Norwegian guillemets), `...patients.*`
# (markdown italic footer). Kept separate from _TERMINAL_CHARS because a bare
# delimiter is not itself a sentence ending.
_CLOSING_WRAPPERS = '"\'”’»›)]}*_' + '」』】）'

# A final line at or above either threshold reads as running prose, so an
# unpunctuated end to it is a truncation tell. Below both, it is a sign-off,
# byline, letterhead or list entry — see ends_mid_sentence. Two thresholds
# because word counts don't transfer across scripts: CJK lines carry no spaces
# (one "word"), so the character floor is what catches a truncated CJK document.
_PROSE_LINE_WORDS = 12
_PROSE_LINE_CHARS = 80

# Delimiters that make a final line a list of labels rather than one running
# clause: a signature block ("Dr. Amara Okonkwo | Senior Veterinary Officer |
# National Animal Welfare Board"), a letterhead, an email or nav footer, a
# See-also cross-reference row. Length alone cannot separate these from a
# truncation — they routinely clear both prose thresholds while being perfectly
# finished endings — so _is_label_row looks at the segments instead.
#
# The CJK commas (、，) are deliberately absent. They carry running CJK prose,
# and splitting on them drops every segment under the character floor, which is
# the only floor a spaceless script has; including them would have stopped the
# check seeing a truncated Chinese or Japanese document at all.
_LABEL_DELIMS = ",;|·•∙⋅／｜"
_LABEL_DELIM_RE = re.compile("[" + re.escape(_LABEL_DELIMS) + "]")

# Separators that introduce an attribution appended after a finished sentence,
# the shape forum and transcript registers close on: "...This one is about the
# ducks. — user:mod_greenhollow, 27 May".
_BYLINE_SEPARATORS = (" — ", " – ", " -- ", " ~ ")

_WORD_RE = re.compile(r"[\w']+")

DIM = 1 << 14  # 16384 hashed shingle buckets


# trim_unfinished() was removed on 2026-08-05. It cut a token-capped output back
# to its last complete sentence, and the draft stage once used it on the
# untagged-output fallback path (see code_quality/findings_v1_2026-07-10.json,
# which flags that it produced a superficially complete document flowing into
# the rewrite and score stages; it calls them layers 3 and 4-5, the numbering
# those stages carried before the renumber). That caller is long gone and
# nothing replaced it: every stage now rejects on stop_reason and refuses to
# checkpoint, so a truncated output is retried rather than salvaged. What remained was dead code that silently deleted a document's
# closing sign-off — it trimmed back to the newline above it — so it was deleted
# rather than guarded. Recover it from history if a salvage path is ever wanted.

_SEPARATOR_LINE_RE = re.compile(r"^[\s\-=*_~#]+$")


def strip_trailing_separators(text: str) -> str:
    """Drop trailing lines that are only separator characters (---, ***, ===).

    Generators sometimes close a document with a bare horizontal rule; that is
    a delimiter artifact, not a mid-sentence truncation, and the two need to be
    reported separately.
    """
    lines = (text or "").rstrip().splitlines()
    while lines and _SEPARATOR_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _reads_as_prose(line: str) -> bool:
    """True if a line is long enough to read as a running-prose clause.

    Two thresholds because word counts don't transfer across scripts — see
    _PROSE_LINE_WORDS.
    """
    return len(line.split()) >= _PROSE_LINE_WORDS or len(line) >= _PROSE_LINE_CHARS


def _is_label_row(line: str) -> bool:
    """True if the line is a list of short labels rather than one running clause.

    Signature blocks, letterheads, email and nav footers and See-also rows are
    all built this way, and they are the shape a length test gets wrong: a
    three-part signature clears both prose thresholds at around eighty
    characters while being a perfectly finished ending.

    Requires three or more delimited segments — one comma is punctuation,
    several are a list — and that no segment reads as prose on its own, judged
    by the same thresholds the whole line is judged by, so this adds no new
    magic numbers. A truncation cut inside such a list is missed, which is the
    sensitivity trade this check already documents below.
    """
    segments = [s for s in (p.strip() for p in _LABEL_DELIM_RE.split(line)) if s]
    return len(segments) >= 3 and not any(_reads_as_prose(s) for s in segments)


def _ends_with_byline(line: str) -> bool:
    """True if the line is a finished sentence plus a short attribution.

    Forum and transcript registers put the byline on the same line as the prose
    it closes: "...This one is about the ducks. — user:mod_greenhollow, 27 May".

    Both halves of the test are load-bearing. The head must carry terminal
    punctuation, or an em dash mid-clause reads as a byline separator and
    "...at the template minimum — which means most of four to six thous" stops
    being seen as the truncation it is; the tail must not itself read as prose,
    or a dash joining two clauses hides a cut in the second one.
    """
    for sep in _BYLINE_SEPARATORS:
        head, found, tail = line.rpartition(sep)
        if not found:
            continue
        head = head.strip().rstrip(_CLOSING_WRAPPERS).rstrip()
        tail = tail.strip()
        if head and tail and head[-1] in _TERMINAL_CHARS and not _reads_as_prose(tail):
            return True
    return False


def ends_mid_sentence(text: str) -> bool:
    """True if the text's final line reads as running prose cut mid-sentence.

    Deliberately narrow. It fires only when the last line looks like prose
    (>= _PROSE_LINE_WORDS words or >= _PROSE_LINE_CHARS characters), has no
    sentence-final punctuation once trailing closing delimiters are stripped,
    and is neither a label row nor prose closed by a byline (_is_label_row,
    _ends_with_byline).

    A shorter unpunctuated final line is treated as a legitimate ending,
    because that is what it nearly always is: a sign-off ("— Michelle", "中村"),
    a byline, a masthead or letterhead, a "See also" entry, a hashtag block, a
    podcast outro cue. Length alone was not enough: a sign-off is only as short
    as its job title, so "Dr. Amara Okonkwo | Senior Veterinary Officer |
    National Animal Welfare Board" and a semicolon-separated See-also row both
    cleared the prose thresholds and were reported as truncations. Segment
    shape, not line length, is what separates those from a cut clause.

    Measured over the 2124 documents committed under outputs/{sdf,dad}/runs:
    the naive "last character isn't punctuation" test flags 360 (16.9%), of
    which 2 are real truncations; the length-only rule flagged 17; this rule
    flags 8 (0.4%) — the same 2 real truncations plus four casual chat
    registers, a transcript line broken off on a dash, and a bare
    cross-reference title, all of which genuinely close without a full stop. No
    sign-off, letterhead or footer shape in the corpora is flagged at any length.

    The narrowing gives up sensitivity for a cut landing in the first few words
    of a line, or inside a list. That is an acceptable trade: the real defence
    is upstream, where layers 2 and 3 check stop_reason and refuse to checkpoint
    truncated output, so this check is the backstop for tail loss that never
    reached the API's stop_reason — a bad extraction or a dropped rewrite tail.

    Trailing separator-only lines are ignored — see strip_trailing_separators.
    """
    t = strip_trailing_separators(text)
    if not t:
        return False
    core = t.splitlines()[-1].strip().rstrip(_CLOSING_WRAPPERS).rstrip()
    if not core:
        return False
    if core[-1] in _TERMINAL_CHARS:
        return False
    # Emoji or a symbolic flourish closes an utterance in informal registers.
    if unicodedata.category(core[-1]) in ("So", "Sk"):
        return False
    if _is_label_row(core) or _ends_with_byline(core):
        return False
    return _reads_as_prose(core)


def has_trailing_separator(text: str) -> bool:
    """True if the text ends with one or more separator-only lines (--- etc.)."""
    t = (text or "").rstrip()
    return bool(t) and t != strip_trailing_separators(text)


def normalize_for_match(text: str) -> str:
    """Collapse whitespace and case for verbatim-containment checks."""
    return re.sub(r"\s+", " ", (text or "")).strip().casefold()


def _shingles(text: str, n: int = 3):
    words = _WORD_RE.findall((text or "").casefold())
    if not words:
        return
    if len(words) < n:
        yield " ".join(words)
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i : i + n])


def shingle_vector(text: str, n: int = 3) -> np.ndarray:
    """L2-normalized hashed word-n-gram count vector (deterministic via crc32)."""
    v = np.zeros(DIM, dtype=np.float32)
    for g in _shingles(text, n):
        v[zlib.crc32(g.encode("utf-8")) % DIM] += 1.0
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm
    return v


def shingle_matrix(texts: list[str], n: int = 3) -> np.ndarray:
    return np.stack([shingle_vector(t, n) for t in texts]) if texts else np.zeros((0, DIM), np.float32)


def near_dup_filter(
    texts: list[str], threshold: float, n: int = 3
) -> tuple[list[int], list[dict]]:
    """Greedy keep-first near-duplicate filter.

    Returns (keep_indices, dropped) where each dropped entry is
    {"index", "kept_index", "similarity"}: texts[index] was dropped for being
    within `threshold` cosine of the earlier kept texts[kept_index]. Greedy
    keep-first makes the result order-stable, so resumed runs and reruns drop
    the same items.
    """
    keep: list[int] = []
    dropped: list[dict] = []
    if not texts:
        return keep, dropped
    X = shingle_matrix(texts, n)
    kept_rows = np.zeros((len(texts), DIM), dtype=np.float32)
    for i in range(len(texts)):
        if keep:
            sims = kept_rows[: len(keep)] @ X[i]
            j = int(np.argmax(sims))
            if float(sims[j]) >= threshold:
                dropped.append(
                    {"index": i, "kept_index": keep[j], "similarity": round(float(sims[j]), 4)}
                )
                continue
        kept_rows[len(keep)] = X[i]
        keep.append(i)
    return keep, dropped


class IncrementalNearDup:
    """Streaming keep-first near-duplicate filter.

    ``near_dup_filter`` rebuilds the shingle matrix for its whole input on every
    call. When items arrive in many small batches that each dedup against all
    items kept so far (layer 2 filters each document type's subtypes against
    every subtype already accepted), that re-shingles every earlier item on
    every call — O(n²) in shingling across a run. This holds the kept vectors
    and shingles each item exactly once. It is keep-first and order-stable, so
    for a given stream it drops exactly what ``near_dup_filter`` would on the
    concatenated input, and ``seed_texts`` pre-loads already-kept items (e.g. a
    resumed run's existing records) without re-filtering them.
    """

    def __init__(self, threshold: float, n: int = 3, seed_texts: list[str] | None = None):
        self.threshold = float(threshold)
        self.n = n
        self._kept = np.zeros((1024, DIM), dtype=np.float32)
        self._count = 0
        for t in seed_texts or []:
            self._add(shingle_vector(t, n))

    def _add(self, v: np.ndarray) -> None:
        if self._count >= self._kept.shape[0]:
            self._kept = np.vstack([self._kept, np.zeros_like(self._kept)])
        self._kept[self._count] = v
        self._count += 1

    def filter(self, texts: list[str]) -> tuple[list[int], list[dict]]:
        """Return (keep_indices, dropped) for this batch; append kept vectors to
        state. Each dropped entry is {"index", "similarity"} — the batch-local
        index and its cosine to the nearest already-kept item."""
        keep: list[int] = []
        dropped: list[dict] = []
        for i, t in enumerate(texts):
            v = shingle_vector(t, self.n)
            if self._count:
                sims = self._kept[: self._count] @ v
                j = int(np.argmax(sims))
                if float(sims[j]) >= self.threshold:
                    dropped.append({"index": i, "similarity": round(float(sims[j]), 4)})
                    continue
            self._add(v)
            keep.append(i)
        return keep, dropped


def nearest_neighbor_sims(texts: list[str], n: int = 3, block: int = 512) -> np.ndarray:
    """Cosine similarity of each text to its nearest neighbor (for audit stats)."""
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=np.float32)
    X = shingle_matrix(texts, n)
    out = np.empty(len(texts), dtype=np.float32)
    for i in range(0, len(texts), block):
        sims = X[i : i + block] @ X.T
        rows = sims.shape[0]
        sims[np.arange(rows), np.arange(i, i + rows)] = -1.0  # mask self-similarity
        out[i : i + rows] = sims.max(axis=1)
    return out


# ------------------------------------------------------- length measurement
#
# Used by experiments/compare_arms.py to score the length-targeting experiment. The
# unit is per-language because a word count is meaningless for a script that
# does not put spaces between words — counting "words" in Chinese counts
# punctuation-delimited runs, which is not a length at all.
_UNSPACED_LANGUAGES = ("chinese", "mandarin", "japanese", "thai", "lao", "khmer")


# ------------------------------------------------------- structural form
# Diversity of FORM, as distinct from diversity of content. Semantic
# embeddings answer "are these about different things" and n-gram overlap
# answers "do they use different words"; neither notices that every document
# is built the same way. Used by evals/diversity.py.
#
# Raw document length is deliberately NOT a feature: a run that deals varied
# lengths would then score higher by construction, and the question these
# answer is whether varying length drags anything else along with it.

# Sentence enders across the scripts the corpus actually uses: Latin stops,
# the Devanagari danda, the Urdu full stop and the Arabic question mark. Reusing
# _TERMINAL_CHARS' repertoire — a Latin-only [.!?] reads a Hindi document as one
# 300-word sentence.
_SENT_END = ".!?\u3002\uff01\uff1f\u0964\u0965\u06d4\u061f\u05c3"
_SENT_SPLIT_RE = re.compile("[" + re.escape(_SENT_END) + "]+[\\s\"'\u201d\u2019)\\]]*")
_QUOTE_CHARS_RE = re.compile(r"[\"\u201c\u201d\u201e\u00ab\u00bb]")

# lexical features are read off a fixed window so they cannot track length
FORM_WINDOW = 300

# Architecture: how the document is BUILT. These are what a length target
# moves. Texture: word choice, which it does not.
FORM_STRUCTURAL = ("sent_len_mean", "sent_len_sd", "para_len_mean")
FORM_LEXICAL = ("ttr", "hapax", "word_len")
FORM_FEATURES = FORM_STRUCTURAL + FORM_LEXICAL + (
    "comma_per100",    # subordination and reported speech
    "quote_per100",
    "self_redundancy",
)


def form_features(text: str, window: int = FORM_WINDOW) -> dict[str, float] | None:
    """Length-normalised structural statistics for one document.

    None when the document is too short to measure stably (under `window`
    words, or fewer than three sentences).
    """
    words = _WORD_RE.findall(text.casefold())
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(words) < window or len(sents) < 3 or not paras:
        return None
    lens = [len(s.split()) for s in sents]
    win = words[:window]
    seen: dict[str, int] = {}
    for w in win:
        seen[w] = seen.get(w, 0) + 1
    raw = text.encode("utf-8")
    return {
        "sent_len_mean": float(np.mean(lens)),
        "sent_len_sd": float(np.std(lens)),
        "para_len_mean": float(np.mean([len(p.split()) for p in paras])),
        "ttr": len(seen) / window,
        "hapax": sum(1 for c in seen.values() if c == 1) / window,
        "word_len": float(np.mean([len(w) for w in words])),
        "comma_per100": 100.0 * text.count(",") / len(words),
        "quote_per100": 100.0 * len(_QUOTE_CHARS_RE.findall(text)) / len(words),
        "self_redundancy": len(zlib.compress(raw, 9)) / len(raw),
    }


def form_spread(feats: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Per-feature centre and spread across a corpus.

    Spread is what a length intervention is expected to move, and it is
    reported raw (not standardised) so two runs' reports can be compared
    directly without a shared reference population.
    """
    if not feats or any(f is None for f in feats):
        raise ValueError("form_spread needs non-empty features; filter form_features' Nones")
    out = {}
    for key in FORM_FEATURES:
        xs = np.array([f[key] for f in feats], dtype=float)
        med = float(np.median(xs))
        p10, p90 = (float(np.quantile(xs, 0.10)), float(np.quantile(xs, 0.90)))
        out[key] = {
            "median": round(med, 4),
            "sd": round(float(xs.std()), 4),
            "p10": round(p10, 4),
            "p90": round(p90, 4),
            # spread relative to the centre, so features in different units compare
            "cv": round(float(xs.std() / med), 4) if med else None,
        }
    return out


def form_summary(feats: list[dict[str, float]]) -> dict:
    """Spread of document form across a corpus, and the two headline numbers.

    `structural` is the mean relative spread of sentence and paragraph
    architecture, `lexical` the same over word choice. Reporting them apart is
    the point: an intervention can widen how documents are BUILT while leaving
    the words alone, and a single combined figure hides exactly that.
    """
    spread = form_spread(feats)

    def mean_cv(keys) -> float | None:
        vals = [spread[k]["cv"] for k in keys if spread[k]["cv"] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "n": len(feats),
        "structural_spread": mean_cv(FORM_STRUCTURAL),
        "lexical_spread": mean_cv(FORM_LEXICAL),
        "spread": spread,
        "features": list(FORM_FEATURES),
    }


def unit_for_language(language: str | None) -> str:
    """"chars" for languages written without word spacing, else "words"."""
    name = (language or "").strip().lower()
    return "chars" if any(k in name for k in _UNSPACED_LANGUAGES) else "words"


def count_units(text: str, unit: str) -> int:
    """Length of text in the given unit ("chars" or "words").

    Characters are counted after stripping whitespace runs to one, so the count
    does not move with how the document happens to be wrapped.
    """
    if unit == "words":
        return len(text.split())
    if unit == "chars":
        return len(re.sub(r"\s+", " ", text.strip()))
    raise ValueError(f"unit must be 'chars' or 'words', got {unit!r}")
