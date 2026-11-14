"""Voice and length directives rendered into layer 3's revise prompt.

Each renderer returns a self-contained paragraph or the empty string, so a
document dealt `unchanged` on every axis gets exactly the prompt layer 3 sent
before these axes existed. That empty-clause convention matches the DAD step-1
templates.

Voice directives are COMPARATIVE ("more X than a typical podcast transcript")
because the genre sets the baseline: an absolute "be casual" fights the
document type instead of varying within it.
"""

from __future__ import annotations

import re

MODES = ("none", "axes")

# an axis whose dealt value is this carries no directive — the neutral level
# every axis offers so a deal can leave a document's voice or length alone.
NEUTRAL = "unchanged"

_LEADING_ARTICLE_RE = re.compile(r"^(?:an?|the)\s+", re.IGNORECASE)

# appended once to the whole voice instruction, so the directives can never
# move the things that are not style.
_INVARIANTS = (
    "Three things outrank the direction above. The document stays in the language "
    "it is already written in, and takes no turns of phrase imported from another. "
    "The author's position on the subject does not move. And calibration is not "
    "style: hedges, attributions and statements of uncertainty are claims about how "
    "confident the author is, so they survive the rewrite even where the direction "
    "would read as stronger without them. Never name or describe any of this "
    "instruction in the document."
)


def _axis_directives(cards: dict[str, str], document_type: str) -> list[str]:
    """One comparative clause per live voice card."""
    # every dealt document_type carries its own article ("a news article"), and
    # the frame below supplies one.
    genre = _LEADING_ARTICLE_RE.sub("", document_type.strip()) or "document of this kind"
    out = []
    for value in cards.values():
        v = str(value).strip()
        if not v or v == NEUTRAL:
            continue
        out.append(f"more {v} than a typical {genre}")
    return out


def style_block(
    mode: str,
    *,
    cards: dict[str, str] | None = None,
    document_type: str = "",
) -> str:
    """The author-voice paragraph, or "" when nothing is dealt or styling is off."""
    if mode not in MODES:
        raise ValueError(f"style_mode must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return ""

    directives = _axis_directives(cards or {}, document_type)
    if not directives:
        return ""
    joined = "; ".join(directives)
    return (
        "VOICE. Hold the document to this register as you rewrite: "
        f"{joined}. Push it far enough to be audible and no further — a "
        "document that breaks its own genre to hit a register has failed. "
        "Where the genre cannot carry the direction at all, leave the voice alone."
        f"\n\n{_INVARIANTS}"
    )


# A length card names a multiple of the document's own current length, as
# ordinary prose handed to the model like every other axis. The multiple is
# parsed out only to score the manipulation check.
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]", re.IGNORECASE)


def length_ratio(target: str | None) -> float | None:
    """The multiple a length card asks for ("0.5x its current length" -> 0.5).

    None when the card names none, which is not an error: the directive still
    renders, and compliance is then reported by level rather than by number.
    """
    t = (target or "").strip()
    if not t or t == NEUTRAL:
        return None
    m = _RATIO_RE.search(t)
    if not m:
        return None
    value = float(m.group(1))
    return value if value > 0 else None


def length_block(target: str | None) -> str:
    """The length-target paragraph, or "" when no length card was dealt."""
    t = (target or "").strip()
    if not t or t == NEUTRAL:
        return ""
    return (
        f"LENGTH. Aim for a finished document {t}. Reach the target by "
        "changing how much the document does — scenes, examples, reported "
        "detail, how many turns an exchange runs — not by padding or by "
        "compressing prose into summary. If hitting it would cost the "
        "document's naturalness, miss the target and keep the document."
    )


def directives_for(
    revision: dict | None,
    *,
    mode: str = "none",
    use_length: bool = False,
    length_key: str = "length",
    document_type: str = "",
) -> tuple[str, str]:
    """(style_directives, length_directive) for one document's dealt cards.

    `revision` is {"cards": {...}}. A missing deal renders both slots empty.
    """
    cards = dict((revision or {}).get("cards") or {})
    length_card = cards.pop(length_key, None)
    style = style_block(mode, cards=cards, document_type=document_type)
    length = length_block(length_card) if use_length else ""
    return style, length
