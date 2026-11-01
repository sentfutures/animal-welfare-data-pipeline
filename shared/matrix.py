"""Generic combinatorial-matrix machinery shared by the pipeline composers.

Extracted verbatim from sdf_pipeline/compose_prompts.py (which re-exports it,
so SDF callers are unchanged) when the DAD pipeline grew its own composer
(dad_pipeline/compose_scenarios.py). Everything here is pipeline-agnostic:
parsing a variables.txt file, validating weights, deck-sampling assignments
whose per-variable shares match the weights by construction, and splitting a
rendered template into system/user sections.

variables.txt format::

    {varname}  # optional description
        plain value
        0.25 :: weighted value
        0.25 :: [stable_id] weighted value with an ID

An optional ``[id] `` prefix (after the weight, if any) gives a value a stable
ID: lowercase letters, digits and underscores, starting with a letter. The
prefix is stripped, so the value text that is rendered and recorded is exactly
what it would be without it; the IDs are exposed separately by
``parse_variables_with_ids`` for callers (the SDF deal constraints) that refer
to values by ID so a value can be reworded without breaking them.

Full-line ``#`` comments and blank lines are ignored anywhere. Weights are
all-or-nothing within a variable and must sum to 1.0 (within
WEIGHT_TOLERANCE); an unweighted variable is uniform.
"""

from __future__ import annotations

import math
import random
import re
import string
from pathlib import Path

VAR_HEADER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
WEIGHT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*::\s*(.+)$")
WEIGHT_TOLERANCE = 1e-6
VALUE_ID_RE = re.compile(r"^\[([a-z][a-z0-9_]*)\]\s+(\S.*)$")

# Prompt files may carry labeled system/user sections; split_sections() parses them.
SYSTEM_MARKER = "=== SYSTEM PROMPT ==="
USER_MARKER = "=== USER PROMPT ==="


def parse_variables_with_ids(
    path: Path, reserved: frozenset[str] | set[str] = frozenset()
) -> tuple[dict[str, list[tuple[str, float | None]]], dict[str, list[str | None]]]:
    """Parse variables.txt into ({name: [(value, weight-or-None), ...]},
    {name: [id-or-None, ...]}) — the second parallel to the first.

    ``reserved`` names are composer-injected slots that must NOT be defined in
    the file — defining one raises, pointing the author at the composer. A
    value starting with ``[`` must carry a well-formed ``[id] `` prefix; IDs
    must be unique within a variable.
    """
    parsed: dict[str, list[tuple[str, float | None]]] = {}
    ids: dict[str, list[str | None]] = {}
    current: str | None = None
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            header = VAR_HEADER_RE.fullmatch(line.strip().split("#")[0].strip())
            if header is None:
                raise ValueError(
                    f"{path.name}:{lineno}: expected a {{variable}} header "
                    f"or an indented value, got: {line!r}"
                )
            current = header.group(1)
            if current in reserved:
                raise ValueError(
                    f"{path.name}:{lineno}: {{{current}}} is reserved — "
                    "the composer injects it; don't define it in variables.txt"
                )
            if current in parsed:
                raise ValueError(f"{path.name}:{lineno}: duplicate variable {{{current}}}")
            parsed[current] = []
            ids[current] = []
        else:
            if current is None:
                raise ValueError(f"{path.name}:{lineno}: value line before any {{variable}} header")
            text = line.strip()
            weighted = WEIGHT_RE.match(text)
            weight = float(weighted.group(1)) if weighted else None
            value = weighted.group(2).strip() if weighted else text
            value_id = None
            if value.startswith("["):
                with_id = VALUE_ID_RE.match(value)
                if with_id is None:
                    raise ValueError(
                        f"{path.name}:{lineno}: {{{current}}} value starts with '[' but has no "
                        "well-formed '[id] ' prefix (lowercase letters, digits, underscores, "
                        f"starting with a letter, then a space): {value[:60]!r}"
                    )
                value_id, value = with_id.group(1), with_id.group(2).strip()
                if value_id in ids[current]:
                    raise ValueError(
                        f"{path.name}:{lineno}: duplicate ID [{value_id}] in {{{current}}}")
            parsed[current].append((value, weight))
            ids[current].append(value_id)
    return parsed, ids


def parse_variables(
    path: Path, reserved: frozenset[str] | set[str] = frozenset()
) -> dict[str, list[tuple[str, float | None]]]:
    """Parse variables.txt into {name: [(value, weight-or-None), ...]}.

    Any ``[id] `` prefixes are stripped (see parse_variables_with_ids for the IDs).
    """
    return parse_variables_with_ids(path, reserved)[0]


def value_ids_by_variable(
    parsed: dict[str, list[tuple[str, float | None]]],
    ids: dict[str, list[str | None]],
    require_ids: bool = False,
    source_name: str = "variables.txt",
) -> dict[str, dict[str, str]]:
    """{variable: {id: value_text}}. With ``require_ids`` every value must carry
    an ID, or this raises naming the variable and the first offending value."""
    by_variable: dict[str, dict[str, str]] = {}
    for name, pairs in parsed.items():
        by_variable[name] = {}
        for (value, _), value_id in zip(pairs, ids[name]):
            if value_id is None:
                if require_ids:
                    raise ValueError(
                        f"{source_name}: {{{name}}} value has no [id] prefix: {value[:70]!r}")
                continue
            by_variable[name][value_id] = value
    return by_variable


def split_weights(
    parsed: dict[str, list[tuple[str, float | None]]],
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    """Split into values + normalized weights, validating the weight rules."""
    values: dict[str, list[str]] = {}
    weights: dict[str, list[float]] = {}
    for name, pairs in parsed.items():
        vals = [v for v, _ in pairs]
        ws = [w for _, w in pairs]
        if all(w is None for w in ws):
            weights[name] = [1.0 / len(vals)] * len(vals) if vals else []
        elif any(w is None for w in ws):
            raise ValueError(
                f"{{{name}}}: weights are all-or-nothing — every value needs a "
                "'<weight> ::' prefix, or none"
            )
        else:
            total = sum(ws)
            if abs(total - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError(f"{{{name}}}: weights sum to {total:.6f}, expected 1.0")
            weights[name] = [float(w) for w in ws]
        values[name] = vals
    return values, weights


def template_placeholders(template: str) -> list[str]:
    """Format-field names in first-appearance order, deduplicated.

    Uses string.Formatter, i.e. exactly what str.format will substitute —
    stray literal braces surface here as the same error rendering would hit.
    """
    names: list[str] = []
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if spec or conversion or not re.fullmatch(r"[A-Za-z0-9_]+", field):
            raise ValueError(f"unsupported placeholder {{{field}}} — plain names only")
        if field not in names:
            names.append(field)
    return names


def matrix_axes(template: str, reserved: frozenset[str] | set[str] = frozenset()) -> list[str]:
    return [n for n in template_placeholders(template) if n not in reserved]


def largest_remainder(weights: list[float], n: int) -> list[int]:
    """Integer quotas summing to n, proportional to weights (largest remainder)."""
    exact = [w * n for w in weights]
    quotas = [math.floor(e) for e in exact]
    shortfall = n - sum(quotas)
    by_remainder = sorted(range(len(weights)), key=lambda i: exact[i] - quotas[i], reverse=True)
    for i in by_remainder[:shortfall]:
        quotas[i] += 1
    return quotas


def deck_sample(
    values: dict[str, list[str]],
    weights: dict[str, list[float]],
    axes: list[str],
    n: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """n assignments whose per-variable value counts match the weights exactly."""
    decks = []
    for name in axes:
        quotas = largest_remainder(weights[name], n)
        deck = [values[name][i] for i, q in enumerate(quotas) for _ in range(q)]
        rng.shuffle(deck)
        decks.append(deck)
    return [dict(zip(axes, row)) for row in zip(*decks)]


def split_sections(rendered: str) -> tuple[str | None, str]:
    """Split a rendered prompt into (system, user) on the section markers.

    A file without markers is a single user prompt (system=None). The markers
    are display/authoring conventions only — nothing between them is sent to
    the API as-is; callers pass the pieces as system_prompt and user_message.
    """
    if USER_MARKER not in rendered:
        return None, rendered.strip()
    system_part, user_part = rendered.split(USER_MARKER, 1)
    system = system_part.replace(SYSTEM_MARKER, "", 1).strip()
    return (system or None), user_part.strip()
