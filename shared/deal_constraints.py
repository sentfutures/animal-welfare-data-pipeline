"""Deal constraints: deterministic rules over dealt variable combinations.

A weighted variables matrix (shared/matrix.py) deals each variable's cards
independently, so some hands are impossible by definition (a document that
is "about nothing at stake" yet dealt "welfare is the central subject").
This module reads a lab-editable rules file (prompts/sdf/deal_constraints.txt)
and applies it to a deck-sampled deal. Pipeline-agnostic on purpose: it
imports nothing from sdf_pipeline, so another composer can adopt it.

Principle: rules only rule combinations OUT, never in. A rule excludes a
combination only if it is impossible by definition; anything else is left open
for a later screen. So the only requirement form is "is not one of".

Rules file format (plain text; full-line ``#`` comments and blank lines are
ignored). Values are referenced by their stable ID, the ``[id]`` prefix on the
value in variables.txt (see shared/matrix.py), never by their text, so a value
can be reworded without breaking a rule. An unknown ID fails loudly at load,
naming the rule::

    [rule_name]
    when <variable> is one of:
        value_id
        another_value_id
    require <variable> is not one of:      # swap-repair exclusion
        forbidden_value_id
    omit planning-prompt lines for:        # rendering rule
        variable_name
        another_variable_name

Every rule has exactly one ``when`` block plus exactly one ``require`` or
``omit`` block. ``require <variable> is one of:`` (inclusion) is rejected.

- A swap-repair requirement is enforced at deal time by swap_repair(): cards
  are swapped between hands within one variable's column, so every
  variable's value counts (and therefore its weighted share) are unchanged.
- A rendering rule leaves the dealt card in the hand (so shares are reported
  honestly) and only drops the template paragraph that renders those
  variables for hands matching the ``when`` block (see drop_template_lines).
- Without n_prompts (full-matrix enumeration) there are no weights to
  preserve, so rule-breaking combinations are filtered out instead.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

RULE_HEADER_RE = re.compile(r"^\[([A-Za-z0-9_]+)\]$")
WHEN_RE = re.compile(r"^when ([A-Za-z0-9_]+) is one of:$")
REQUIRE_RE = re.compile(r"^require ([A-Za-z0-9_]+) is (not )?one of:$")
EXCLUSION_ONLY_MESSAGE = (
    "deal rules only rule combinations out, never in: use 'require <variable> is not one of:' "
    "(anything left open is for the batch screen to catch)"
)
OMIT_LINE = "omit planning-prompt lines for:"

SWAP_KIND = "swap"
RENDER_KIND = "render"


class InfeasibleDealConstraintError(ValueError):
    """The dealt quotas cannot satisfy a rule, so no swap repair can succeed."""


@dataclass(frozen=True)
class DealRule:
    name: str
    kind: str                              # SWAP_KIND or RENDER_KIND
    condition_variable: str
    condition_values: tuple[str, ...]      # full variable values, resolved from IDs
    target_variable: str | None = None     # swap rules: the column whose card is swapped
    target_values: tuple[str, ...] = ()    # the excluded values of target_variable
    omitted_variables: tuple[str, ...] = ()  # rendering rules

    def applies_to(self, hand: dict[str, str]) -> bool:
        return hand[self.condition_variable] in self.condition_values

    def is_violated_by(self, hand: dict[str, str]) -> bool:
        """Swap rules only: the hand matches the condition but breaks the requirement."""
        if self.kind != SWAP_KIND or not self.applies_to(hand):
            return False
        return hand[self.target_variable] in self.target_values

    def card_satisfies(self, card: str) -> bool:
        return card not in self.target_values

    def describe(self) -> str:
        if self.kind == RENDER_KIND:
            return (f"{self.name}: when {self.condition_variable} matches, omit the planning-prompt "
                    f"lines for {', '.join(self.omitted_variables)}")
        return (f"{self.name}: when {self.condition_variable} matches, "
                f"{self.target_variable} must not be one of the listed values")


@dataclass
class RepairReport:
    """Per-rule statistics from one repair (or one filtering) pass over a deal."""
    violating_before: dict[str, int] = field(default_factory=dict)
    swaps_made: dict[str, int] = field(default_factory=dict)
    violating_after: dict[str, int] = field(default_factory=dict)
    hands_affected_by_rendering_rule: dict[str, int] = field(default_factory=dict)
    combinations_filtered_out: dict[str, int] = field(default_factory=dict)

    def format_lines(self, rules: list[DealRule]) -> list[str]:
        lines = []
        for rule in rules:
            if rule.kind == RENDER_KIND:
                lines.append(
                    f"  rule {rule.name}: rendering rule applied to "
                    f"{self.hands_affected_by_rendering_rule.get(rule.name, 0)} hands"
                )
            elif rule.name in self.combinations_filtered_out:
                lines.append(
                    f"  rule {rule.name}: {self.combinations_filtered_out[rule.name]} "
                    "combinations filtered out of the full matrix"
                )
            else:
                lines.append(
                    f"  rule {rule.name}: {self.violating_before.get(rule.name, 0)} hands "
                    f"violated before repair, {self.swaps_made.get(rule.name, 0)} swaps made, "
                    f"{self.violating_after.get(rule.name, 0)} violating after"
                )
        return lines


# --- parsing ---

def _resolve_id(
    rule_name: str, variable: str, value_id: str, value_ids: dict[str, dict[str, str]], where: str
) -> str:
    """The value text for an ID of ``variable``; raises naming the rule and ID."""
    if variable not in value_ids:
        raise ValueError(f"{where}: rule [{rule_name}] names unknown variable {{{variable}}}")
    if value_id not in value_ids[variable]:
        raise ValueError(
            f"{where}: rule [{rule_name}] names unknown ID {value_id!r} for {{{variable}}} "
            f"(known IDs: {', '.join(value_ids[variable]) or 'none'})"
        )
    return value_ids[variable][value_id]


def parse_deal_constraints(
    text: str, value_ids: dict[str, dict[str, str]], source_name: str = "deal_constraints.txt"
) -> list[DealRule]:
    """Parse and validate a rules file against {variable: {id: value_text}}."""
    # Collect raw rules first: {name: [(directive_line, [indented items], lineno)]}
    raw_rules: dict[str, list[tuple[str, list[str], int]]] = {}
    current_rule: str | None = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        where = f"{source_name}:{lineno}"
        if not line.startswith((" ", "\t")):
            header = RULE_HEADER_RE.match(line.strip())
            if header:
                current_rule = header.group(1)
                if current_rule in raw_rules:
                    raise ValueError(f"{where}: duplicate rule [{current_rule}]")
                raw_rules[current_rule] = []
            elif current_rule is None:
                raise ValueError(f"{where}: expected a [rule_name] header, got: {line!r}")
            else:
                raw_rules[current_rule].append((line.strip(), [], lineno))
        else:
            if current_rule is None or not raw_rules[current_rule]:
                raise ValueError(f"{where}: indented value before any directive")
            raw_rules[current_rule][-1][1].append(line.strip())

    rules: list[DealRule] = []
    for name, directives in raw_rules.items():
        where = f"{source_name}: rule [{name}]"
        when = [d for d in directives if WHEN_RE.match(d[0])]
        requires = [d for d in directives if REQUIRE_RE.match(d[0])]
        omits = [d for d in directives if d[0] == OMIT_LINE]
        unknown = [d for d in directives if d not in when + requires + omits]
        if unknown:
            raise ValueError(f"{source_name}:{unknown[0][2]}: unrecognised directive {unknown[0][0]!r}")
        if len(when) != 1 or len(requires) + len(omits) != 1:
            raise ValueError(
                f"{where}: needs exactly one 'when' block and exactly one "
                "'require' or 'omit' block"
            )
        when_line, when_items, when_lineno = when[0]
        condition_variable = WHEN_RE.match(when_line).group(1)
        if not when_items:
            raise ValueError(f"{source_name}:{when_lineno}: rule [{name}] 'when' block is empty")
        condition_values = tuple(
            _resolve_id(name, condition_variable, item, value_ids, source_name)
            for item in when_items
        )
        if requires:
            require_line, require_items, require_lineno = requires[0]
            match = REQUIRE_RE.match(require_line)
            if match.group(2) is None:
                raise ValueError(f"{source_name}:{require_lineno}: rule [{name}]: {EXCLUSION_ONLY_MESSAGE}")
            target_variable = match.group(1)
            if target_variable == condition_variable:
                raise ValueError(f"{where}: 'when' and 'require' name the same variable")
            if not require_items:
                raise ValueError(f"{source_name}:{require_lineno}: rule [{name}] 'require' block is empty")
            target_values = tuple(
                _resolve_id(name, target_variable, item, value_ids, source_name)
                for item in require_items
            )
            rules.append(DealRule(
                name=name, kind=SWAP_KIND, condition_variable=condition_variable,
                condition_values=condition_values, target_variable=target_variable,
                target_values=target_values,
            ))
        else:
            _, omit_items, omit_lineno = omits[0]
            if not omit_items:
                raise ValueError(f"{source_name}:{omit_lineno}: rule [{name}] 'omit' block is empty")
            for variable in omit_items:
                if variable not in value_ids:
                    raise ValueError(f"{where}: omits unknown variable {{{variable}}}")
            rules.append(DealRule(
                name=name, kind=RENDER_KIND, condition_variable=condition_variable,
                condition_values=condition_values, omitted_variables=tuple(omit_items),
            ))
    return rules


def load_deal_constraints(path: Path, value_ids: dict[str, dict[str, str]]) -> list[DealRule]:
    return parse_deal_constraints(path.read_text(encoding="utf-8"), value_ids, path.name)


# --- checking and repair ---

def swap_rules(rules: Iterable[DealRule]) -> list[DealRule]:
    return [rule for rule in rules if rule.kind == SWAP_KIND]


def violated_rule_names(rules: list[DealRule], hand: dict[str, str]) -> list[str]:
    return [rule.name for rule in rules if rule.is_violated_by(hand)]


def check_feasibility(hands: list[dict[str, str]], rules: list[DealRule]) -> None:
    """Raise if a swap rule needs more cards than the deal contains.

    Swaps never change a column's card counts, so the counts of the dealt
    hands are the quotas. Rule R needs one satisfying card in its target
    column for every hand whose condition card matches; fewer such cards than
    matching hands can never be repaired by swapping.
    """
    for rule in swap_rules(rules):
        needing = sum(1 for hand in hands if rule.applies_to(hand))
        satisfying = sum(1 for hand in hands if rule.card_satisfies(hand[rule.target_variable]))
        if needing > satisfying:
            raise InfeasibleDealConstraintError(
                f"deal constraint [{rule.name}] is infeasible: {needing} of {len(hands)} hands "
                f"match its condition on {{{rule.condition_variable}}} but only {satisfying} "
                f"{{{rule.target_variable}}} cards satisfy it ({rule.describe()})"
            )


def swap_repair(
    hands: list[dict[str, str]], rules: list[DealRule], rng: random.Random
) -> tuple[list[dict[str, str]], RepairReport]:
    """Repair rule-breaking hands by swapping cards within one column.

    Returns repaired copies of the hands (the inputs are not modified) and a
    RepairReport. Every swap exchanges one variable's card between two hands,
    so each variable's value counts are identical before and after. A swap is
    accepted only if the partner hand then satisfies ALL rules and the repaired
    hand's violations shrink without gaining new ones; passes repeat until no
    hand breaks a rule, and a pass that makes no swap raises.
    """
    rules = swap_rules(rules)
    hands = [dict(hand) for hand in hands]
    report = RepairReport()
    for rule in rules:
        report.violating_before[rule.name] = sum(rule.is_violated_by(hand) for hand in hands)
        report.swaps_made[rule.name] = 0
    check_feasibility(hands, rules)

    while True:
        violating_indices = [i for i, hand in enumerate(hands) if violated_rule_names(rules, hand)]
        if not violating_indices:
            break
        rng.shuffle(violating_indices)
        swaps_this_pass = 0
        for index in violating_indices:
            for rule in rules:
                if not rule.is_violated_by(hands[index]):
                    continue
                if _swap_one_card(hands, index, rule, rng, rules):
                    report.swaps_made[rule.name] += 1
                    swaps_this_pass += 1
        if swaps_this_pass == 0:
            stuck = {
                name for i in violating_indices for name in violated_rule_names(rules, hands[i])
            }
            raise InfeasibleDealConstraintError(
                f"swap repair made no progress: {len(violating_indices)} hands still break "
                f"rule(s) {sorted(stuck)}; no swap within one column leaves both hands valid"
            )

    for rule in rules:
        report.violating_after[rule.name] = sum(rule.is_violated_by(hand) for hand in hands)
    return hands, report


def _swap_one_card(
    hands: list[dict[str, str]], index: int, rule: DealRule, rng: random.Random,
    rules: list[DealRule],
) -> bool:
    """Swap hand[index]'s target-variable card with a seeded-random partner hand."""
    column = rule.target_variable
    hand = hands[index]
    before = set(violated_rule_names(rules, hand))
    partner_indices = [j for j in range(len(hands)) if j != index and hands[j][column] != hand[column]]
    rng.shuffle(partner_indices)
    for partner_index in partner_indices:
        partner = hands[partner_index]
        hand[column], partner[column] = partner[column], hand[column]
        after = set(violated_rule_names(rules, hand))
        accepted = (
            rule.name not in after
            and after <= before
            and not violated_rule_names(rules, partner)
        )
        if accepted:
            return True
        hand[column], partner[column] = partner[column], hand[column]  # undo
    return False


def without_rule_breaking_hands(
    hands: Iterable[dict[str, str]], rules: list[DealRule], report: RepairReport | None = None
) -> Iterator[dict[str, str]]:
    """Full-matrix enumeration: yield only combinations that satisfy every swap rule."""
    rules = swap_rules(rules)
    for hand in hands:
        broken = violated_rule_names(rules, hand)
        if broken:
            if report is not None:
                for name in broken:
                    report.combinations_filtered_out[name] = (
                        report.combinations_filtered_out.get(name, 0) + 1
                    )
            continue
        yield hand


# --- rendering rules ---

def render_rule_variables_to_omit(rules: list[DealRule], hand: dict[str, str]) -> frozenset[str]:
    """Variables whose planning-prompt lines this hand omits under the rendering rules."""
    omitted: set[str] = set()
    for rule in rules:
        if rule.kind == RENDER_KIND and rule.applies_to(hand):
            omitted.update(rule.omitted_variables)
    return frozenset(omitted)


def count_rendering_rule_hands(rules: list[DealRule], hands: list[dict[str, str]], report: RepairReport) -> None:
    for rule in rules:
        if rule.kind == RENDER_KIND:
            report.hands_affected_by_rendering_rule[rule.name] = sum(
                rule.applies_to(hand) for hand in hands
            )


def drop_template_lines(template: str, variable_names: Iterable[str]) -> str:
    """Remove every template line that holds a ``{variable}`` placeholder for the
    named variables. A line that stood alone as a paragraph takes one adjacent
    blank line with it, so no double gap is left behind. Everything else is
    returned byte-for-byte."""
    placeholders = tuple("{" + name + "}" for name in variable_names)
    if not placeholders:
        return template
    kept: list[str] = []
    lines = template.split("\n")
    skip_next_blank = False
    for line in lines:
        if any(placeholder in line for placeholder in placeholders):
            if kept and kept[-1] == "":
                skip_next_blank = True   # drop the blank after this line, not the one before
            continue
        if skip_next_blank and line == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        kept.append(line)
    return "\n".join(kept)
