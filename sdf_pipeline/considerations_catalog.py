"""The considerations catalog: parsing prompts/sdf/considerations.txt.

Pure code, no API calls. The file has the same format as variables.txt (and is
parsed by the same ``parse_variables``), with one difference: the ``<weight> ::``
prefix on an entry is its TARGET SHARE of the eligible documents, so the shares
of a list need not sum to 1.

Sections (see the header of considerations.txt for the full contract):
``moral_considerations`` and ``epistemic_considerations`` are the two ordered
lists; ``epistemic_considerations_digital_minds_only`` entries are placed first
on the epistemic list and are eligible only on documents whose sentient_category
is one of ``digital_mind_categories``; ``skipped_resolutions`` names the
resolution values whose documents get no considerations. The category and
resolution values are named by a substring that must match exactly one value in
variables.txt (the matcher shared with deal_constraints.txt), so rewording a
value there makes loading fail instead of silently making an entry unreachable.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline import compose_prompts as cp
from shared import deal_constraints

CONSIDERATIONS_FILENAME = "considerations.txt"
VARIABLES_FILENAME = "variables.txt"

MORAL_LIST = "moral"
EPISTEMIC_LIST = "epistemic"
LIST_NAMES = (MORAL_LIST, EPISTEMIC_LIST)

LABEL_GLOSS_SEPARATOR = ": "

SECTION_MORAL = "moral_considerations"
SECTION_EPISTEMIC = "epistemic_considerations"
SECTION_DIGITAL_MINDS_ONLY = "epistemic_considerations_digital_minds_only"
SECTION_DIGITAL_MIND_CATEGORIES = "digital_mind_categories"
SECTION_SKIPPED_RESOLUTIONS = "skipped_resolutions"
_REQUIRED_SECTIONS = (
    SECTION_MORAL, SECTION_EPISTEMIC, SECTION_DIGITAL_MINDS_ONLY,
    SECTION_DIGITAL_MIND_CATEGORIES, SECTION_SKIPPED_RESOLUTIONS,
)


@dataclass(frozen=True)
class Consideration:
    label: str
    gloss: str
    target_share: float
    list_name: str                  # MORAL_LIST or EPISTEMIC_LIST
    digital_minds_only: bool = False

    @property
    def display(self) -> str:
        return f"{self.label}{LABEL_GLOSS_SEPARATOR}{self.gloss}"


@dataclass(frozen=True)
class ConsiderationCatalog:
    moral: tuple[Consideration, ...]
    epistemic: tuple[Consideration, ...]      # digital-minds-only entries first
    digital_mind_categories: frozenset[str]   # full sentient_category values
    skipped_resolutions: frozenset[str]       # full resolution values

    @property
    def all_entries(self) -> tuple[Consideration, ...]:
        return self.moral + self.epistemic

    def entries_of(self, list_name: str) -> tuple[Consideration, ...]:
        return self.moral if list_name == MORAL_LIST else self.epistemic

    def entry_by_label(self, label: str) -> Consideration:
        for entry in self.all_entries:
            if entry.label == label:
                return entry
        raise KeyError(label)

    def is_digital_mind_document(self, variables: dict) -> bool:
        return variables.get("sentient_category") in self.digital_mind_categories

    def is_skipped_resolution(self, variables: dict) -> bool:
        return variables.get("resolution") in self.skipped_resolutions


def target_count(target_share: float, eligible_documents: int) -> int:
    """Documents an entry may reach: the share of the eligible documents, rounded
    half up and never below one (so every entry can be offered in a small run)."""
    if eligible_documents <= 0:
        return 0
    return max(1, math.floor(round(target_share * eligible_documents, 9) + 0.5))


def _split_entry(path_name: str, section: str, text: str) -> tuple[str, str]:
    label, separator, gloss = text.partition(LABEL_GLOSS_SEPARATOR)
    if not separator or not label.strip() or not gloss.strip():
        raise ValueError(
            f"{path_name}: {{{section}}} entry {text[:60]!r} must read "
            f"'Label{LABEL_GLOSS_SEPARATOR}one-line gloss'"
        )
    return label.strip(), gloss.strip()


def _parse_entries(
    path_name: str, section: str, pairs: list[tuple[str, float | None]],
    list_name: str, digital_minds_only: bool,
) -> list[Consideration]:
    entries = []
    for text, share in pairs:
        label, gloss = _split_entry(path_name, section, text)
        if share is None or not 0 < share <= 1:
            raise ValueError(
                f"{path_name}: {{{section}}} entry {label!r} needs a target share "
                f"'<share> ::' with 0 < share <= 1"
            )
        entries.append(Consideration(label, gloss, share, list_name, digital_minds_only))
    return entries


def load_considerations(prompts_dir: Path) -> ConsiderationCatalog:
    """Parse considerations.txt, resolving its category and resolution names
    against the variables.txt beside it. Raises ValueError on any malformed
    entry, duplicate label, missing section, or name that does not match exactly
    one variables.txt value."""
    prompts_dir = Path(prompts_dir)
    path = prompts_dir / CONSIDERATIONS_FILENAME
    parsed = cp.parse_variables(path)
    missing = [name for name in _REQUIRED_SECTIONS if name not in parsed]
    if missing:
        raise ValueError(f"{path.name}: missing section(s): " + ", ".join(f"{{{n}}}" for n in missing))
    unknown = [name for name in parsed if name not in _REQUIRED_SECTIONS]
    if unknown:
        raise ValueError(f"{path.name}: unknown section(s): " + ", ".join(f"{{{n}}}" for n in unknown))

    moral = _parse_entries(path.name, SECTION_MORAL, parsed[SECTION_MORAL], MORAL_LIST, False)
    epistemic_digital = _parse_entries(
        path.name, SECTION_DIGITAL_MINDS_ONLY, parsed[SECTION_DIGITAL_MINDS_ONLY], EPISTEMIC_LIST, True)
    epistemic_general = _parse_entries(
        path.name, SECTION_EPISTEMIC, parsed[SECTION_EPISTEMIC], EPISTEMIC_LIST, False)
    epistemic = epistemic_digital + epistemic_general
    if not moral or not epistemic:
        raise ValueError(f"{path.name}: both the moral and the epistemic list need at least one entry")

    labels = [entry.label for entry in moral + epistemic]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: labels must be unique across both lists, repeated: {duplicates}")

    values, _ = cp.split_weights(cp.parse_variables(prompts_dir / VARIABLES_FILENAME))

    def resolve(section: str, variable: str) -> frozenset[str]:
        substrings = [text for text, _ in parsed[section]]
        return frozenset(
            deal_constraints._resolve_substring(section, variable, substring, values, path.name)
            for substring in substrings
        )

    return ConsiderationCatalog(
        moral=tuple(moral),
        epistemic=tuple(epistemic),
        digital_mind_categories=resolve(SECTION_DIGITAL_MIND_CATEGORIES, "sentient_category"),
        skipped_resolutions=resolve(SECTION_SKIPPED_RESOLUTIONS, "resolution"),
    )
