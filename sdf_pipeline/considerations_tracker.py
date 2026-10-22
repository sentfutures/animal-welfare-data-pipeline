"""The run-level considerations tracker: pure code, no API calls, no files.

It owns every rule about WHICH considerations may be offered to a document and
whether a pick may be accepted; the selector stage (layer1_select_considerations)
only asks the model and feeds the answers in.

- Priority: an entry's position in its list (considerations.txt order) is its
  priority; the digital-minds-only entries sit first on the epistemic list.
- Eligibility: a digital-minds-only entry is offered only on digital-mind
  documents, and its target is a share of those documents; every other entry is
  offered on every eligible document, with a target that is a share of all of them.
- Hard cap: an entry is removed from the lists once its count reaches its
  target, and never offered (or accepted) again.
- Waves: the selector asks several documents at once. Each wave sees the lists
  as they stood after the previous wave; the wave's picks are then accepted in
  document order, and a pick that would push its entry past target is rejected
  ("over_cap") and left unresolved, so that document is re-asked for that list
  in the next wave against the updated lists.
- Passes: pass 1 asks every eligible document for each list. A later pass
  re-asks every eligible document, offering only entries still under target that
  the document does not already carry, so a document can end up with several
  considerations.
- Rebuild: every answered call is one row (``record_answer`` returns it);
  ``replay`` applies saved rows, so a resumed run starts from exactly the state
  the paid work had reached.
"""

from dataclasses import dataclass

from sdf_pipeline.considerations_catalog import (
    LIST_NAMES, MORAL_LIST, ConsiderationCatalog, Consideration, target_count,
)

STATUS_ACCEPTED = "accepted"
STATUS_NONE = "none"
STATUS_OVER_CAP = "over_cap"


@dataclass(frozen=True)
class Pick:
    prompt_id: str
    list_name: str
    label: str
    reason: str
    pass_number: int
    wave_number: int


@dataclass(frozen=True)
class PendingCall:
    prompt_id: str
    list_names: tuple[str, ...]   # the lists still unresolved for this document in this pass


class ConsiderationTracker:
    def __init__(self, catalog: ConsiderationCatalog, eligible_documents: list[tuple[str, bool]]):
        """``eligible_documents``: (prompt_id, is_digital_mind_document) in the
        order the selector should process them (the seeded shuffle)."""
        self.catalog = catalog
        self.document_order = [prompt_id for prompt_id, _ in eligible_documents]
        self._digital = {prompt_id: is_digital for prompt_id, is_digital in eligible_documents}
        self.eligible_count = len(self._digital)
        self.digital_count = sum(self._digital.values())
        self.targets: dict[str, int] = {}
        for entry in catalog.all_entries:
            denominator = self.digital_count if entry.digital_minds_only else self.eligible_count
            self.targets[entry.label] = target_count(entry.target_share, denominator)
        self.counts: dict[str, int] = {entry.label: 0 for entry in catalog.all_entries}
        self.over_cap_counts: dict[str, int] = {entry.label: 0 for entry in catalog.all_entries}
        self.picks: list[Pick] = []
        self._carried: dict[str, set[str]] = {prompt_id: set() for prompt_id in self._digital}
        self._resolved: set[tuple[int, str, str]] = set()      # (pass, prompt_id, list_name)
        self._over_cap_asks: set[tuple[int, str]] = set()      # (pass, prompt_id) re-ask first
        self._last_wave: dict[int, int] = {}                   # pass -> highest wave seen

    # ------------------------------------------------------------ the lists

    def is_open(self, entry: Consideration) -> bool:
        return self.counts[entry.label] < self.targets[entry.label]

    def all_targets_met(self) -> bool:
        return not any(self.is_open(entry) for entry in self.catalog.all_entries)

    def labels_carried_by(self, prompt_id: str) -> set[str]:
        return set(self._carried[prompt_id])

    def picks_for(self, prompt_id: str) -> list[Pick]:
        return [pick for pick in self.picks if pick.prompt_id == prompt_id]

    def offered_entries(self, prompt_id: str, list_name: str) -> list[Consideration]:
        """The list as this document sees it now: priority order, eligible,
        still under target, and not already carried by the document."""
        carried = self._carried[prompt_id]
        is_digital = self._digital[prompt_id]
        return [
            entry for entry in self.catalog.entries_of(list_name)
            if self.is_open(entry)
            and entry.label not in carried
            and (is_digital or not entry.digital_minds_only)
        ]

    # ------------------------------------------------------------ scheduling

    def _pending_call_for(self, prompt_id: str, pass_number: int) -> PendingCall | None:
        list_names = tuple(
            name for name in LIST_NAMES
            if (pass_number, prompt_id, name) not in self._resolved
            and self.offered_entries(prompt_id, name)
        )
        return PendingCall(prompt_id, list_names) if list_names else None

    def pending_calls(
        self, pass_number: int, skip: frozenset[str] = frozenset(), limit: int | None = None
    ) -> list[PendingCall]:
        """Calls still to make in this pass, in order: documents whose pick was
        rejected over cap come first (they are re-asked in the next wave), then
        documents not yet asked, both in document order. A document is asked only
        for lists still unresolved in this pass that have something to offer."""
        re_asked = [p for p in self.document_order if (pass_number, p) in self._over_cap_asks]
        fresh = [p for p in self.document_order if (pass_number, p) not in self._over_cap_asks]
        calls: list[PendingCall] = []
        for prompt_id in re_asked + fresh:
            if prompt_id in skip:
                continue
            call = self._pending_call_for(prompt_id, pass_number)
            if call is not None:
                calls.append(call)
                if limit is not None and len(calls) >= limit:
                    break
        return calls

    def next_wave_calls(
        self, pass_number: int, wave_size: int, skip: frozenset[str] = frozenset()
    ) -> list[PendingCall]:
        return self.pending_calls(pass_number, skip, limit=max(1, wave_size))

    def next_wave_number(self, pass_number: int) -> int:
        return self._last_wave.get(pass_number, 0) + 1

    # ------------------------------------------------------------ accepting answers

    def record_answer(
        self,
        prompt_id: str,
        pass_number: int,
        wave_number: int,
        choices: dict[str, tuple[str | None, str]],
    ) -> dict:
        """Apply one document's answer (``{list_name: (label or None, reason)}``) and
        return the row to persist. Call it for a wave's documents in document order:
        that is what makes acceptance honest against the cap. A label that is not on
        the document's current offer raises ValueError, except that an entry the
        wave itself has just filled is rejected over cap, not an error."""
        outcomes = []
        for list_name, (label, reason) in choices.items():
            if label is None:
                status = STATUS_NONE
            else:
                entry = self.catalog.entry_by_label(label)
                if entry.list_name != list_name:
                    raise ValueError(f"{label!r} is not on the {list_name} list")
                if entry.digital_minds_only and not self._digital[prompt_id]:
                    raise ValueError(f"{label!r} is not eligible for {prompt_id}")
                if label in self.labels_carried_by(prompt_id):
                    raise ValueError(f"{prompt_id} already carries {label!r}")
                status = STATUS_ACCEPTED if self.is_open(entry) else STATUS_OVER_CAP
            outcomes.append({"list": list_name, "label": label, "reason": reason, "status": status})
        row = {"prompt_id": prompt_id, "pass": pass_number, "wave": wave_number, "outcomes": outcomes}
        self._apply(row)
        return row

    def replay(self, rows: list[dict]) -> None:
        """Rebuild state from saved rows, trusting their recorded statuses. Rows for
        documents that are no longer eligible (dropped by an earlier stage on a later run) are ignored."""
        for row in rows:
            if row["prompt_id"] in self._digital:
                self._apply(row)

    def _apply(self, row: dict) -> None:
        prompt_id, pass_number, wave_number = row["prompt_id"], row["pass"], row["wave"]
        self._last_wave[pass_number] = max(self._last_wave.get(pass_number, 0), wave_number)
        for outcome in row["outcomes"]:
            list_name, label, status = outcome["list"], outcome["label"], outcome["status"]
            if status == STATUS_OVER_CAP:
                self.over_cap_counts[label] += 1
                self._over_cap_asks.add((pass_number, prompt_id))
                continue
            self._resolved.add((pass_number, prompt_id, list_name))
            if status == STATUS_ACCEPTED:
                self.counts[label] += 1
                self._carried[prompt_id].add(label)
                self.picks.append(Pick(prompt_id, list_name, label, outcome["reason"],
                                       pass_number, wave_number))

    def accepted_in_pass(self, pass_number: int) -> int:
        return sum(pick.pass_number == pass_number for pick in self.picks)

    # ------------------------------------------------------------ reporting

    def coverage_report(self) -> dict:
        """Per entry: target, achieved, how many came from pass 2 or later and how
        many picks were rejected over cap; plus how documents ended up covered."""
        entries = []
        for entry in self.catalog.all_entries:
            denominator = self.digital_count if entry.digital_minds_only else self.eligible_count
            entries.append({
                "label": entry.label,
                "list": entry.list_name,
                "digital_minds_only": entry.digital_minds_only,
                "target_share": entry.target_share,
                "eligible_documents": denominator,
                "target_count": self.targets[entry.label],
                "achieved": self.counts[entry.label],
                "from_later_passes": sum(
                    pick.label == entry.label and pick.pass_number >= 2 for pick in self.picks),
                "rejected_over_cap": self.over_cap_counts[entry.label],
            })
        per_document = {prompt_id: {name: 0 for name in LIST_NAMES} for prompt_id in self.document_order}
        for pick in self.picks:
            per_document[pick.prompt_id][pick.list_name] += 1
        return {
            "eligible_documents": self.eligible_count,
            "digital_mind_documents": self.digital_count,
            "entries": entries,
            **document_coverage_shares(list(per_document.values())),
        }


def document_coverage_shares(per_document_counts: list[dict[str, int]]) -> dict:
    """How documents are covered, from each document's pick count per list:
    shares with no consideration from either list, with picks from only one list,
    and from both, and the number carrying two or more picks from the same list."""
    total = len(per_document_counts)
    none = only_one = both = multiple_in_a_list = 0
    for counts in per_document_counts:
        lists_used = sum(counts[name] > 0 for name in LIST_NAMES)
        none += lists_used == 0
        only_one += lists_used == 1
        both += lists_used == 2
        multiple_in_a_list += any(counts[name] >= 2 for name in LIST_NAMES)
    share = (lambda n: round(n / total, 4)) if total else (lambda n: None)
    return {
        "documents_with_no_consideration": none,
        "documents_with_only_one_list": only_one,
        "documents_with_both_lists": both,
        "share_with_no_consideration": share(none),
        "share_with_only_one_list": share(only_one),
        "share_with_both_lists": share(both),
        "documents_with_two_or_more_in_one_list": multiple_in_a_list,
    }


def corpus_coverage(catalog: ConsiderationCatalog, variables_list: list[dict]) -> dict:
    """The same report measured on finished documents' ``variables`` (the final
    corpus), where drafting failures and the score gate have shifted the shares
    since selection. Documents with a skipped resolution are left out; there is
    no pass or rejection data here, so those columns are absent."""
    documents = [v for v in variables_list if not catalog.is_skipped_resolution(v)]
    digital = [v for v in documents if catalog.is_digital_mind_document(v)]
    entries = []
    for entry in catalog.all_entries:
        pool = digital if entry.digital_minds_only else documents
        field = "moral_considerations" if entry.list_name == MORAL_LIST else "epistemic_considerations"
        achieved = sum(entry.label in (v.get(field) or []) for v in pool)
        entries.append({
            "label": entry.label, "list": entry.list_name,
            "digital_minds_only": entry.digital_minds_only,
            "target_share": entry.target_share, "eligible_documents": len(pool),
            "achieved": achieved,
            "achieved_share": round(achieved / len(pool), 4) if pool else None,
        })
    per_document = [
        {MORAL_LIST: len(v.get("moral_considerations") or []),
         "epistemic": len(v.get("epistemic_considerations") or [])}
        for v in documents
    ]
    return {
        "eligible_documents": len(documents),
        "digital_mind_documents": len(digital),
        "entries": entries,
        **document_coverage_shares(per_document),
    }
