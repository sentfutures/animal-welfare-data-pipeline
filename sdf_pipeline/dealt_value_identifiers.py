"""Which value id each dealt value has, kept apart from the text that goes in prompts.

``variables.txt`` gives every value a stable ``[id]`` that is stripped before rendering, so a hand
records only the text. This module looks the ids up again, in two directions:

* ``value_ids_of_dealt_hand`` is used by the deal as it deals: the ids come from the live
  ``{axis: {id: text}}`` map (``compose_prompts.load_variables``), and are written to the
  ``round_dealt`` event only. No stage file, request or final corpus record carries them.
* ``value_ids_of_dealt_hand_from_snapshot`` is used by the loader for runs dealt before the event
  carried ids: it looks each dealt text up in the run's frozen ``inputs/prompts/variables.txt``.

Both return ``(ids_by_axis, notes)``. An axis whose value cannot be matched has no entry in
``ids_by_axis`` and one sentence in ``notes``. Nothing that builds a prompt reads either.
"""

from pathlib import Path

from shared import matrix as _matrix

FROZEN_VARIABLES_RELATIVE_PATH = Path("inputs") / "prompts" / "variables.txt"


def value_ids_of_dealt_hand(dealt_variables: dict, value_ids_by_axis: dict) -> tuple[dict, list]:
    """``({axis: value id}, notes)`` for one hand's ``{axis: text}``, matched against
    ``{axis: {id: text}}``. A text no id has, or that two ids share, gets no id and a note."""
    ids_by_axis: dict = {}
    notes: list = []
    for axis, text in (dealt_variables or {}).items():
        ids_of_text = [value_id for value_id, candidate in (value_ids_by_axis.get(axis) or {}).items()
                       if candidate == text]
        if len(ids_of_text) == 1:
            ids_by_axis[axis] = ids_of_text[0]
        elif not ids_of_text:
            notes.append(f"dealt value of {{{axis}}} has no id: no value of that axis has this text "
                         f"({text[:50]!r})")
        else:
            notes.append(f"dealt value of {{{axis}}} has no id: {len(ids_of_text)} values share this text "
                         f"({text[:50]!r})")
    return ids_by_axis, notes


def load_frozen_value_ids(run_directory) -> tuple[dict | None, str | None]:
    """``({axis: {id: text}}, None)`` from the run's frozen ``inputs/prompts/variables.txt``, or
    ``(None, reason)`` when the run has no snapshot or its snapshot cannot be read."""
    path = Path(run_directory) / FROZEN_VARIABLES_RELATIVE_PATH
    if not path.is_file():
        return None, f"the run has no frozen {FROZEN_VARIABLES_RELATIVE_PATH.as_posix()}"
    try:
        parsed, ids = _matrix.parse_variables_with_ids(path)
    except ValueError as error:
        return None, f"the frozen {FROZEN_VARIABLES_RELATIVE_PATH.as_posix()} cannot be read ({error})"
    value_ids = _matrix.value_ids_by_variable(parsed, ids, require_ids=False)
    if not any(value_ids.values()):
        return None, (f"the frozen {FROZEN_VARIABLES_RELATIVE_PATH.as_posix()} predates value ids "
                      "(no value carries an [id] prefix)")
    return value_ids, None


def value_ids_of_dealt_hand_from_snapshot(dealt_variables: dict, frozen_value_ids: dict | None,
                                          why_no_snapshot: str | None) -> tuple[dict, list]:
    """The loader's version: the same lookup against the frozen snapshot, or no ids and one note
    saying why when there is no usable snapshot."""
    if not dealt_variables:
        return {}, []
    if frozen_value_ids is None:
        return {}, [f"dealt values have no ids: {why_no_snapshot}"]
    return value_ids_of_dealt_hand(dealt_variables, frozen_value_ids)
