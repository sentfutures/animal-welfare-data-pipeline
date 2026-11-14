"""SDF model settings that were renamed by decision 17 (the ``<step name>_model`` rule), still accepted under
their old names.

``sdf.select_model`` is now ``sdf.consideration_scoring_model`` and ``sdf.rewrite_model`` is now
``sdf.revise_model`` ("rewrite" also names the description rewrite). A config that sets only the old key keeps
working, and one line names the new key; when both are set the new key wins and the line says the old one is
ignored. Every place that reads either setting goes through ``model_setting``.
"""

OLD_MODEL_SETTING_NAMES = {"consideration_scoring_model": "select_model", "revise_model": "rewrite_model"}

_keys_already_announced: set = set()


def model_setting(sdf: dict, key: str):
    """``sdf[key]``, else (for a renamed setting) the value under its old name, with a one-line notice."""
    value = sdf.get(key)
    old_key = OLD_MODEL_SETTING_NAMES.get(key)
    if old_key is None or not sdf.get(old_key):
        return value
    if (key, old_key) not in _keys_already_announced:
        _keys_already_announced.add((key, old_key))
        if value:
            print(f"Notice: sdf.{old_key} is the old name of sdf.{key}; both are set, so sdf.{key} is used "
                  f"and sdf.{old_key} is ignored. Remove sdf.{old_key}.")
        else:
            print(f"Notice: sdf.{old_key} is the old name of sdf.{key}; it still works, but rename it to sdf.{key}.")
    return value or sdf[old_key]
