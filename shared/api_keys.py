"""One place that decides where an API key comes from.

The repo must be usable by other people, including labs on servers where keys arrive as
environment variables, so which source is trusted is an explicit choice, made once per
process by the command-line flag added by ``add_api_key_source_flags``. Two modes:

1. Default (no flag), ``REPO_ROOT_DOTENV_ONLY``: the key comes only from
   ``<repo root>/.env``. The repo root is the directory that contains ``shared/``, found from
   this file's own location: never by searching parent folders and never from the current
   working directory. The operating system environment is never used. A missing file, or a
   missing or empty value in it, raises ``MissingApiKeyError`` naming the path and the key.
2. ``--find-dotenv``, ``SEARCH_UPWARD_THEN_ENVIRONMENT``: the nearest .env file found from the
   repo root upward is used (as python-dotenv's ``find_dotenv`` would), and if it does not set the
   key the operating system environment is used.

A value counts only if it is non-empty after stripping whitespace. A key read from a .env file
is written to ``os.environ`` so code that reads the environment keeps working. For an optional
key, an absent key returns None instead of raising. The source is printed once per variable per
process; the key value is never printed.
"""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent

REPO_ROOT_DOTENV_ONLY = "repo_root_dotenv_only"
SEARCH_UPWARD_THEN_ENVIRONMENT = "search_upward_then_environment"

FIND_DOTENV_FLAG = "--find-dotenv"

_api_key_source_mode = REPO_ROOT_DOTENV_ONLY
_sources_already_announced: set[str] = set()


class MissingApiKeyError(RuntimeError, KeyError):
    """No key for a variable in the source the current mode allows.

    Subclasses both RuntimeError and KeyError because callers have historically raised
    either for a missing key."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def set_api_key_source_mode(mode: str) -> None:
    """Choose where keys come from for the rest of this process (see the module docstring)."""
    global _api_key_source_mode
    if mode not in (REPO_ROOT_DOTENV_ONLY, SEARCH_UPWARD_THEN_ENVIRONMENT):
        raise ValueError(f"Unknown API key source mode: {mode!r}")
    _api_key_source_mode = mode


def add_api_key_source_flags(parser: argparse.ArgumentParser) -> None:
    """Add the ``--find-dotenv`` flag. Call ``apply_api_key_source_flags`` on the parsed
    arguments before anything loads a key."""
    parser.add_argument(FIND_DOTENV_FLAG, dest="api_key_source_mode", action="store_const",
                        const=SEARCH_UPWARD_THEN_ENVIRONMENT, default=REPO_ROOT_DOTENV_ONLY,
                        help="Search upward from the repo root for a .env (like python-dotenv's find_dotenv), "
                             "then fall back to the operating system environment. Default: read keys only from "
                             "the repo-root .env.")


def apply_api_key_source_flags(arguments: argparse.Namespace) -> None:
    set_api_key_source_mode(arguments.api_key_source_mode)


def api_key_source_flags_for_subprocess() -> list[str]:
    """The flag (if any) that makes a child Python process use the same key source as this one."""
    return [FIND_DOTENV_FLAG] if _api_key_source_mode == SEARCH_UPWARD_THEN_ENVIRONMENT else []


def repo_dotenv_path() -> Path:
    return (REPO_ROOT / ".env").resolve()


def _nearest_dotenv_path_searching_upward() -> Path | None:
    for directory in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _announce_once(variable_name: str, message: str) -> None:
    if variable_name in _sources_already_announced:
        return
    _sources_already_announced.add(variable_name)
    print(message)


def _value_in_dotenv_file(dotenv_path: Path | None, variable_name: str) -> str:
    if dotenv_path is None or not dotenv_path.is_file():
        return ""
    return (dotenv_values(dotenv_path).get(variable_name) or "").strip()


def _value_in_operating_system_environment(variable_name: str) -> str:
    return os.environ.get(variable_name, "").strip()


def load_optional_api_key(variable_name: str) -> str | None:
    """Return the key for ``variable_name`` from the current mode's source, or None if absent."""
    if _api_key_source_mode == SEARCH_UPWARD_THEN_ENVIRONMENT:
        dotenv_path = _nearest_dotenv_path_searching_upward()
    else:
        dotenv_path = repo_dotenv_path()
    value_in_file = _value_in_dotenv_file(dotenv_path, variable_name)
    if value_in_file:
        os.environ[variable_name] = value_in_file
        _announce_once(variable_name, f"{variable_name} from {dotenv_path} (value not printed)")
        return value_in_file

    if _api_key_source_mode == SEARCH_UPWARD_THEN_ENVIRONMENT:
        value_in_environment = _value_in_operating_system_environment(variable_name)
        if value_in_environment:
            if variable_name not in _sources_already_announced:
                if dotenv_path is not None:
                    print(f"{dotenv_path} has no {variable_name}; using the key from the environment")
                print(f"{variable_name} from the environment")
                _sources_already_announced.add(variable_name)
            return value_in_environment
    return None


def hide_environment_value_unless_environment_is_allowed(variable_name: str) -> None:
    """For a library that reads ``variable_name`` from the operating system environment by itself: in the
    default mode remove it, so the library cannot use a key this module would never have used. Call after
    ``load_optional_api_key`` (which puts a key found in the .env file into the environment)."""
    if _api_key_source_mode == REPO_ROOT_DOTENV_ONLY:
        os.environ.pop(variable_name, None)


def load_api_key(variable_name: str) -> str:
    """Return the key for ``variable_name``; raise MissingApiKeyError if the current mode finds none."""
    key = load_optional_api_key(variable_name)
    if key is not None:
        return key
    if _api_key_source_mode == SEARCH_UPWARD_THEN_ENVIRONMENT:
        raise MissingApiKeyError(
            f"{variable_name} not found: no .env searched upward from {REPO_ROOT} sets it, and it is not set "
            f"in the operating system environment. Fill it in {repo_dotenv_path()} (see .env.example in the "
            f"repo root) or export it.")
    dotenv_path = repo_dotenv_path()
    problem = (f"{dotenv_path} does not exist" if not dotenv_path.is_file()
               else f"{dotenv_path} has no non-empty {variable_name}")
    raise MissingApiKeyError(
        f"{variable_name} not found: {problem}. Fill in {variable_name} in {dotenv_path} "
        f"(see .env.example in the repo root), or pass {FIND_DOTENV_FLAG} to search parent folders for a .env "
        f"and then fall back to the operating system environment.")

