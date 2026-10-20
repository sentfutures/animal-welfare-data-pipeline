"""DAD content profiles.

A profile is everything domain-specific the DAD pipeline needs beyond the
files under prompts/<PROMPTS_SUBDIR>/: the affected-party table (TAXA), the
archetypes, the thin-domain watch list, labels, the scope-axis wording and the
batch checklist's manual lines, plus where its prompts, principles and runs
live. The pipeline code is shared; only the profile changes.

One profile is active per process, chosen by config `dad.profile` (default
"animal") and activated by dad_pipeline/run.py before step 1. Code reads it
at call time through current(). The animal profile is the original pipeline:
its tables were moved here verbatim, so an animal run is unchanged.
"""
from importlib import import_module
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = "animal"
PROFILES = {
    "animal": "dad_pipeline.profiles.animal",
}

_current: ModuleType | None = None


def get(name: str) -> ModuleType:
    if name not in PROFILES:
        raise ValueError(f"unknown DAD profile {name!r}; known: {sorted(PROFILES)}")
    return import_module(PROFILES[name])


def use(name: str) -> ModuleType:
    """Activate a profile for this process and return it."""
    global _current
    _current = get(name)
    return _current


def current() -> ModuleType:
    return _current or get(DEFAULT)


def reset() -> None:
    global _current
    _current = None


def name_from_config(config: dict) -> str:
    return (config.get("dad") or {}).get("profile") or DEFAULT


def prompts_dir(profile: ModuleType | None = None) -> Path:
    """The repo's live prompt folder for a profile (runs read their snapshot)."""
    return REPO_ROOT / "prompts" / (profile or current()).PROMPTS_SUBDIR
