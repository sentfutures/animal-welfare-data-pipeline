"""SDF content profiles.

A profile is everything domain-specific the SDF pipeline needs beyond the
files under prompts/<PROMPTS_SUBDIR>/: the per-category example pools drawn
into {sentient_example}, plus where its prompts, principles, failure modes
and runs live. The pipeline code is shared; only the profile changes.

One profile is active per process, chosen by config `sdf.profile` (default
"animal") and activated by sdf_pipeline/run.py before layer 1. Code reads it
at call time through current(). The animal profile is the original pipeline:
its example pools were moved here verbatim, so an animal run is unchanged.
"""
from importlib import import_module
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = "animal"
PROFILES = {
    "animal": "sdf_pipeline.profiles.animal",
    "human": "sdf_pipeline.profiles.human",
}

_current: ModuleType | None = None


def get(name: str) -> ModuleType:
    if name not in PROFILES:
        raise ValueError(f"unknown SDF profile {name!r}; known: {sorted(PROFILES)}")
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
    return (config.get("sdf") or {}).get("profile") or DEFAULT


def principles_filenames(profile: ModuleType | None = None) -> tuple[str, ...]:
    """The profile's principles file, then any older names a run snapshotted
    before it took that file, newest first."""
    profile = profile or current()
    return (profile.PRINCIPLES_FILENAME, *getattr(profile, "LEGACY_PRINCIPLES_FILENAMES", ()))


def prompts_dir(profile: ModuleType | None = None) -> Path:
    """The repo's live prompt folder for a profile (runs read their snapshot)."""
    return REPO_ROOT / "prompts" / (profile or current()).PROMPTS_SUBDIR
