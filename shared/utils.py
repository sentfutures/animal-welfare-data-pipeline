"""Shared utilities: JSONL I/O, checkpointing, prompt loading, run scoping."""

import json
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, UTC

import yaml

from shared import sdf_run_layouts


def parallel_map(fn, items: list, workers: int):
    """Map fn over items with a thread pool, yielding in input order; fn must be side-effect free so callers keep file writes on their own thread."""
    if workers <= 1 or len(items) <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(fn, items)


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_relative(path: str | Path) -> str:
    """A path relative to the repo root, for values written to reports and manifests (an absolute path would bake in the machine); outside the repo only the name is kept."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- SDF layout ----
# A run's stage folders are named by its layout (``shared/sdf_run_layouts.py``) and found by folder KEY, never by trying a name and falling back;
# the layouts are told apart by folder names (the pre-renumber and four-layer ones overlap on ``layer3``/``layer4``; ``layer12`` separates them).

# stage key -> (folder key, output filename)
_SDF_STAGE_FILES = {"dealt": (sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, "prompts.jsonl"),
                    "plan": (sdf_run_layouts.DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, "plans.jsonl"),
                    "draft": (sdf_run_layouts.DRAFT, "drafts.jsonl"),
                    "rewrite": (sdf_run_layouts.CRITIQUE_AND_REVISE, "rewrites.jsonl"),
                    "score": (sdf_run_layouts.SCORE, "scores.jsonl")}


def sdf_layout_of_run(run_dir: str | Path) -> sdf_run_layouts.SdfRunLayout:
    """The layout of a run folder: told by its folder names, else its manifest's record, else (a folder
    with neither: a run not created yet) the five-layer layout every new run uses."""
    run = Path(run_dir)
    layout = sdf_run_layouts.layout_of_run_folder(run)
    if layout is None and (run / "run_manifest.json").is_file():
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        layout = sdf_run_layouts.layout_of_run_folder(run, manifest)
    return layout or sdf_run_layouts.FIVE_NAMED_LAYERS


def sdf_stage_folder(run_dir: str | Path, folder_key: str) -> Path:
    """The folder of one kind of SDF stage file (a ``sdf_run_layouts`` folder key) in the run's own layout,
    for reading and writing."""
    return sdf_layout_of_run(run_dir).folder_of(run_dir, folder_key)


def sdf_stage_dir(run_dir: str | Path, layer: int) -> Path:
    """The folder of ``--layer`` number ``layer`` of the run, in the run's own layout (1 to 5 in the
    five-layer layout, 1 to 4 in the four-layer one). Where the layout has two folders for a layer, the first."""
    layout = sdf_layout_of_run(run_dir)
    if layer not in layout.layer_numbers:
        raise KeyError(layer)
    return layout.folder_of(run_dir, layout.folder_key_of_layer(layer))


def sdf_stage_dir_of_existing_run(run_dir: str | Path, layer: int) -> Path:
    """``sdf_stage_dir`` for reading a run in any layout."""
    return sdf_stage_dir(run_dir, layer)


_SDF_TEMPLATES = {1: "layer1.txt", 2: "layer2.txt", 4: "layer4.txt"}


def sdf_template_path(prompts_dir: str | Path, layer: int) -> Path:
    """One layer's prompt template (a run's frozen snapshot or ``prompts/sdf``). The template file names do not
    change with the layers; ``layer`` is the number in their names (1 deal, 2 draft, 4 score; layer 3's templates are named by the critique and revise steps)."""
    return Path(prompts_dir) / _SDF_TEMPLATES[layer]


def sdf_stage_file(run_dir: str | Path, stage: str) -> Path:
    """Path to one SDF stage's output jsonl in the run's layout (read-only); ``stage`` is dealt/plan/draft/rewrite/score, named because numbers differ between layouts."""
    folder_key, filename = _SDF_STAGE_FILES[stage]
    return sdf_stage_folder(run_dir, folder_key) / filename


def save_jsonl(data: list[dict], path: str | Path, append: bool = False) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_json(text: str):
    """Parse the JSON value in a model response, tolerating surrounding chatter: the longest value that parses from any ``[``/``{``.
    Raises JSONDecodeError when none parses, and when the payload broke partway (its fragments are disqualified, never returned); ``strict=False`` tolerates raw control characters."""
    decoder = json.JSONDecoder(strict=False)
    candidates = []  # (start, end, value)
    failures = []  # (start, position where the parse gave up)
    for match in re.finditer(r"[\[{]", text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as err:
            failures.append((match.start(), err.pos))
            continue
        candidates.append((match.start(), end, value))

    broken = [q for q, p in failures
              if any(q < s and e <= p for s, e, _ in candidates)]
    eligible = [c for c in candidates
                if not any(c[0] > q for q in broken)]
    if eligible:
        return max(eligible, key=lambda c: c[1] - c[0])[2]
    if broken:
        raise json.JSONDecodeError(
            "JSON container is malformed or truncated", text, min(broken)
        )
    raise json.JSONDecodeError("no JSON value found in response", text, 0)


def salvage_json_objects(text: str) -> list:
    """Complete top-level ``{...}`` objects by brace matching, for a truncated or garbage-trailed stream (the ``recover=True`` last resort)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1], strict=False))
                except json.JSONDecodeError:
                    pass
                start = None
    return objs


def extract_json_object(text: str, recover: bool = False) -> dict:
    """``extract_json`` narrowed to an object (a wrong shape raises JSONDecodeError); ``recover=True`` takes the first complete object via ``salvage_json_objects``."""
    try:
        value = extract_json(text)
    except json.JSONDecodeError:
        if recover and (objs := salvage_json_objects(text)):
            return objs[0]
        raise
    if isinstance(value, dict):
        return value
    if recover and (objs := salvage_json_objects(text)):
        return objs[0]
    raise json.JSONDecodeError("JSON value is not an object", text, 0)


def extract_json_array(text: str, recover: bool = False) -> list:
    """``extract_json`` narrowed to an array (a wrong shape raises JSONDecodeError); ``recover=True`` unwraps a single-key object holding the array and salvages complete objects from a broken container."""
    try:
        value = extract_json(text)
    except json.JSONDecodeError:
        if recover and (objs := salvage_json_objects(text)):
            return objs
        raise
    if isinstance(value, list):
        return value
    if recover and isinstance(value, dict):
        # unwrap a single array-valued entry ({"reasons": [...]}); a genuinely
        # ambiguous object (0 or >1 list values) still raises, so we never guess.
        lists = [v for v in value.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
        # A brackets-dropped object STREAM ('{...},\n{...},...' — no opening
        # '[', sometimes no commas): extract_json parses ONE of those objects,
        # so the except-branch salvage above never runs; salvage here reads the whole stream.
        if (objs := salvage_json_objects(text)) and len(objs) > 1:
            return objs
    raise json.JSONDecodeError("JSON value is not an array", text, 0)


def load_prompt(path: str | Path, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if kwargs:
        text = text.format(**kwargs)
    return text


def looks_like_transcript_echo(text: str) -> bool:
    """True when a generation starts with a role marker ("USER:"), i.e. a transcript replay; callers skip it without checkpointing so ``--resume`` retries."""
    return bool(re.match(r"\s*(USER|ASSISTANT|HUMAN)\s*:", text[:40]))


# A template that carries both a system and a user half separates them with a
# line equal to this marker. See load_split_prompt.
_PROMPT_SPLIT_MARKER = "===USER==="


def load_split_prompt(path: str | Path, **kwargs) -> tuple[str, str]:
    """Load a two-part template as (system, user), split on a line equal to ``===USER===``; no marker gives ("", whole formatted template)."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    system_part, user_part = "", text
    for i, line in enumerate(lines):
        if line.strip() == _PROMPT_SPLIT_MARKER:
            system_part = "\n".join(lines[:i])
            user_part = "\n".join(lines[i + 1:])
            break

    def _fmt(part: str) -> str:
        return part.format(**kwargs) if kwargs else part

    return _fmt(system_part).strip(), _fmt(user_part).strip()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_language(distribution: dict[str, float], rng: random.Random | None = None) -> str:
    chooser = rng or random
    languages = list(distribution.keys())
    weights = list(distribution.values())
    return chooser.choices(languages, weights=weights, k=1)[0]


def new_run_id(label: str) -> str:
    """Mint a run ID: timestamp (to the minute) + sanitized label suffix."""
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "-", label.strip())
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{safe_label}"


def _git_status() -> tuple[str | None, str | None, bool, list[str]]:
    """``(short_commit, branch, dirty, dirty_files)``, or ``(None, None, False, [])`` outside git; branch is "HEAD" on a detached checkout."""
    cwd = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", check=True, cwd=cwd,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", check=True, cwd=cwd,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", check=True, cwd=cwd,
        ).stdout
        dirty_files = [line[3:].strip() for line in porcelain.splitlines() if line.strip()]
        return commit, branch, bool(dirty_files), dirty_files
    except Exception:
        return None, None, False, []


MAIN_REF = "origin/main"
# git fetch talks to GitHub; a hung network must not wedge a publish.
_FETCH_TIMEOUT_S = 20


def _git(*args: str, cwd: Path | None = None,
         timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a git command in the repo; a non-zero returncode is the caller's to inspect. ``timeout=`` can raise ``subprocess.TimeoutExpired``, which every such call site must handle."""
    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True, encoding="utf-8",
        cwd=cwd or Path(__file__).parent, timeout=timeout,
    )


def merge_state(run_commit: str | None, *, fetch: bool = True,
                repo: Path | None = None) -> dict:
    """Merge state of this checkout and of ``run_commit`` relative to origin/main, as a dict (branch, head, head_merged, ahead, run_commit, run_commit_merged, fetched, notes).
    The ``*_merged`` fields are None when undeterminable and callers MUST treat None as not merged; ``fetch=False`` skips the network (origin/main may be stale)."""
    def git(*args: str, timeout: int | None = None):
        return _git(*args, cwd=repo, timeout=timeout)

    state = {
        "branch": None, "head": None, "head_merged": None, "ahead": None,
        "run_commit": run_commit, "run_commit_merged": None,
        "fetched": False, "notes": [],
    }

    head = git("rev-parse", "--short", "HEAD")
    if head.returncode != 0:
        state["notes"].append(
            "not a git checkout, so nothing about this run's provenance could "
            "be verified")
        return state
    state["head"] = head.stdout.strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch.returncode == 0:
        state["branch"] = branch.stdout.strip()

    if fetch:
        try:
            fetched = git("fetch", "--quiet", "origin", "main",
                           timeout=_FETCH_TIMEOUT_S)
            state["fetched"] = fetched.returncode == 0
        except subprocess.SubprocessError:
            state["fetched"] = False
        if not state["fetched"]:
            state["notes"].append(
                f"could not reach the remote, so {MAIN_REF} may be out of date")
    else:
        state["notes"].append(
            f"the remote was not contacted, so {MAIN_REF} may be out of date")

    if git("rev-parse", "--verify", "--quiet", MAIN_REF).returncode != 0:
        state["notes"].append(
            f"this clone has no {MAIN_REF} reference to compare against")
        return state

    state["head_merged"] = git(
        "merge-base", "--is-ancestor", "HEAD", MAIN_REF).returncode == 0
    count = git("rev-list", "--count", f"{MAIN_REF}..HEAD")
    if count.returncode == 0 and count.stdout.strip().isdigit():
        state["ahead"] = int(count.stdout.strip())

    if not run_commit:
        state["notes"].append(
            "this run's manifest records no git commit, so the code that "
            "generated it cannot be identified")
        return state
    # A commit that only ever existed on someone's laptop is absent here, which
    # is a different problem from "on a branch" and needs saying differently.
    if git("cat-file", "-e", f"{run_commit}^{{commit}}").returncode != 0:
        state["notes"].append(
            f"commit {run_commit} is not in this clone (never pushed?), so it "
            "could not be checked")
        return state
    state["run_commit_merged"] = git(
        "merge-base", "--is-ancestor", run_commit, MAIN_REF).returncode == 0
    return state


def _update_latest_symlink(parent: Path, run_dir: Path) -> None:
    """Point parent/latest at run_dir, falling back to a directory junction on Windows, then to a warning (``resolve_run_dir`` does not use the link)."""
    link = parent / "latest"
    # lexists also catches broken symlinks and junctions, which exists() misses.
    if os.path.lexists(link):
        link.unlink()
    try:
        link.symlink_to(run_dir.relative_to(parent), target_is_directory=True)
    except OSError:
        try:
            # Junction targets must be absolute; mklink is a cmd builtin.
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(run_dir.resolve())],
                check=True, capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            print(
                f"  WARNING: could not update the {link} pointer (no symlink "
                "privilege, junction fallback failed); runs are unaffected.",
                file=sys.stderr,
            )


def create_run_dir(
    runs_root: str | Path,
    label: str,
    config: dict,
    snapshot_dirs: dict[str, Path] | None = None,
    extra_manifest_fields: dict | None = None,
) -> Path:
    """Create a run directory with a manifest and point ``latest`` at it. ``extra_manifest_fields`` are added as given;
    ``snapshot_dirs`` ({name: source}) are copied into ``inputs/<name>`` so the run stays reproducible."""
    runs_root = Path(runs_root)
    run_id = new_run_id(label)
    run_dir = runs_root / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = runs_root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    if snapshot_dirs:
        for name, src in snapshot_dirs.items():
            shutil.copytree(src, run_dir / "inputs" / name)

    commit, branch, dirty, dirty_files = _git_status()
    manifest = {
        "manifest_version": 3,
        "run_id": run_dir.name,
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "git_dirty_files": dirty_files,
        "inputs_snapshot": bool(snapshot_dirs),
        "model": config.get("model"),
        "config": config,
        **(extra_manifest_fields or {}),
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _update_latest_symlink(runs_root.parent, run_dir)
    return run_dir


def resolve_constitution_dir(prompts_dir: str | Path) -> Path | None:
    """The sibling ``inputs/constitution`` of a run's ``inputs/prompts`` snapshot, else None (callers use the live ``constitution/``)."""
    prompts_dir = Path(prompts_dir)
    if prompts_dir.name == "prompts" and prompts_dir.parent.name == "inputs":
        candidate = prompts_dir.parent / "constitution"
        if candidate.is_dir():
            return candidate
    return None


_RUN_DIR_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_")


def resolve_run_dir(runs_root: str | Path, run_id: str | None = None) -> Path:
    """An existing run directory: ``run_id`` if given (a missing one raises SystemExit, no fallback), else the latest pipeline-created one.
    Hand-made ``local_*`` directories are ignored for "latest" and stay reachable by ``--run-id``."""
    runs_root = Path(runs_root)
    if run_id:
        run_dir = runs_root / run_id
        if not run_dir.is_dir():
            raise SystemExit(f"Run '{run_id}' not found under {runs_root}")
        return run_dir
    runs = sorted(
        d for d in runs_root.iterdir() if d.is_dir() and _RUN_DIR_TS_RE.match(d.name)
    ) if runs_root.is_dir() else []
    if not runs:
        raise SystemExit(f"No runs found under {runs_root} — nothing to resume.")
    return runs[-1]


def warn_if_backend_changed(run_dir: str | Path, live_config: dict) -> None:
    """On ``--resume``, warn if the config's ``backend`` differs from the run's recorded one (allowed, but it mixes cost accounting)."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    started = (manifest.get("config") or {}).get("backend", "api")
    current = live_config.get("backend", "api")
    if started != current:
        print(
            f"  WARNING: this run started on backend {started!r} but config.yaml now says "
            f"{current!r}. Resuming will finish it under a different backend (mixed generation "
            "semantics and cost accounting in one run). Each cost_log.jsonl row is tagged with "
            "its backend.",
            file=sys.stderr,
        )


class Checkpoint:
    """Persist a set of completed IDs to disk so runs can be resumed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict = {"completed": [], "last_updated": None}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        self._completed: set = set(self._data.get("completed", []))

    def is_done(self, id_: str | int) -> bool:
        return str(id_) in self._completed

    def mark_done(self, id_: str | int) -> None:
        key = str(id_)
        if key not in self._completed:
            self._completed.add(key)
            self._data["completed"] = list(self._completed)
            self._data["last_updated"] = datetime.now(UTC).isoformat()
            ensure_dir(self.path.parent)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)

    @property
    def done_count(self) -> int:
        return len(self._completed)
