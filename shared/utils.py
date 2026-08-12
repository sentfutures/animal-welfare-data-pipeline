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


def parallel_map(fn, items: list, workers: int):
    """Map fn over items with a thread pool, yielding results in input order.

    fn must be side-effect free (API call + parsing only): because results come
    back in input order, callers can zip() them with items and keep all file
    writes and checkpoint marks on their own thread. If a call ultimately fails,
    the exception surfaces here; items already yielded are safely checkpointed
    and --resume picks up the rest.
    """
    if workers <= 1 or len(items) <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(fn, items)


REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_relative(path: str | Path) -> str:
    """A path rendered relative to the repo root, for values that get WRITTEN
    into reports and manifests. An absolute path bakes the machine it ran on
    (a home directory, a username) into files that are committed and, for audit
    reports, published — so anything inside the repo is recorded as
    "outputs/dad/runs/..." instead. A path outside the repo keeps only its file
    or directory name, since its parent directories say nothing about the run."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- SDF layout
#
# The SDF layers were renumbered from 1-2/3/4/5 to 1/2/3/4 (composition and
# planning had been counted as two layers when only one call is made). Runs
# made before the renumber keep the old directory names, and they are read by
# the viewer, evals/report_sdf.py and the handoff page, so both layouts have to
# resolve.
#
# The two layouts OVERLAP rather than merely differ: old "layer3" holds drafts
# while new "layer3" holds rewrites, and old "layer4" holds rewrites while new
# "layer4" holds scores. A resolver that tried the new directory name first and
# fell back to the old one would therefore read an old run's drafts as its
# rewrites. So resolution keys off the OUTPUT FILE, whose basename is unique per
# stage, and a stage directory alone is only ever resolved for writing.
_SDF_STAGE_DIRS = {1: ("layer1", "layer12"), 2: ("layer2", "layer3"),
                   3: ("layer3", "layer4"), 4: ("layer4", "layer5")}

# stage key -> (layer number, output filename)
_SDF_STAGE_FILES = {"dealt": (1, "prompts.jsonl"), "plan": (1, "plans.jsonl"),
                    "draft": (2, "drafts.jsonl"), "rewrite": (3, "rewrites.jsonl"),
                    "score": (4, "scores.jsonl")}


def sdf_stage_dir(run_dir: str | Path, layer: int) -> Path:
    """The directory for one SDF layer, honouring a pre-renumber run's layout.

    The layout is decided ONCE for the whole run, not per layer, because the two
    naming schemes overlap: a pre-renumber run already contains a "layer4" (its
    rewrites), so asking "does layer4 exist?" when resolving layer 4 — the score
    stage, which such a run keeps in "layer5" — answers yes for the wrong
    directory and a resume would write scores over rewrites. Only "layer12"
    appears in the old scheme and never in the new one, so it is the marker.
    """
    run = Path(run_dir)
    new, old = _SDF_STAGE_DIRS[layer]
    return run / (old if (run / "layer12").is_dir() else new)


# Layer number -> (current template name, pre-renumber template name).
_SDF_TEMPLATES = {1: ("layer1.txt", "layers1-2.txt"), 2: ("layer2.txt", "layer3.txt"),
                  3: ("layer3.txt", "layer4.txt"), 4: ("layer4.txt", "layer5.txt")}


def sdf_template_path(prompts_dir: str | Path, layer: int) -> Path:
    """One layer's prompt template, honouring a pre-renumber snapshot.

    Each run freezes its templates into inputs/prompts/, so --resume on a run
    started before the renumber must read that run's own OLD filenames. The old
    names overlap the new ones (old layer3.txt is the draft template, new
    layer3.txt is the rewrite template), so the layout is decided once for the
    whole directory rather than per file: only a pre-renumber snapshot contains
    ``layers1-2.txt``, and that file is the marker.
    """
    d = Path(prompts_dir)
    new, old = _SDF_TEMPLATES[layer]
    return d / (old if (d / "layers1-2.txt").exists() else new)


def sdf_stage_file(run_dir: str | Path, stage: str) -> Path:
    """Path to one SDF stage's output jsonl, honouring a pre-renumber layout.

    ``stage`` is one of dealt/plan/draft/rewrite/score — named, not numbered,
    because the numbers moved and the names did not. Falls back to the old
    location only if the file is actually there; otherwise returns the current
    path, so a missing-file message names the layout in use today.
    """
    layer, filename = _SDF_STAGE_FILES[stage]
    new, old = _SDF_STAGE_DIRS[layer]
    run = Path(run_dir)
    if not (run / new / filename).exists() and (run / old / filename).exists():
        return run / old / filename
    return run / new / filename


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
    """Parse the JSON value in a model response, tolerating surrounding chatter.

    Models occasionally wrap their JSON in markdown fences or add prose before
    or after it ("Here are the subtypes: [...] Let me know if..."), which bare
    json.loads rejects ("Extra data" / "Expecting value") — crashing a paid run
    on an otherwise usable response. This tries a full parse from every `[`/`{`
    in the text and returns the longest value that parses, so a short bracketed
    aside in the preamble can't shadow the real payload.

    Raises json.JSONDecodeError when no complete JSON value is present — and
    also when the payload itself is broken: truncated by max_tokens, or
    malformed mid-array (missing/trailing comma, both common LLM slip-ups).
    A broken container usually contains smaller values that do parse, and
    salvaging such a fragment would feed the caller a wrong-shaped result
    (a dict where a list was expected, with elements silently dropped)
    instead of a clean parse error. The unifying signal: a failed parse
    whose consumed region fully contains a successfully parsed candidate is
    a real payload that broke partway — candidates inside or after it are
    its fragments and are disqualified, while a complete value found before
    it (a genuine payload followed by broken chatter) is still returned.

    strict=False: literal control characters inside string values (raw
    newlines/tabs) are tolerated — the way prose-heavy JSON at temperature 1.0
    most often goes invalid, and the historical cause of silently empty scopes.
    """
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
    """Extract top-level {...} objects one at a time via brace matching, so a
    truncated or trailing-garbage array/stream still yields its complete
    objects. strict=False matches extract_json's control-char tolerance, so a
    salvageable object isn't dropped for a literal newline inside a string.

    This is the last-resort recovery the recover=True paths below share (and
    that step-1 dilemma parsing relies on) — a model slip that breaks the outer
    container but leaves individual objects intact still yields usable data."""
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
    """extract_json narrowed to an object. A wrong-shaped value raises
    json.JSONDecodeError so shape failures join parse failures on the caller's
    existing error path, instead of crashing later with AttributeError when
    .get() hits a list.

    recover=True adds the step-1 salvage fallback: when the container is broken
    (truncated/trailing garbage) or the value isn't an object, take the first
    complete top-level {...} object via salvage_json_objects. The strict default
    coerces nothing, preserving the clean parse/shape error."""
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
    """extract_json narrowed to an array; wrong shape raises json.JSONDecodeError
    (see extract_json_object).

    recover=True adds two model-slip fallbacks, for callers that would rather
    salvage than lose a paid response (evals, step 1): an array wrapped in a
    single-key object — {"reasons": [...]}, a common judge slip — is unwrapped
    to its array, and a broken/truncated container yields its complete top-level
    {...} objects via salvage_json_objects. The strict default coerces nothing,
    preserving extract_json's no-wrong-shape guarantee."""
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
        # '[', sometimes no commas): extract_json legitimately parses ONE of
        # those objects, so the except-branch salvage above never runs and a
        # fully recoverable reply was being discarded. Observed live at ~14% of
        # eval judge calls in one paid pass (11 of 79), every one of this shape.
        # Salvaging here reads the whole stream.
        if (objs := salvage_json_objects(text)) and len(objs) > 1:
            return objs
    raise json.JSONDecodeError("JSON value is not an array", text, 0)


def load_prompt(path: str | Path, **kwargs) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if kwargs:
        text = text.format(**kwargs)
    return text


def looks_like_transcript_echo(text: str) -> bool:
    """True when a generation came back wrapped in a transcript replay
    ("USER: <message>\\nASSISTANT: <reply>") instead of the reply alone —
    observed live from a step-3 rewrite call. Such output must never become a
    training record; callers treat it like a truncated reply (skip without
    checkpointing so --resume retries). Deliberately narrow: only a role
    marker at the very start of the text counts, so advice that merely
    mentions "USER:" mid-reply is not flagged."""
    return bool(re.match(r"\s*(USER|ASSISTANT|HUMAN)\s*:", text[:40]))


# A template that carries both a system and a user half separates them with a
# line equal to this marker. See load_split_prompt.
_PROMPT_SPLIT_MARKER = "===USER==="


def load_split_prompt(path: str | Path, **kwargs) -> tuple[str, str]:
    """Load a two-part prompt template as (system, user).

    The system half and the user half are separated by a line equal to
    `===USER===`. Each half is formatted with the same kwargs — a half simply
    ignores any placeholder it does not contain. A template with NO marker
    returns ("", <whole formatted template>): callers and pre-split run
    snapshots that predate the split still send a user-only prompt with an
    empty system prompt, identical to load_prompt's behaviour."""
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
    """Return (short_commit, branch, dirty, dirty_files) for the repo, or
    (None, None, False, []) outside git.

    branch is the literal "HEAD" on a detached checkout (CI's actions/checkout
    leaves one), so it records where the run came from without pretending a
    detached HEAD is a branch name.
    """
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
    """Run a git command in the repo. A non-zero returncode is the caller's to
    inspect, not an exception — so git failing is never itself fatal.

    The one thing that DOES raise: passing timeout= can raise
    subprocess.TimeoutExpired. Every such call site must handle it (merge_state
    wraps its one fetch in try/except subprocess.SubprocessError).
    """
    return subprocess.run(
        ["git", *args],
        capture_output=True, text=True, encoding="utf-8",
        cwd=cwd or Path(__file__).parent, timeout=timeout,
    )


def merge_state(run_commit: str | None, *, fetch: bool = True,
                repo: Path | None = None) -> dict:
    """Merge state of this checkout, and of the commit a run was generated from,
    relative to origin/main.

    Returns a dict with:
      branch, head             -- current checkout ("HEAD" if detached)
      head_merged              -- is HEAD reachable from origin/main?
      ahead                    -- commits on HEAD not in origin/main (None if unknown)
      run_commit               -- the commit echoed back, for callers' messages
      run_commit_merged        -- is run_commit reachable from origin/main?
      fetched                  -- was origin/main refreshed from the remote?
      notes                    -- plain-English reasons anything is unknown

    Both *_merged fields are None when the answer could not be determined (no
    repo, no origin/main, a commit this clone has never seen). Callers MUST
    treat None as NOT merged: an unverifiable provenance claim is not a safe
    one. `fetch=False` skips the network entirely, at the cost of comparing
    against a possibly stale origin/main (see the note it records). `repo`
    overrides which checkout is inspected (defaults to this file's own).
    """
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
    """Point parent/latest at run_dir. Symlinks on Windows need Developer Mode
    or elevation (WinError 1314), so fall back to a directory junction (no
    privilege required), and failing that warn and continue — the pointer is a
    convenience; resolve_run_dir orders runs by directory name, not this link."""
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
) -> Path:
    """Create a new run directory with a manifest, and point the `latest` symlink at it.

    snapshot_dirs maps name -> source directory; each is copied into
    run_dir/inputs/<name> so the run stays reproducible even after the
    source files (prompt templates, constitution) change.
    """
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
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _update_latest_symlink(runs_root.parent, run_dir)
    return run_dir


def resolve_constitution_dir(prompts_dir: str | Path) -> Path | None:
    """If prompts_dir is a run's input snapshot (.../inputs/prompts), return the
    sibling inputs/constitution dir; otherwise None (callers fall back to the
    repo's live constitution/)."""
    prompts_dir = Path(prompts_dir)
    if prompts_dir.name == "prompts" and prompts_dir.parent.name == "inputs":
        candidate = prompts_dir.parent / "constitution"
        if candidate.is_dir():
            return candidate
    return None


_RUN_DIR_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_")


def resolve_run_dir(runs_root: str | Path, run_id: str | None = None) -> Path:
    """Find an existing run directory: by ID if given, otherwise the most recent.

    An explicit run_id that names no directory under runs_root is an error,
    not a fallback: the call raises SystemExit naming the missing run_id and
    runs_root, rather than quietly resuming the latest run.

    "Most recent" considers only pipeline-created dirs (timestamp-prefixed
    names). Hand-made dirs (e.g. local_* scratch runs) sort after every
    timestamp lexicographically and would otherwise hijack every bare
    --resume; they remain reachable explicitly via --run-id.
    """
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
    """On --resume, warn if the live config's `backend` differs from the one the
    run started with (recorded in run_manifest.json).

    Switching mid-run is allowed — flipping to `api` after hitting the
    claude_code usage limit is the documented recovery — but it mixes generation
    semantics and cost accounting within one run, so surface it rather than
    letting it happen silently.
    """
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
