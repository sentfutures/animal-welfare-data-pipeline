"""Shared helpers for the viewer pages."""

import difflib
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dad_pipeline import profiles
from viewer import loader

FOLD_THRESHOLD = 2000  # chars; variable values longer than this get folded out of prompts


def json_block(obj, key: str, label: str = "JSON", expanded: bool = False) -> None:
    """Collapsible, whole-text-copyable JSON. st.json's tree offers only
    per-node copy, st.code can't collapse, and expanders don't nest — so a
    toggle gates a code block (which has the copy-everything button)."""
    if st.toggle(label, value=expanded, key=f"json_{key}"):
        st.code(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                language="json", wrap_lines=True)


def pick_run(sidebar: bool = True) -> loader.RunInfo | None:
    """Run selector seeded from query params; keeps params in sync.
    Renders in the sidebar by default so the content area stays clean."""
    container = st.sidebar if sidebar else st
    runs = loader.list_runs()
    if not runs:
        st.info("No runs found under outputs/. Run a pipeline first.")
        return None

    # loader.PIPELINES order puts SDF first (the default)
    pipelines = [p for p in loader.PIPELINES if any(r.pipeline == p for r in runs)]
    qp_pipeline = st.query_params.get("pipeline")
    pipeline = container.selectbox(
        "Pipeline", pipelines,
        index=pipelines.index(qp_pipeline) if qp_pipeline in pipelines else 0,
        key="sel_pipeline",
    )

    pipeline_runs = [r for r in runs if r.pipeline == pipeline]
    run_ids = [r.run_id for r in pipeline_runs]
    qp_run = st.query_params.get("run")
    run_id = container.selectbox(
        "Run", run_ids,
        index=run_ids.index(qp_run) if qp_run in run_ids else 0,
        key="sel_run",
    )

    st.query_params["pipeline"] = pipeline
    st.query_params["run"] = run_id
    run = next(r for r in pipeline_runs if r.run_id == run_id)
    if run.pipeline == "dad":
        # labels, scope axes and the principles file follow the run's own profile
        profiles.use(profiles.name_from_config(run.config))
    return run


def run_provenance_note(run: loader.RunInfo) -> None:
    """One-line provenance caveat, shown at most once per page."""
    if not run.has_snapshot:
        st.caption(f":material/history: Prompts for this run are reconstructed from git "
                   f"commit `{run.git_commit}`" +
                   (" — repo had uncommitted changes at run time, so they may differ from what actually ran."
                    if run.git_dirty else "."))


def fold_long_values(text: str, variables: dict) -> tuple[str, dict[str, str]]:
    """Replace very long substituted values (constitution, preamble) with a
    short marker so the prompt stays readable; return (folded_text, folded)."""
    folded = {}
    for name, value in variables.items():
        # Injected data blocks (*_block) fold at a much lower bar than free text:
        # they're duplicated from earlier pipeline stages, so inline they only
        # add scrolling. 200 chars avoids pointless toggles for tiny blocks.
        if isinstance(value, str) and value in text and (
                len(value) > FOLD_THRESHOLD or (name.endswith("_block") and len(value) > 200)):
            marker = f"⟨{name}: {len(value):,} chars — expand below⟩"
            text = text.replace(value, marker)
            folded[name] = value
    return text, folded


def show_rendered_prompt(rendered, key: str = "", show_run_warnings: bool = True) -> None:
    """Render a RenderedPrompt: system + user with long values folded.
    `key` must be unique per prompt on the page (e.g. the stage name).
    Run-level provenance warnings are suppressed when show_run_warnings=False
    (pages show them once at the top instead)."""
    run_level = ("Pre-snapshot run", "Repo was dirty")
    for w in rendered.warnings:
        if any(w.startswith(prefix) for prefix in run_level):
            if show_run_warnings:
                st.warning(w)
        else:
            st.warning(w)
    if not rendered.is_llm_call:
        st.caption("No LLM call at this stage for this record.")
        return

    # System prompt collapsed by default — it's often the largest block (e.g. the
    # constitution principles), and you shouldn't have to scroll past it to reach
    # the per-case user message.
    if rendered.system:
        if st.toggle(f"System prompt — {rendered.system_label} ({len(rendered.system):,} chars)",
                     value=False, key=f"sys_{key}_{rendered.stage}"):
            st.code(rendered.system, language=None, wrap_lines=True)
    folded_all = {}
    if rendered.user:
        user_text, folded = fold_long_values(rendered.user, rendered.variables)
        folded_all.update(folded)
        st.markdown("**User message**")
        st.code(user_text, language=None, wrap_lines=True)
    for name, value in folded_all.items():
        if st.toggle(f"Show {name} ({len(value):,} chars)", key=f"fold_{key}_{rendered.stage}_{name}"):
            st.code(value, language=None, wrap_lines=True)


def show_diff(before: str, after: str, from_label: str, to_label: str, key: str) -> None:
    """Side-by-side by default (easier to read whole texts), with a toggle for
    the unified diff (better for spotting exact line changes)."""
    if before == after:
        st.caption("No changes — output identical to input.")
        return
    if st.toggle("Unified diff", key=f"diff_{key}"):
        diff = "\n".join(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=from_label, tofile=to_label, lineterm="",
        ))
        st.code(diff, language="diff", wrap_lines=True)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{from_label}**")
            st.code(before, language=None, wrap_lines=True)
        with col_b:
            st.markdown(f"**{to_label}**")
            st.code(after, language=None, wrap_lines=True)
