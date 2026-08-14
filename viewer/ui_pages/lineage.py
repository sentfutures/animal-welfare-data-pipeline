"""Document lineage — the app's main page. Pick a run, click a document row,
and read the final document side by side with the prompts that produced it."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dad_pipeline import step2_responses
from dad_pipeline.step1_dilemmas import dealt_cards
from viewer import loader, rendering
from viewer.ui_pages import common

PANEL_HEIGHT = 750  # px; the two comparison panels scroll independently

run = common.pick_run()
if run is None:
    st.stop()

manifest = loader.load_manifest(run.run_dir)
st.title(run.label or run.run_id)
st.caption((run.created_at or "").replace("T", " ")[:16])
common.run_provenance_note(run)

finals = loader.load_final(run.run_dir, run.pipeline)
id_key = "doc_id" if run.pipeline == "sdf" else "record_id"
ids = [r[id_key] for r in finals]

# Label helpers live in loader (pure, reused by the compare page); thin aliases
# keep this page's call sites unchanged.
_doc_title = loader.doc_first_line
_goal_label = loader.dad_goal_label


def _pick_document(options: list[str], labels: dict[str, str], noun: str) -> str | None:
    """Document dropdown, seeded from the ?doc= query param.

    The widget needs a stable key: without one, Streamlit derives its identity
    from its parameters INCLUDING `index`, and since we compute `index` from
    the query param, every selection changed the identity on the next rerun and
    orphaned the user's choice (the select-twice bug). The key is scoped to the
    run so switching runs starts fresh."""
    if not options:
        st.caption(f"No {noun}s match the current filters.")
        return None
    key = f"doc_pick_{run.pipeline}_{run.run_id}"
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)  # stale value (e.g. filters changed) — reseed below
    qp_doc = st.query_params.get("doc")
    choice = st.selectbox(
        f"{noun.capitalize()} ({len(options)})", options,
        index=options.index(qp_doc) if qp_doc in options else 0,
        format_func=lambda i: labels.get(i, str(i)),
        key=key,
    )
    st.query_params["doc"] = choice
    return choice


# --- Document selection ---
selected_id = None
dad_by_prompt = False  # incomplete DAD run: enumerate step-1 prompts, not final records
if run.pipeline == "dad" and not finals:
    dilemmas = loader.load_stage(run.run_dir, "dad", "step1_dilemmas")
    if dilemmas:
        dad_by_prompt = True
        st.info("No responses generated yet — showing the step-1 dilemma prompts.")
        options, labels = [], {}
        for d in dilemmas:
            pid = loader._pkey(d)
            options.append(pid)
            labels[pid] = _goal_label(dealt_cards(d), d.get("user_message"))
        selected_id = _pick_document(options, labels, "prompt")
    else:
        st.info("No dilemmas generated in this run yet.")
elif not finals:
    st.info("No final corpus in this run yet (incomplete run).")
elif run.pipeline == "sdf":
    subtypes = {s["subtype_id"]: s for s in loader.load_stage(run.run_dir, "sdf", "subtypes")}
    f1, f2 = st.columns([3, 1])
    all_types = sorted({subtypes.get(d.get("subtype_id"), {}).get("type_name", "") for d in finals})
    type_filter = f1.multiselect(
        "Filter by document type (from Layer 1)", all_types, placeholder="All types",
        help="Top-level document categories generated in Layer 1.",
    )
    min_score = f2.slider("Min score (from Layer 4)", 1, 10, 1,
                          help="Minimum alignment and realism scores assigned in Layer 4.")

    options, labels = [], {}
    for d in sorted(finals, key=lambda d: str(d.get("subtype_id", ""))):
        st_rec = subtypes.get(d.get("subtype_id"), {})
        scores = d.get("scores", {})
        if type_filter and st_rec.get("type_name", "") not in type_filter:
            continue
        if (scores.get("alignment") or 0) < min_score or (scores.get("realism") or 0) < min_score:
            continue
        options.append(d["doc_id"])
        labels[d["doc_id"]] = f"{_doc_title(d.get('content'))}   —   {st_rec.get('subtype_name', '?')}"

    st.caption("Dropdown labels: *document title — subtype (from Layer 2)*")
    selected_id = _pick_document(options, labels, "document")
else:
    dad_legacy = loader.dad_is_legacy(run.run_dir)
    audits = {a["record_id"]: a for a in loader.load_stage(
        run.run_dir, "dad", "step6" if dad_legacy else "step3_rewrites")}

    if dad_legacy:
        injections = sorted({a.get("injection_used", "") for a in audits.values() if a.get("injection_used")})
        inj_filter = st.multiselect(
            "Filter by injection", injections, placeholder="All injections",
            help="The system-prompt injection active when the draft response was sampled.",
        )
        keep = lambda audit: not inj_filter or audit.get("injection_used", "?") in inj_filter
        suffix = lambda audit: audit.get("injection_used", "?")
        sort_key = lambda rec: audits.get(rec.get("record_id"), {}).get("injection_used", "")
        st.caption("Dropdown labels: *user message — injection (from the response sampling step)*")
    else:
        # Current-format runs: label each record with its stable ids then the
        # goal — "E-#### · P-#### · R-#### · <goal>". Every id the corpus audit
        # can reference is here (example E-, prompt P-, response R-), so a graph
        # or reasons row flagged by any of them is easy to find. prompt_gid (P-)
        # isn't on the final/rewrite record, so join it from step 1.
        keep = lambda audit: True
        suffix = None
        sort_key = lambda rec: loader._pkey(audits.get(rec.get("record_id"), {}))
        pgid_by_pid = {k: d.get("prompt_gid")
                       for d in loader.load_stage(run.run_dir, "dad", "step1_dilemmas")
                       for k in loader._pkeys(d)}
        st.caption("Dropdown labels: *example E- · prompt P- · response R- — goal* "
                   "(these stable ids match the corpus audit).")

    options, labels = [], {}
    for rec in sorted(finals, key=sort_key):
        audit = audits.get(rec.get("record_id"), {})
        if not keep(audit):
            continue
        user_msg = rec["messages"][0]["content"] if rec.get("messages") else ""
        options.append(rec["record_id"])
        if suffix:
            labels[rec["record_id"]] = f"{_doc_title(user_msg)}   —   {suffix(audit)}"
        else:
            goal = _goal_label(dealt_cards(audit), user_msg)
            ids = [rec.get("example_gid") or audit.get("example_gid"),
                   pgid_by_pid.get(loader._pkey(audit)),
                   rec.get("response_gid") or audit.get("response_gid")]
            prefix = " · ".join(x for x in ids if x)
            labels[rec["record_id"]] = f"{prefix} · {goal}" if prefix else goal

    selected_id = _pick_document(options, labels, "record")


def _call_stats_line(cost_stage: str, item_id: str | None) -> str | None:
    """One-line summary of the API call(s) behind a stage: model · cost ·
    wall-clock · retries. Falls back to model-only for runs logged before
    per-record stats existed."""
    s = loader.call_stats(run.run_dir, cost_stage, item_id)
    if s is None:
        return None
    models = ", ".join(s["models"])
    if not s["per_item"]:
        return f"{models} · per-record cost/time/retries not recorded in this run"
    bits = [models, f"${s['cost_usd']:.4f}"]
    if s.get("duration_s") is not None:
        bits.append(f"{s['duration_s']:.1f}s")
    if s.get("retries") is not None:
        bits.append("no retries" if s["retries"] == 0
                    else f"{s['retries']} retr{'y' if s['retries'] == 1 else 'ies'}")
    if s["calls"] > 1:
        bits.append(f"{s['calls']} calls")
    if s.get("batch_size", 1) > 1:
        bits.append(f"one call drafted a batch of {s['batch_size']} (cost/time are the whole batch's)")
    return " · ".join(bits)


def stage_expander(title: str, stage: str, lineage: dict, output_fn,
                   stats: tuple[str, str | None] | None = None,
                   gid: str | None = None):
    """One stage: call stats, the rendered prompt, then the output it produced.
    `stats` is (cost-log stage tag, item id) for the API call(s) behind the
    stage; `gid` is the stage output's stable cross-run id (S-/P-/R-/E-),
    shown ahead of the stats."""
    with st.expander(f":blue[{title}]"):
        line = _call_stats_line(*stats) if stats else None
        bits = [b for b in (gid, line) if b]
        if bits:
            st.caption(f":material/speed: {' · '.join(bits)}")
        rendered = rendering.render_prompt(run.pipeline, stage, run.run_dir, manifest, lineage)
        common.show_rendered_prompt(rendered, key=stage, show_run_warnings=False)
        st.markdown("##### Output at this stage")
        output_fn()


# --- Side-by-side: document (left) vs prompts (right) ---
if selected_id is None:
    if finals or dad_by_prompt:
        st.caption("Click a document above to open it.")
elif run.pipeline == "sdf":
    lin = loader.sdf_lineage(run.run_dir, selected_id)
    subtype = lin.get("subtype") or {}
    st.divider()
    doc_col, prompts_col = st.columns(2)

    with doc_col:
        st.subheader(subtype.get("subtype_name") or f"Document {selected_id[:8]}")
        with st.container(height=PANEL_HEIGHT):
            st.code((lin.get("final") or {}).get("content", ""), language=None, wrap_lines=True)

    with prompts_col:
        st.subheader("Prompts")
        st.caption("Each layer's prompt and what it produced")
        with st.container(height=PANEL_HEIGHT):
            stage_expander("Legacy layer 1 — document type", "document_types", lin,
                           lambda: common.json_block(lin["doc_type"], key="l1", label="document type", expanded=True)
                           if lin["doc_type"] else st.caption("not found"))
            stage_expander("Legacy layer 2 — subtype", "subtypes", lin,
                           lambda: common.json_block(lin["subtype"], key="l2", label="subtype", expanded=True)
                           if lin["subtype"] else st.caption("not found"))
            stage_expander("Layer 2 — draft", "draft", lin,
                           lambda: st.code(lin["draft"]["content"], language=None, wrap_lines=True)
                           if lin["draft"] else st.caption("not reached"))

            def layer3_output():
                rw = lin["rewrite"]
                if not rw:
                    st.caption("not reached")
                    return
                if rw.get("review_notes"):
                    st.info(f"Review notes: {rw['review_notes']}")
                common.show_diff(rw["original"], rw["rewritten"], "draft", "rewritten", key="l4")
            stage_expander("Layer 3 — constitutional rewrite", "rewrite", lin, layer3_output)

            stage_expander("Layer 4 — scoring", "score", lin,
                           lambda: common.json_block((lin["score"] or {}).get("scores", {}),
                                                     key="l5", label="scores", expanded=True)
                           if lin["score"] else st.caption("not reached"))
else:
    lin = (loader.dad_lineage_by_prompt(run.run_dir, selected_id) if dad_by_prompt
           else loader.dad_lineage(run.run_dir, selected_id))
    audit = lin.get("rewrite") or {}
    dilemma = lin.get("dilemma") or {}
    st.divider()
    doc_col, prompts_col = st.columns(2)

    with doc_col:
        # Stable global ids as the headline — the finished artifact's identity
        # (example / prompt / response). The upstream ids (S-#### scenario,
        # C-#### control) live on their stage expanders below, as do the
        # per-run scenario ids; classification tag small underneath.
        annotation = dealt_cards(dilemma) or dealt_cards(audit)
        head = []
        if audit.get("example_gid"):
            head.append(f"example {audit['example_gid']}")
        if dilemma.get("prompt_gid"):
            head.append(f"prompt {dilemma['prompt_gid']}")
        if audit.get("response_gid"):
            head.append(f"response {audit['response_gid']}")
        st.subheader("  ·  ".join(head)
                     or (loader._pkey(dilemma) or loader._pkey(audit) or str(selected_id)))
        if lin.get("format") == "v2":
            taxa = dilemma.get("taxa_subcategory") or annotation.get("taxa_category")
            lev = annotation.get("leverage")
            tag = " · ".join(str(x) for x in [
                ", ".join(annotation.get("domain") or []),
                taxa,
                annotation.get("direction"),
                annotation.get("welfare_magnitude"),
                f"{lev} leverage" if lev else None,
            ] if x)
            st.caption(tag or "step-1 dilemma prompt")
        else:
            st.caption(f"injection `{audit.get('injection_used')}` · principle {audit.get('principle_id')}")
        with st.container(height=PANEL_HEIGHT):
            messages = (lin.get("final") or {}).get("messages", [])
            if messages:
                for msg in messages:
                    st.markdown(f"**{msg['role']}**")
                    st.code(msg["content"], language=None, wrap_lines=True)
            else:
                # Incomplete run: no final response yet — show the dilemma prompt itself.
                st.markdown("**user** *(dilemma prompt — no response generated yet)*")
                st.code(dilemma.get("user_message", ""), language=None, wrap_lines=True)
                common.json_block(dealt_cards(dilemma), key="doc_ann", label="scenario cards")

    with prompts_col:
        st.subheader("Prompts")
        st.caption("Each step's prompt and what it produced")
        with st.container(height=PANEL_HEIGHT):
            if lin.get("format") == "v2":
                # Ids linking each stage's expander to the cost-log rows of the
                # API call(s) that produced it (item_id, logged since 2026-07).
                scenario_id = dilemma.get("scenario_id")
                pid = loader._pkey(dilemma) or loader._pkey(audit)
                resp = lin.get("response") or {}
                resp_item = (f"{loader._pkey(resp)}_s{resp.get('sample_index', 0)}"
                             if loader._pkey(resp) else None)

                # Step 1a — scenario deal + plan (a billed call since 2026-07).
                # Pre-plan runs (no scenario_description) were pure sampling.
                sc = lin.get("scenario") or {}
                if sc.get("scenario_description"):
                    def step1a_output():
                        st.code(sc.get("scenario_description", ""), language=None, wrap_lines=True)
                        common.json_block(
                            {k: v for k, v in sc.items()
                             if k not in ("scenario_description", "variables")},
                            key="s1a", label="dealt scenario")
                    stage_expander("Step 1a — scenario deal + plan", "step1a_plan", lin,
                                   step1a_output, stats=("scenario_plan", scenario_id),
                                   gid=sc.get("scenario_gid") or dilemma.get("scenario_gid"))
                else:
                    with st.expander(":blue[Step 1a — scenario generation (sampled, no model call)]"):
                        if sc:
                            st.caption("Stratified categorical assignment for this example, drawn by the "
                                       "sampler — no LLM call (pre-plan run).")
                            common.json_block(sc, key="s1a", label="scenario")
                        else:
                            st.caption("scenario record not found (older run, or pre-scenario snapshot)")

                def step1b_output():
                    d = lin.get("dilemma") or {}
                    if not d:
                        st.caption("dilemma record not found")
                        return
                    # the 1b draft is draft_user_message when 1c ran, else the stored user_message
                    st.code(d.get("draft_user_message") or d.get("user_message", ""),
                            language=None, wrap_lines=True)
                    common.json_block(dealt_cards(d), key="s1b_ann", label="scenario cards")
                stage_expander("Step 1b — prompt draft", "step1_dilemmas", lin, step1b_output,
                               stats=("prompt_draft", scenario_id),
                               gid=dilemma.get("prompt_gid"))

                def step1c_output():
                    # 1c gate: pass/fail on the 1b draft, never a rewrite. On
                    # composed runs a passing draft goes on to the 1d refine;
                    # on gate-era runs it shipped directly.
                    d = lin.get("dilemma") or {}
                    gate = lin.get("gate")
                    if gate is None:
                        st.caption("not run (dad.dilemmas.gate was off, or the "
                                   "run predates the gate)")
                        return
                    passed = gate.get("passed")
                    if passed is True:
                        st.success("Gate: PASSED — the draft went on to shipping/refine.")
                    elif passed is False:
                        st.error("Gate: FAILED after the redraft cap — shipped anyway, "
                                 "flagged with gate_failures.")
                    else:
                        st.caption(":material/warning: gate reply unusable after retries — "
                                   "the draft shipped (raws in step1/gate_failures.jsonl)")
                    for f in (gate.get("failures") or d.get("gate_failures") or []):
                        st.write(f"- {f}")
                stage_expander("Step 1c — quality gate (pass/fail)",
                               "step1_gate", lin, step1c_output,
                               stats=("prompt_gate", scenario_id))

                def step1d_output():
                    # 1d refine: the draft→refined rewrite (also the sole 1c
                    # stage on pre-gate legacy runs).
                    d = lin.get("dilemma") or {}
                    if d.get("refine_failed"):
                        st.caption(":material/warning: every refine attempt was unusable — "
                                   "the 1b draft shipped unrefined (raw outputs in "
                                   "step1/refine_failures.jsonl)")
                        return
                    if d.get("draft_user_message") is None:
                        st.caption("not run (dad.dilemmas.refine was off for this run)")
                        return
                    if d.get("refine_notes"):
                        st.info(f"Notes: {d['refine_notes']}")
                    common.show_diff(d["draft_user_message"], d.get("user_message", ""),
                                     "1b draft", "refined", key="s1d")
                stage_expander("Step 1d — refine (review & rewrite; legacy 1c)",
                               "step1_refine", lin, step1d_output,
                               stats=("prompt_refine", scenario_id))

                scope_rec = lin.get("scope") or {}
                sel_source = scope_rec.get("selection_source")

                def _triggered_toggle(key: str):
                    """Retrieval provenance: the full rows whose trigger
                    conditions fired for this prompt (runs since retrieval)."""
                    trig = scope_rec.get("triggered_entries")
                    if not trig:
                        return
                    if scope_rec.get("selection_fallback"):
                        st.caption(":material/warning: selection unusable — the whole "
                                   "library was injected")
                    elif sel_source == "repair":
                        st.caption(":material/build: triggered_entries was missing from "
                                   "the scope output — recovered by a selection-only "
                                   "repair call (this run predates the standing "
                                   "select call)")
                    common.json_block(trig, key=key,
                                      label=f"triggered library entries ({len(trig)})")

                def step2_scope_output():
                    sc = scope_rec.get("scope")
                    if sc:
                        common.json_block(sc, key="s2a", label="scope", expanded=True)
                    else:
                        st.caption("not reached")
                    # No select call for this record (selection arrived inside
                    # the scope JSON, or predates selection_source): retrieval
                    # provenance shows here instead of in a 2a.5 expander.
                    if sel_source not in step2_responses.SELECT_CALL_SOURCES:
                        _triggered_toggle("s2a_trig")
                stage_expander("Step 2a — scope the case (patients, goal, levers, cost, magnitude, upside, counterfactual)",
                               "step2_scope", lin, step2_scope_output,
                               stats=("response_scope", pid))

                # Step 2a.5 — the dedicated retrieval call (and its miss-only
                # "repair" precursor). Absent for scope-time selections and for
                # runs predating library retrieval.
                if sel_source in step2_responses.SELECT_CALL_SOURCES:
                    stage_expander("Step 2a.5 — select library entries (retrieval)",
                                   "step2_select", lin,
                                   lambda: _triggered_toggle("s2a5_trig"),
                                   stats=("response_select", pid))

                # Tension retrieval was removed from the pipeline; still shown for
                # older runs that recorded it, so their lineage stays complete.
                if lin.get("tension_tag"):
                    stage_expander("Tensions (retrieval — earlier pipeline)", "step2_tag", lin,
                                   lambda: common.json_block(lin.get("tension_tag"), key="s2tag",
                                                             label="tensions", expanded=True))

                stage_expander("Step 2b — response draft (first take, scope, reasoning library)",
                               "step2_respond", lin,
                               lambda: st.code((lin.get("response") or {}).get("assistant_response", ""),
                                               language=None, wrap_lines=True)
                               if lin.get("response") else st.caption("not reached"),
                               stats=("response_draft", resp_item) if resp_item else None,
                               gid=resp.get("response_gid"))

                def step3_output():
                    if not audit:
                        st.caption("not reached")
                        return
                    common.show_diff(audit["draft_response"], audit["rewritten_response"],
                                     "draft response", "rewritten response", key="s3")
                stage_expander("Step 3 — rewrite against the distilled principles", "step3_rewrite", lin,
                               step3_output,
                               stats=("constitution_rewrite", audit.get("response_id")) if audit else None,
                               gid=audit.get("example_gid"))

                # Baseline control arm — only rendered when the run recorded one
                # (dad.baseline; absent for runs predating the stage or with it
                # off). Deliberately NOT a stage_expander: its prompt is the 1c
                # user message and its counterpart is the final response, both
                # already in the left panel — repeating them here only cramped
                # the text. The panels scroll independently on purpose: the two
                # responses don't align paragraph-for-paragraph.
                baseline_rec = lin.get("baseline")
                if baseline_rec:
                    base_model = baseline_rec.get("model") or "model"
                    with st.expander(f":blue[Baseline — plain {base_model}, "
                                     "no system prompt (comparison only)]"):
                        line = _call_stats_line("baseline_response", pid)
                        bits = [b for b in (baseline_rec.get("plain_gid"), line) if b]
                        if bits:
                            st.caption(f":material/speed: {' · '.join(bits)}")
                        st.caption("Control arm: the step-1c user prompt (left panel) sent "
                                   "verbatim to a plain model — no system prompt, no reasoning "
                                   "library, no constitution. Read it against the pipeline's "
                                   "response on the left; never enters the training corpus. "
                                   "(On fused-era runs, where 2b revised this draft, the "
                                   "word-diff below is the informative view.)")
                        final_msgs = (lin.get("final") or {}).get("messages") or []
                        target = (final_msgs[1]["content"] if len(final_msgs) > 1
                                  else (lin.get("response") or {}).get("assistant_response"))
                        if target and st.toggle(
                            "Word-diff vs final response (best on fused-era runs)",
                            value=False, key="baseline_diff",
                        ):
                            st.caption(":green[highlight] = added by the pipeline · "
                                       "~~struck~~ = dropped from the draft")
                            st.markdown(
                                '<div style="max-height:60vh;overflow-y:auto;padding:0.75rem 1rem;'
                                'border:1px solid rgba(128,128,128,.35);border-radius:8px;'
                                'font-size:0.9rem;line-height:1.55;">'
                                + rendering.inline_word_diff_html(
                                    baseline_rec["baseline_response"], target)
                                + "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.code(baseline_rec["baseline_response"], language=None, wrap_lines=True)
            else:
                # Legacy 7-step runs (pre-spec pipeline)
                stage_expander("Step 1 — principle annotation", "step1", lin,
                               lambda: common.json_block(
                                   {k: v for k, v in (lin.get("principle") or {}).items() if k != "content"},
                                   key="leg_s1", label="principle", expanded=True)
                               if lin.get("principle") else st.caption("principle record not found"))
                stage_expander("Step 2 — scenario", "step2", lin,
                               lambda: common.json_block(lin.get("scenario"), key="leg_s2",
                                                         label="scenario", expanded=True)
                               if lin.get("scenario") else st.caption("not found"))
                stage_expander("Step 3 — draft user prompt", "step3", lin,
                               lambda: st.code((lin.get("prompt") or {}).get("user_message", "—"),
                                               language=None, wrap_lines=True))

                def step4_output():
                    ref = lin.get("refined")
                    if not ref:
                        st.caption("not reached")
                        return
                    common.show_diff(ref["original"], ref["refined"], "draft prompt", "refined prompt", key="s4")
                stage_expander("Step 4 — refine user prompt", "step4", lin, step4_output)

                def step5_output():
                    resp = lin.get("response")
                    if not resp:
                        st.caption("not reached")
                        return
                    st.code(resp["assistant_response"], language=None, wrap_lines=True)
                    st.markdown(f"**Kept:** {'✅' if resp.get('kept') else '❌ (ruthless judge rejected)'}")
                stage_expander("Step 5 — response under injection", "step5", lin, step5_output)

                # Historical runs only — ruthless sampling was later removed from the pipeline
                if (lin.get("response") or {}).get("injection_used") == "ruthless":
                    stage_expander("Step 5b — ruthless judge", "step5_judge", lin,
                                   lambda: st.markdown(f"**Verdict (kept):** {(lin.get('response') or {}).get('kept')}"))

                def step6_output():
                    if not audit:
                        st.caption("not reached")
                        return
                    common.show_diff(audit["draft_response"], audit["rewritten_response"],
                                     "draft response", "constitutional rewrite", key="s6")
                stage_expander("Step 6 — constitutional rewrite (critical step)", "step6", lin, step6_output)

# --- Run-scoped template browser ---
st.divider()
with st.expander("All prompt templates for this run"):
    for template in rendering.list_templates(run.run_dir, run.git_commit, run.pipeline):
        st.markdown(f"**{template.name}**" + ("" if template.source == "snapshot" else f" · {template.source}"))
        if template.text is None:
            st.error("Not available in snapshot or git.")
        else:
            st.code(template.text, language=None, wrap_lines=True)
