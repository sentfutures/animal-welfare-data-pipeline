#!/usr/bin/env bash
# kickoff.sh — file tiered GitHub issues from the code-quality audit ledger.
#
# Runs a headless `claude -p` (billed to your Claude subscription) that reads
# code_quality/findings_v2_*.json, re-verifies each finding against the current
# code, clusters findings that touch the same files, and creates one gh issue
# per item with a tier label (tier-light / tier-standard / tier-heavy — NEVER
# tier-max, which is a human-only escalation tier) plus `claude-fix-ready`.
#
# Nothing is armed automatically: the `claude-fix` label — which launches the
# fixer workflow — is applied by a human, one issue at a time, as the throttle.
#
# Usage:
#   scripts/kickoff.sh [--dry-run] [--limit N] [--min-severity high|medium|low]
#                      [--report <path>] [--model <model>] [--repo <owner/name>]
#
#   --dry-run        print the summary table only; create nothing
#   --limit N        create at most N issues (after clustering)
#   --min-severity   high (default) | medium | low. `info` findings are never
#                    filed (the audit treats them as observations, not defects)
#   --report         findings ledger (default: code_quality/findings_v2_2026-07-29.json)
#   --model          model for the kickoff run (default: opus)
#   --repo           owner/name (default: the current repo per gh)
#
# Requires: claude CLI (logged in), gh (authenticated), jq.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
LIMIT=0
MIN_SEVERITY=high
REPORT=code_quality/findings_v2_2026-07-29.json
MODEL=opus
REPO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --limit) LIMIT="$2"; shift ;;
    --min-severity) MIN_SEVERITY="$2"; shift ;;
    --report) REPORT="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --repo) REPO="$2"; shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

case "$MIN_SEVERITY" in
  high) SEVERITIES="high" ;;
  medium) SEVERITIES="high, medium" ;;
  low) SEVERITIES="high, medium, low" ;;
  *) echo "--min-severity must be high, medium, or low" >&2; exit 2 ;;
esac

for tool in claude gh jq; do
  command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
gh auth status >/dev/null || { echo "gh is not authenticated (run: gh auth login)" >&2; exit 1; }
[ -f "$REPORT" ] || { echo "Findings ledger not found: $REPORT" >&2; exit 1; }
[ -n "$REPO" ] || REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

# Label bootstrap (idempotent) — the fixer workflows assume these exist.
bootstrap_label() { gh label create "$1" --color "$2" --description "$3" --repo "$REPO" >/dev/null 2>&1 || true; }
bootstrap_label claude-fix        1D76DB "Arm the Claude fixer on this issue (human-applied throttle)"
bootstrap_label claude-fix-ready  C5DEF5 "Filed by scripts/kickoff.sh; a human applies claude-fix to launch"
bootstrap_label needs-human       D93F0B "Automation stopped deliberately; a human must review before anything continues"
bootstrap_label claude-quota-wait FBCA04 "Claude run hit a subscription usage limit; the retry cron will relaunch it"
bootstrap_label claude-busy-wait  F9D0C4 "Deferred by the 2-concurrent-Claude-jobs cap; the retry cron will relaunch it"
bootstrap_label tier-light        BFE5BF "Fixer tier: single phase, sonnet/medium (mechanical fixes)"
bootstrap_label tier-standard     0E8A16 "Fixer tier: plan opus/high, implement sonnet/high (default)"
bootstrap_label tier-heavy        5319E7 "Fixer tier: plan opus/xhigh, implement opus/high (structural/subtle)"
bootstrap_label tier-max          B60205 "Fixer tier: plan claude-fable-5/high, implement opus/high — escalation only, never auto-assigned"

LIMIT_RULE="There is no cap on the number of issues."
if [ "$LIMIT" -gt 0 ]; then
  LIMIT_RULE="Create at most $LIMIT issues (after clustering); pick by severity, then by batch order, and list anything cut from this run in the table as 'deferred'."
fi

MODE_RULE="After printing the table, create the issues exactly as specified."
ALLOWED_TOOLS="Read,Glob,Grep,Bash(gh issue list:*),Bash(gh issue view:*),Bash(gh search:*),Bash(gh issue create:*)"
if [ "$DRY_RUN" = true ]; then
  MODE_RULE="DRY RUN: do NOT create anything — print the summary table of what you WOULD create, then stop."
  ALLOWED_TOOLS="Read,Glob,Grep,Bash(gh issue list:*),Bash(gh issue view:*),Bash(gh search:*)"
fi

PROMPT=$(cat <<EOF
You are filing GitHub issues (repo: $REPO) from this repository's committed
code-quality audit, for an automated fix pipeline to work through.

SOURCES (read all three first):
- $REPORT — the machine-readable findings ledger. Source of truth.
- code_quality/README.md — the ground rules for agents filing issues from the
  audit. Follow its "For agents filing or resolving issues" section.
- code_quality/CODE_QUALITY_REPORT.md — the prose report; use it to map each
  finding to its report number (2.x / 3.x / 4.x) where one exists.

SCOPE:
- Only findings with severity in: $SEVERITIES. Never file 'info' findings.
- v2 ledger only; the v1 file is history.
- Re-verify before filing: for every candidate finding, Read the cited
  file:lines in the CURRENT code (line numbers drift). If the defect no longer
  exists, do not file it — record it in the table as 'skipped (no longer
  reproduces)' with one line of evidence.
- $LIMIT_RULE

CLUSTERING:
- Findings whose 'file'/'instances' overlap belong to the same batch (give them
  consecutive batch-order numbers so their PRs land serially, minimizing merge
  conflicts). Merge multiple findings into ONE issue only when the fix is
  literally the same edit.

EACH ISSUE:
- Title: [CQ <report-number>] <finding title> — e.g. "[CQ 2.2] ..." — or
  [CQ <dimension>] <finding title> if it maps to no numbered report item.
- Body, in order:
  1. The HTML marker line, exactly:
     <!-- claude-kickoff finding=<report-number-or-dimension-slug> batch=<k> after=<n|none> -->
     where after=<n> is the ISSUE NUMBER of the batch predecessor whenever
     this finding shares files with an earlier issue in the same cluster —
     create issues in batch order so that number already exists — and
     after=none for cluster-independent findings. The fixer workflow reads
     this marker and automatically defers an issue until its predecessor
     closes, so batch order is enforced even if every issue is armed at once.
  2. "## Motivation" — the finding's failure_scenario, lightly edited.
  3. "## Affected files" — the finding's file + instances, as a list.
  4. "## Acceptance criteria" — from the finding's recommendation, as concrete,
     checkable bullets.
  5. "## Verification" — how to prove the fix: the repo test gate
     (python -m compileall -q shared sdf_pipeline dad_pipeline pref_pipeline evals viewer && pytest)
     plus finding-specific checks (from recommendation/verifier_notes).
  6. A visible callout, exactly:
     "> 🤖 This issue was generated by Claude Code (scripts/kickoff.sh) from the committed code-quality audit. A human launched the run but did not author this text."
- Labels: exactly ONE tier label plus claude-fix-ready
  (gh issue create --label "<tier>" --label "claude-fix-ready" ...):
  - tier-light: mechanical, single-file, S-effort fixes (lint-ish, naming, dead
    code, small refactors, missing tests for existing behavior)
  - tier-standard: the default — multi-file, unclear root cause, moderate refactors
  - tier-heavy: cross-cutting/structural, concurrency, subtle correctness
  NEVER tier-max (human-only escalation tier) and NEVER claude-fix (a human
  arms issues manually).
- Write bodies with --body-file (Write the body to /tmp/kickoff_issue_<k>.md
  first); never inline multi-line --body strings, command substitution, pipes,
  or heredocs in Bash commands.

IDEMPOTENCY:
- Before creating each issue, search for an existing one:
  gh search issues --repo $REPO "<finding title>" --state open --json number,title
  (also try --state closed). If a match exists, skip it and mark it 'exists' in
  the table.

FINAL OUTPUT — always end with this summary table (markdown), one row per
finding considered:
| issue | title | files | tier + one-line reasoning | batch order | status |
where status is one of: created / would-create (dry run) / exists / skipped
(no longer reproduces) / deferred (over limit).

$MODE_RULE
EOF
)

echo "== kickoff: repo=$REPO report=$REPORT severities=[$SEVERITIES] dry_run=$DRY_RUN limit=$LIMIT model=$MODEL"
echo "== launching headless claude (this bills your subscription)..."

# Write tool is needed for --body-file bodies; /tmp is outside the repo, hence --add-dir.
claude -p "$PROMPT" \
  --model "$MODEL" \
  --effort high \
  --max-turns 100 \
  --add-dir /tmp \
  --allowedTools "$ALLOWED_TOOLS,Write" \
  --disallowedTools "Task" \
  | tee /tmp/kickoff_summary.md

if [ "$DRY_RUN" = true ]; then
  echo "== dry run complete (nothing created). Table saved to /tmp/kickoff_summary.md"
  exit 0
fi

echo ""
echo "== post-checks"
# Belt and braces: the tier-max rule is enforced by prompt; verify it held.
bad=$(gh issue list --repo "$REPO" --label tier-max --label claude-fix-ready --json number --jq 'length')
if [ "${bad:-0}" -gt 0 ]; then
  echo "!! WARNING: $bad kickoff-created issue(s) carry tier-max — that tier is human-only. Fix the labels before arming anything."
  exit 1
fi
echo "== issues currently labeled claude-fix-ready:"
gh issue list --repo "$REPO" --label claude-fix-ready --json number,title,labels \
  --jq '.[] | "#\(.number)\t\([.labels[].name | select(startswith("tier-"))] | join(","))\t\(.title)"'
echo "== done. Arm an issue with: gh issue edit <n> --repo $REPO --add-label claude-fix"
