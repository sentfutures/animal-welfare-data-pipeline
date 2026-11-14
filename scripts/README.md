# scripts/

Operator scripts. The repository's `CLAUDE.md` keeps the rules the CI sessions themselves follow
("When running in CI"); this README holds the operator runbook for the claude-fix pipeline.

| Script | What it does |
|---|---|
| `kickoff.sh` | Files `claude-fix` issues from a committed audit in `code_quality/` (see below) |
| `print_full_sdf_request_a_stage_would_send_for_one_document.py` | Builds and prints the full request an SDF stage would send for one document, from the step catalogue (`sdf_pipeline/pipeline_steps.py`) |
| `regenerate_sdf_prompt_equivalence_recorded_requests.py` | Rewrites `tests/fixtures/sdf_prompt_equivalence/`; only for a pull request that changes requests on purpose (see `tests/CLAUDE.md`) |
| `regenerate_sdf_audit_equivalence_recorded_requests.py` | Rewrites `tests/fixtures/sdf_audit_equivalence/`; same rule |
| `regenerate_sdf_resume_from_main_recorded_runs.py` | Makes the resume-from-main recordings; must be run on unmodified `origin/main` code |
| `regenerate_dad_request_settings_equivalence_recorded_requests.py` | Rewrites the DAD request-settings equivalence recordings; same rule as the other regenerators |

## Fixing audit findings (the claude-fix pipeline)

Committed audits live in `code_quality/`; the claude-fix pipeline turns their findings into
reviewed PRs on the maintainer's Claude subscription. The operator loop:

1. **File the issues**: `scripts/kickoff.sh --dry-run` prints the would-create table without creating anything; drop the flag to file for real. Defaults to the pinned ledger's high-severity findings; `--min-severity medium` widens the net, `--report <path>` targets a future audit's JSON, `--limit N` caps the batch. Re-running is idempotent — existing `[CQ …]` titles are reported as `exists`, never duplicated. Issues arrive labeled `claude-fix-ready` plus exactly one `tier-*` (never `tier-max`, which is a human-only escalation tier).
2. **Arm one**: add the `claude-fix` label (`gh issue edit <n> --add-label claude-fix`). That label is the only trigger, and deliberately the only throttle. Arm issues that touch the **same files serially** — wait for the previous PR to merge; kickoff's `batch` column encodes the intended order — because nothing in the pipeline serializes overlapping branches, and concurrent PRs on shared files end in merge conflicts a human has to untangle. **Disjoint issues can be armed together**: the repo-wide 2-concurrent-Claude-jobs busy gate and the retry cron pace them automatically. For kickoff-created issues carrying an `after=<issue>` marker, same-cluster ordering is enforced by the pipeline itself — a successor defers (`claude-busy-wait`) until its predecessor closes — so arming a whole kickoff batch at once is safe; the serial-arming rule above matters only for issues without that marker.
3. **What happens**: two-phase tiers plan read-only (writes tool-blocked), post the plan as an issue comment, and stop with `needs-human` on a HIGH risk class (auth, data models/migrations, public API contracts, payments, PII — fail-closed); otherwise they implement on `claude/issue-<n>` with incremental pushes and open a PR (authored by `github-actions[bot]`) only when `compileall + pytest` is green. The auto-review approves or requests changes; `claude-review-fix.yml` addresses change requests (3-cycle cap). A human always merges.
4. **When it stops**: `needs-human` means exactly that — the newest issue/PR comment says why and how to re-trigger (clear the label, then remove **and re-add** `claude-fix`; re-adding an already-present label fires no event; `gh workflow run claude-fix.yml -f issue_number=<n>` always works). A run that exhausts its turn budget while visibly progressing (new commits pushed) **auto-continues** — at most twice, with a model-written digest comment of what got done and what remains — before paging a human; runs the digest judges stuck page a human immediately. `claude-quota-wait` / `claude-busy-wait` need no action — the 30-minute cron requeues them (8-retry cap; tier-max **plan-phase** quota failures are frozen for manual relaunch, protecting the Fable allowance). To escalate a stuck tier-heavy issue: relabel it `tier-max` and re-trigger — the planner upgrades to Fable while the existing branch and (re-generated) plan carry forward.

The full operator runbook — canary walkthroughs, the `simulate_quota` test hook, edge cases — is
in PR #118's description. The workflows are `.github/workflows/claude-fix.yml` and
`claude-review-fix.yml`.
