---
name: pr-review-watch
description: Watch a pull request for incoming review feedback, then triage and act on it. Use whenever the user asks to poll/watch/check a PR for a review, wait for CI or a reviewer, "let me know when the review comes in", "respond to the review comments", "address the feedback on the PR", or handle change requests on a PR they own. Verifies each review claim against the code before acting, applies the ones that are right, and escalates disagreements to the user instead of silently complying or silently ignoring. Never posts comments, replies, or reviews on the PR itself — that is always the human's to do.
---

# Watching and answering PR review feedback

## Two rules that override everything else

**1. You never post on the PR.** No comments, no replies, no reviews, no resolving threads, no dismissing reviews. All of that is the human's. You may *draft* reply text for them to post — drafting is fine, posting is not.

The reason isn't noise, it's honesty: an agent commenting under the author's account makes a PR look like a human weighed the review when none did. Keeping the poster human keeps the record truthful about who actually decided. Attribution notes are a weaker fix, because they can be forgotten — a ban can't be half-followed.

**2. Never silently comply, and never silently ignore.** A review comment has exactly three honest outcomes, all of which end in a report to the user rather than a post:

1. **It's right** → fix it, push, and tell the user what changed and where.
2. **It's wrong, or you disagree** → **stop and bring it to the user.** Don't quietly skip it, and don't argue with the reviewer anywhere.
3. **It's a question, not a change request** → answer it *to the user*, who can relay it.

Reviewers are often right, and bot reviewers are wrong often enough that compliance-by-default is a real hazard: it produces confident code changes justified by nothing. Verify every claim against the actual code before you touch anything.

## Step 1: check whether the feedback is already there

Do this before arming any watcher. Reviews frequently land before the user asks you to watch for them, and arming a monitor for a past event means waiting forever.

```bash
gh pr view <N> --repo <owner/repo> \
  --json state,reviewDecision,mergeable,reviews,comments,statusCheckRollup
```

`reviewDecision` is the fast answer: `APPROVED`, `CHANGES_REQUESTED`, `REVIEW_REQUIRED`, or empty. If feedback exists, skip to Step 3.

## Step 2: waiting, if it genuinely hasn't arrived

Pick by how many notifications you need:

**One notification when the review lands** — Bash with `run_in_background` and a loop that exits on the condition. This is the common case.

```bash
until [ -n "$(gh pr view <N> --repo <owner/repo> --json reviewDecision -q .reviewDecision)" ]; do
  sleep 60
done
```

A caveat on that condition: a **comment-only** review does not set `reviewDecision`, so a loop waiting on it alone can hang after the reviewer has actually spoken. When the reviewer is a bot with its own check, wait on the check leaving `pending` instead, then read the reviews:

```bash
bucket=$(gh pr checks <N> --repo <owner/repo> --json name,bucket \
  -q '.[] | select(.name=="claude-review") | .bucket')
```

**One notification per new comment, indefinitely** — the `Monitor` tool with `persistent: true`, polling `gh api repos/<owner>/<repo>/issues/<N>/comments?since=<ts>` and emitting a line per new comment. Only worth it for an active back-and-forth.

Cadence: 60s+ for a bot review (usually 1–5 min), 10–30 min for a human. Don't poll a remote API faster than 30s. If you're in a `/loop`, use `ScheduleWakeup` with a delay matched to the reviewer, not a tight poll.

Also watch CI: a red test check is feedback too, and usually more urgent than a comment.

## Step 3: gather every feedback surface

Review bodies and inline comments live in different places. Missing the inline ones is the most common failure:

```bash
# Review summaries + verdicts. Read the NEWEST, not the first — a push
# invalidates the prior review but leaves it here as DISMISSED, so
# .reviews[0] is often a stale verdict on an older commit. Every entry
# carries .commit.oid; check it against current HEAD before trusting it.
gh pr view <N> --repo <owner/repo> --json reviews \
  -q '.reviews[] | "[\(.submittedAt)] \(.author.login) \(.state) commit=\(.commit.oid[0:7])"'

# Inline (line-anchored) review comments — NOT in the above
gh api repos/<owner>/<repo>/pulls/<N>/comments \
  --jq '.[] | {path, line, user: .user.login, body}'

# Top-level PR conversation
gh pr view <N> --repo <owner/repo> --json comments

# Failing checks
gh pr checks <N> --repo <owner/repo>
```

## Step 4: triage each item

Sort every item into one of four buckets, and say which is which when you report:

| Bucket | Meaning | Action |
|---|---|---|
| **Blocking** | Real defect, or reviewer explicitly requested a change | Verify, then fix |
| **Non-blocking but right** | Correct observation the reviewer didn't block on | Usually fix — cheap and it's true |
| **Wrong** | The claim doesn't hold against the code | **Escalate to the user** |
| **Judgment call** | Correct but trades off against a deliberate decision | **Escalate to the user** |

An approval with a minor observation attached still deserves the fix if the observation is correct. "Non-blocking" describes the reviewer's stance, not whether it's true.

## Step 5: verify before you change anything

For each item, actually go read the code path it names. Confirm:

- The claim is true of the current code (not of an earlier revision, and not of code that only looks similar).
- The failure it describes can actually occur — trace the inputs. A reviewer flagging a branch as reachable is a claim to check, not a fact; if the state it describes is impossible, that's bucket 3 or a no-change with recorded reasoning.
- The fix doesn't undo something deliberate. Check `CLAUDE.md`, nearby comments, and the commit that introduced the line; a comment explaining *why* is a strong signal the reviewer missed context.

When you decline a change because the state it describes can't occur, give the user the reasoning explicitly — not just "not changing it". They're the one who'll defend it to the reviewer, so hand them the argument, not the conclusion.

## Step 6: make the changes

- One commit per coherent piece of feedback, message naming it as review follow-up.
- **Re-run the test suite after each change** (see below for this repo's command).
- Add or update a test whenever the feedback was about behavior — a fix without a test invites the same comment next time.
- Push normally. **Never force-push a branch someone else may have committed to**; if history needs rewriting, ask first.

## Step 7: report to the user — they post, not you

Give them one consolidated account, item by item, so they can act on the PR without re-reading it themselves:

- **Fixed** → what changed, and the commit SHA.
- **Declined** → the reasoning, in enough detail that they can post it as-is if they choose.
- **Question** → the answer.
- **Needs their decision** → see the escalation section below.

Only report something as fixed if it is actually fixed and verified. If a fix is partial, say which part.

If a reply to the reviewer looks warranted, offer the drafted text and let them decide whether to post it. Don't post it yourself, and don't resolve or dismiss anything on the PR.

## Escalating to the user

When you hit bucket 3 or 4, stop and report — briefly, with a recommendation, not a survey:

1. What the reviewer asked for (quote the relevant line).
2. Why you think it's wrong, or what the trade-off is.
3. What you'd do, and what it costs either way.

Then wait. Don't push a change to a disputed point before the user answers. If several items are disputed, batch them into one message rather than interrupting repeatedly.

## In this repo

- **Tests**: `pytest` from the repo root — offline, a few seconds. `CLAUDE.md` requires running it after any change under `shared/`, `sdf_pipeline/`, `dad_pipeline/`, `evals/`, and again before each commit and push. A behavior change without a test is a review comment waiting to happen.
- **Checks on every PR**: `smoke` (CI: compileall + pytest, no API secret exposed) and `claude-review` (the bot reviewer, ~3 min). A third `claude` check usually shows `skipping`.
- **The merge gate accepts the bot's approval.** `main` requires CI `smoke` plus one approval, and `claude[bot]`'s counts — so a same-repo PR can technically merge with no human having read it. **Never merge on that basis; merge only when the user asks.**
- **Review responses are posted by a human, never by an agent** — the team's standing convention, and the reason for rule 1 above. `CLAUDE.md`'s "PR expectations" section states the same rule; the two must agree.
- **Claude-authored GitHub content must say so.** PR *descriptions* written by Claude from this account need an explicit callout naming Claude as the author, e.g. a `> [!NOTE]` line at the top. (Comments don't arise — you don't post them.)
- **PR descriptions need a "How to test" section** (`.github/pull_request_template.md`). `gh pr create --body` bypasses the template, so write the section in explicitly.
- **`gh pr edit` fails on this repo** — it queries deprecated Projects-classic cards and exits 1. To edit a PR description use the REST API instead: `gh api --method PATCH repos/<owner>/<repo>/pulls/<N> -F body=@<file>`.
- **Local suite failures may be environmental**, not yours: `tiktoken` and `wordfreq` are declared in `requirements.txt` but are easy to have missing from an older `.venv`, which fails `tests/test_embeddings.py` and part of `tests/test_audit_dad.py` on import. Check whether the failures predate your change before chasing them.

## Gotchas

- **`gh` and sandboxing.** If `gh` fails with a TLS/certificate error (e.g. `x509: OSStatus -26276`) or a spurious "token is invalid", that's the sandbox blocking the system trust store, not broken auth. Retry the same command with the sandbox disabled. Don't send the user off to re-authenticate over this.
- **New pushes can invalidate an approval.** After pushing follow-up commits, re-check `reviewDecision` rather than assuming the earlier approval still stands — a fix that addresses a review will typically dismiss that very review.
- **A re-review may be needed.** A bot reviewer triggered by pushes re-runs itself; a human who requested changes needs re-review requested explicitly.
- **Silence from CI is not success** — a check that never started looks the same as one that passed if you only test for absence of failure. Read the actual conclusions.
- **Working in a git worktree?** Run everything from the worktree directory, and never use bare `git stash`/`git stash pop` — the stash stack is shared with the main checkout and other worktrees.
