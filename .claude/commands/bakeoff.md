---
description: Resolve a doubt by competing implementations judged on a pre-fixed check
argument-hint: [the doubt to resolve]
allowed-tools: Read, Write, Grep, Glob, Bash(.venv/bin/python -m pytest *), Bash(.venv/bin/python experiments/*)
---
Doubt: $ARGUMENTS

Step 0. Triage. If a run under ten minutes, an existing test or a lookup answers this,
do that instead and stop. If it is cheap to reverse, pick and log it. If it changes a
claim or a preregistration, stop and tell me.

Step 1. Write decisions/<date>-<slug>.md with: the doubt, the alternatives, and the
check that will decide it (a test, or a metric on dev seeds with its threshold). The
check is fixed from this point. Do not state which alternative you expect to win.

Step 2. Spawn three arena-contestant agents in one batch, each with a different brief:
BUILD the leading option; OPPOSE with the strongest alternative or the null; RED-TEAM
by finding the check that would embarrass the winner. Give each the decision file and
only the code it needs. None may see the others' work. Contestants work in worktrees
branched from the last commit, so uncommitted work is invisible to them: if the doubt
depends on uncommitted changes, stop and tell me to commit first.

Step 3. Run the fixed check on every submission, on dev seeds only, within budget.

Step 4. Append to the decision file: each result, the red team's findings, and the
outcome. If one option passes and the red team found nothing fatal, adopt it and
continue. If results are tied or the red team's objection stands, write UNDECIDED and
stop for me.
