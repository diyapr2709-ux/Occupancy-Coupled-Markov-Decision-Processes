---
description: Run one paired OC-SAC versus SAC comparison and verify it
argument-hint: [instance-name] [tier]
allowed-tools: Read, Bash(.venv/bin/python experiments/*), Bash(git status *), Bash(git log *)
---
Paired comparison: $ARGUMENTS

First read the matching preregistration and restate its decision rule. Then state the
exact command (experiments/train_agent.py with --tier, --seeds and, on the test tier,
--prereg), the seed list and the expected wall-clock cost. If the training script
cannot run the named instance yet, say so and stop.

Launch on the dev or val tier if the daily budget allows. On the test tier, launch only
if the preregistration is committed and its approvals file exists; otherwise stop and
say what is missing.

After the run finishes:
1. Confirm its report under runs/train/ records the git commit, the tier, the seed
   list, the instance parameters, the entropy target values and the preregistration
   path. That report is the run's manifest; never edit it.
2. Delegate to the stats-verifier agent to recompute everything from the raw logs.
3. Report the result against the preregistered rule, in those terms only. Do not
   describe a non-significant gap as a trend or as promising.
