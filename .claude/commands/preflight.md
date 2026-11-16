---
description: Run the four pre-flight correctness checks before any training
allowed-tools: Read, Grep, Glob, Bash(.venv/bin/python *), Bash(git status *)
---
!`git status --short || true`

Run all four pre-flight checks and report a table of pass or fail with evidence.
Do not fix anything; report only.

1. Observation contents: confirm the agent observation carries the pending occupancy
   (the last `delay` quarters of prescribing not yet visible to the kernel), not only
   the current occupancy. Print the actual shape and contents for one step of the
   environment built by experiments/train_agent.py:make_env. If the stack is absent,
   stop and say the current results describe a partially observed problem.
2. Entropy target scaling: find where the target is set (ocmdp/agents/oc_sac.py) and
   report whether it scales per agent as c*log(M) and c*log(K), or is shared and flat.
3. Normaliser validity: find every use of the planner as a denominator and report which
   ones run at sigma greater than zero, and with which replanning interval.
4. Planner tractability: run .venv/bin/python experiments/planner_restarts.py at the
   largest instance in use and report the spread over planner seeds.

Finish by delegating to the theory-checker agent to confirm the two-drug closed-form
result still reproduces.
