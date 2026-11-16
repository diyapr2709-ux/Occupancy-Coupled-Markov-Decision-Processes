---
description: Drive the instance-grid separation sweep
argument-hint: [arm: imbalance | scale | anchor] [tier]
allowed-tools: Read, Bash(.venv/bin/python experiments/*), Bash(git status *), Bash(git log *)
---
Sweep arm: $ARGUMENTS

For each instance in this arm of the grid, report the planned (K, M, n) and its KL
value, then the full command list and total cost. Launch on the dev or val tier if the
daily budget allows; on the test tier only with a committed preregistration and its
approvals file. Otherwise stop and say what is missing.

After the runs complete:
1. For every instance, record the paired gap divided by that instance's uniform-mixing
   EFR, and separately the realised within-class conditional entropy of trained SAC.
2. Apply the preregistered floor and ceiling exclusions, and list which instances were
   excluded and why.
3. Fit gap against KL and log K with an instance random effect. Report the KL
   coefficient and its bootstrap interval over instances. That coefficient is the
   headline number.
4. Plot the gap against both the KL bound and the realised within-class entropy. If it
   tracks the realised quantity but not the bound, say so explicitly; that is support
   for the hypothesis with a loose bound, not a null result.
