---
description: Write a preregistration before an experiment runs
argument-hint: [experiment-name]
allowed-tools: Read, Write, Bash(git status *), Bash(git rev-parse *), Bash(date *)
---
Write a new preregistration for: $ARGUMENTS

Create preregistrations/YYYY-MM-DD-<slug>.yaml containing, and refuse to proceed if any
is unknown rather than filling a placeholder:

- hypothesis, in one falsifiable sentence
- the primary decision rule, as a single test on a single quantity
- the equivalence margin, so a null result can be stated as equivalence
- seed count, with the detectable effect size it supports and the rho it assumes
- instance grid: every (K, M, class-size vector) with its KL value
- entropy target rule and the value of c
- floor and ceiling exclusion thresholds
- which metrics are primary and which are secondary
- git commit at time of writing

Do not commit it. Tell me the file is ready for me to commit, and that it is frozen
from that commit on: a changed rule means a new dated file.
