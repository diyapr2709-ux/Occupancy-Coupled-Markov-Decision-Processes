---
description: Audit every number and claim in a draft section
argument-hint: [file-or-section]
allowed-tools: Read, Grep, Glob
---
Audit: $ARGUMENTS

Run two agents and report both, separately, without reconciling them:
- run-auditor, for provenance of every number.
- adversarial-reviewer, for the claims themselves.

Then list, in priority order, what must change before this is shared.
