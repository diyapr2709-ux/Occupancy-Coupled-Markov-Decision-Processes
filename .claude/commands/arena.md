---
description: Resolve a design decision with competing independent proposals and a blind judge
argument-hint: [the decision to resolve]
allowed-tools: Read, Write, Grep, Glob, WebSearch, WebFetch
---
Decision: $ARGUMENTS

Step 1. Before spawning anything, write the decision criteria to
arena/<slug>-criteria.md and show them to me. State: what the decision affects
downstream, what would make one option clearly better, what the hard constraints are,
and what evidence would settle it empirically instead. If the last field is non-empty
and the experiment is cheap, STOP and say the experiment should be run rather than
holding an arena.

Step 2. Spawn three arena-proposer agents in ONE batch, each with the identical brief
and none of the others' output. Give each only: the decision, the relevant code or
report sections, and the criteria file.

Step 3. Collect the three proposals. Strip every signal of which agent produced which
and number them 1 to 3.

Step 4. Spawn arena-judge with the criteria file and the numbered proposals only. It
must not see this conversation or know how many models were involved.

Step 5. Report to me: the judge's ranking with its reasoning, every point on which the
proposals disagreed, and anything all three missed. If all three converged, say so and
treat the convergence as weak evidence rather than strong, since identical briefs
produce correlated answers.

Never implement the winner in the same turn. The decision comes back to me first.
