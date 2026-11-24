# Arena: rung 1, the reproduction of a published RL drug-cycling result

Fixed before any proposal is written. The judge scores against these and nothing else.

## The decision

Which published result is reproduced as rung 1 of the claims ladder (CLAUDE.md, Claims ladder), by which route, and against which tolerance fixed in advance. Answer explicitly:

1. The target paper and the exact reported quantities to reproduce.
2. The route: reimplement from the paper, run the authors' released code, or a combination (for example, their landscape values in a reimplemented learner).
3. The tolerance: a numeric criterion on the reported quantities, stated before any run.
4. What is done if the published numbers cannot be reproduced (CLAUDE.md: report the discrepancy and stop; do not tune the reimplementation until it matches).
5. How the result leads to rungs 2 and 3 (control, then the published setup plus occupancy coupling).

## What it affects downstream

CLAUDE.md requires rung 1 before any new method work, and rung 3 is the only comparison with prior work on shared ground. A target with no numeric results, or a route that cannot be audited, leaves every later claim without that ground.

## Facts the proposals must take as given

- The leading candidate, Weaver et al. (PNAS 2024, DOI 10.1073/pnas.2303165121), reports its base case in figures only. Stated numbers include RL beating random cycling and all two-drug cycles in 100 of 100 replicates, RL-fit beating random in 98 of 100, mean population fitness 1.41 against 1.88 for random at noise modifier 40, and edge-weight correlations 0.96 and 0.82; a measurement delay above one step leaves RL no better than random. Details and caveats: inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md (released code under GPL-3.0 with TensorFlow 2.11, no tagged release, the figure-producing commit not identified, landscapes hard-coded with one internal inconsistency, no seeds set).
- abx_amr_simulator (2026) reports no RL results. The application literature search was cut short (inbox/2026-11-23-application-queries.md lists what was not run); other candidates may exist.
- Installing or running downloaded code is allowed only after review, cloned to a separate directory (CLAUDE.md, Gathered material).

## Criteria, in priority order

1. **Shared ground.** The reproduced quantity is one the original authors report, so agreement or disagreement means something to a reader of that paper.
2. **A tolerance that can be fixed now and can fail**, on reported quantities, with the replicate count needed to apply it.
3. **Auditability.** Every step can be rerun from this repository; licences permit what is done (GPL-3.0 code may be run and read, but copying it into this repository has licence consequences).
4. **Cost** in working days, and the risk that it stalls.
5. **Use for rungs 2 and 3**: whether OC-SAC, SAC and occupancy coupling can be added to the same setup.
6. **Security:** no downloaded code executed before review.

## Hard constraints

- The tolerance is fixed before any reproduction run and never changed after.
- Installing the authors' package into this repository's environment is not required by any option; if a proposal needs their code run, it says where and how in isolation.

## What would settle parts of it empirically instead

None cheaply: the target's numbers are what they are. A proposal may name a short look at the released repository (read-only) that would settle a factual question.

## Material

CLAUDE.md; inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md; inbox/2026-11-23-application-queries.md; inbox/2026-11-23-reviewer-leads.md; inbox/2026-11-16-prior-art-antibiotic-rl.md; the plan at <a scratch file outside the repository> (its final section on the claims ladder).
