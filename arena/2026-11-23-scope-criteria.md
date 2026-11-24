# Arena: research scope after the prior-art findings

Fixed before any proposal is written. The judge scores against these and nothing else.

## The decision

Which claims the project pursues from here, in what order, and against which comparators. Answer each of these explicitly:

1. Keep, narrow or drop the claim that entropy over resistance mechanisms preserves drug efficacy better than entropy over drugs. If narrowed, state the exact claim.
2. What carries the paper: the control theory and phase diagram of occupancy-coupled MDPs (when acting on the observed state beats a committed schedule), the entropy comparison, something else, or a combination, and in what order.
3. The comparator list for any entropy test and for the late completeness batch (the plan's ranks 3 and 7), each with its strongest published version and whether reference code exists.
4. Whether agents in the main experiments are given the true cross-resistance matrix (as drug embeddings and as the observation map), or learn without it.
5. Whether to build a reference that is optimal under noise for small cells (dynamic programming on a discretised state, or scenario-tree MPC), given that the current planners are certainty-equivalent.
6. If an entropy test remains: the instance family, including which measured cross-resistance source and which grouping rule fixed in advance.

## What it affects downstream

Every preregistration, the build order in the plan doc's revision section, compute, and the paper's framing. A wrong scope costs weeks.

## Facts the proposals must take as given

- Two identities are verified numerically on the implementation (decisions/2026-11-23.md): with exactly interchangeable drugs and the current tied networks, SAC equals OC-SAC plus a reward bonus alpha(s) log n_class(a), and OC-SAC's entropy is KL-regularisation toward a class-balanced prior. The logged tie on equal classes is therefore structural.
- Under the protocol's target rule (c log M for OC-SAC, c log K for SAC), SAC's target is unreachable for c below log n_min / log K; logged temperatures of both agents sit at the clamp floor at evaluation states.
- Prior art filed in inbox/ (read the files): Baram et al. UAI 2021 (MinRed, entropy over redundant-action classes), Tennenholtz et al. ICML 2023 (history-dependent dynamic contexts), Rank et al. UAI 2024 (gradually shifting performative RL), rebounding bandits (NeurIPS 2021), linear bandits with memory, abx_amr_simulator (2026), decoupled regularisation (ICLR 2024).
- An exploratory dev map of the (delay, memory, noise) phase diagram exists (runs/phase/phase_20261123_190013_15752.json). A blind derivation of its boundary is in progress and must not be influenced, so proposals must not state targets for that derivation.

## Criteria, in priority order

1. **Novelty that survives a hostile reader of the filed prior art.** Each claim kept must say what remains new against the closest paper, in one sentence.
2. **Falsifiability and severity.** Each claim has a preregistrable test that can fail, with power reachable inside the budget: 40 launches a day, CPU on Modal, a test tier of 100 seeds (seeds.yaml), and the measured seed-chaos noise (paired standard deviations in the runs/ reports).
3. **Value at NeurIPS main-track standard per unit of work.**
4. **Survives null results.** The plan yields a publishable statement whichever way each test comes out.
5. **Comparator strength.** The strongest honest version of each comparator a reviewer would require, with a source; no comparator weakened to make a method win.
6. **Compliance with CLAUDE.md**, read it: the claims ladder (rung 1, reproduction, before new method work), no search over instances, seeds, thresholds or metrics, preregistration before confirmatory runs.

## Hard constraints

- No edits by an agent to ocmdp/eval/, ocmdp/envs/, ocmdp/core/, seeds.yaml, budget.yaml, CLAUDE.md or .claude/; anything needing them is a step for the owner.
- No claim worded beyond the rung of the claims ladder it stands on.
- The objective is to establish or refute claims at stated power, never to make a method win.

## What would settle parts of it empirically instead

Sub-questions 4 and parts of 6 are partly empirical (for example, one dev run of both agents with raw K-way heads and drug-indexed observations). A proposal should name each sub-question that an experiment under ten minutes or one dev launch could settle, and specify that experiment, rather than argue it.

## Material

CLAUDE.md; the plan at /private/tmp/claude-501/-Users-diyap-Occupancy-Coupled-Markov-Decision-Processes/dd0e1ca7-75ac-4fdd-9fa8-5a9faedfa4d9/scratchpad/plan-doc.txt; decisions/2026-11-16.md and decisions/2026-11-23.md; every file in inbox/; README.md; ocmdp/ and experiments/ as needed; the reports in runs/.
