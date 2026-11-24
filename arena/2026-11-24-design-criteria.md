# Arena: research-design decisions left open by the 2026-11-24 plan revision

Fixed before any proposal is written. The judge scores against these and nothing else.

## The decisions

Answer each separately, concretely enough to write into a preregistration or a code change. Keep each answer short.

1. **Rules every entropy preregistration carries.** Decide all four:
   - **A "not tested" branch.** If either agent's selected temperature is 0, or the entropy bonus is too small a share of its objective, both agents optimise nearly or exactly the same objective. An equivalence verdict then cannot fail. State the threshold.
   - **A matched-temperature arm at test.** Each agent also runs at the other's selected temperature, so the entropy space can be separated from the temperature level.
   - **The selection statistic.** Selection on val uses the same statistic as the confirmatory test. IQM over 10 seeds hides non-learning runs.
   - **Non-learning runs.** How they are counted and reported.
2. **Test-tier capacity.** The test tier holds 100 seeds (`seeds.yaml`, owner-only). Per-report paired SDs between the agents are 0.0024 to 0.026; pooled over three reports they are about 0.018, with an upper bound of about 0.030 (derived, from superseded-design reports). Decide one:
   - widen the test tier (the owner's edit);
   - set equivalence margins at 0.01 or more;
   - restrict equivalence claims, for example the phase-boundary claim to noise of 0.2 or less.
3. **The noise-optimal reference.** The every-quarter replanner is certainty-equivalent MPC, not an optimum. A dynamic program on a discretised state is feasible only for small cells (2 mechanisms, delay up to 2). Decide where it lives: `ocmdp/baselines/` (owner approval per edit), `ocmdp/theory/`, or `experiments/`. Decide its scope, how it is verified, and whether it must exist before the phase-boundary preregistration.
4. **Claim B2: structure-aware entropy on a measured, partially overlapping matrix.** Decide:
   - the primary metric: expected failure rate, or efficacy at the horizon;
   - the one entropy and kernel. The candidates are a cosine-kernel similarity entropy, which a scratch check found not concave on random matrices; the code's Gaussian similarity kernel, which has a free bandwidth; and the pushforward through row-normalised C;
   - whether drug-entropy SAC must also get C as input, to remove a privileged-information confound;
   - which comparators are required and which are ablations;
   - what happens to the reserve critic, whose dual variable is 0 in every logged window.
5. **Rung 1, reproducing the 2024 PNAS drug-cycling DQN.** Decide:
   - the minimum gate if the published random-cycling expectation (1.88) cannot be traced to its source;
   - the Gate-0 additions: greedy's exact status on the L1 test, a positive control the MDP policy must pass, and a starved-budget DQN;
   - the rung-3 estimand: the entropy-space effect with coupling minus the effect without it.
6. **Collateral sensitivity** (negative cross-resistance), which the dynamics floor at 0. Either it becomes a change to `ocmdp/core/` (the owner's), or it is a stated limitation.
7. **Sequencing.** Decide:
   - whether the phase-boundary (Claim A) planner work runs alongside rung 1 or after it;
   - when the blind derivation of the phase boundary is restarted (the spend limit cut it off before it delivered predictions);
   - what the derivation agent may and may not read.

## What it affects downstream

Every confirmatory test, its power, its comparators, and whether the paper's spine reaches the main-track bar.

## Facts the proposals must take as given

- **Scope.** The adopted outcome is in `arena/2026-11-23-scope-outcome.md`: Claim A, the control characterisation of occupancy-coupled MDPs, is the spine. B1 is a check of a known identity. B2 is structure-aware entropy on a measured matrix.
- **Prior work around Claim A.** arXiv 2610.33412 derives an open-loop, certainty-equivalent and optimal-control regime map for linear-quadratic systems with action-dependent observations; it is RELATED, not PRE-EMPTS. The 2011 informed-switching study is a BASELINE: feedback drug switching beats fixed cycling in a stochastic ward model (`inbox/2026-11-24-*`).
- **MinRed SAC** (UAI 2021, arXiv 2102.11329) grades redundancy by proximity of induced next-state transitions, using a learned inverse model.
- **The plan revision of 2026-11-24** is the current order of work. Its text is the section "Revision, 2026-11-24 (plan)" of the project's plan document, and was given to the proposers as a file.
- **Seeds are draws.** Two identical pinned Modal launches trained identically on 7 of 10 jobs.
- **Locked.** `CLAUDE.md`, `.claude/`, `seeds.yaml`, `budget.yaml`, `ocmdp/eval/`, `ocmdp/envs/` and `ocmdp/core/` are owner-only, and `ocmdp/baselines/` and `ocmdp/theory/` need the owner's approval per edit.

## Criteria, in priority order

1. **Severity.** Each test can fail, and its failure is reported as a result. No rule converts a null into a win.
2. **Strength of comparators.** Each claim faces the strongest honest version of every required comparator, tuned at parity.
3. **Fixed in advance.** Thresholds, margins, metrics and seed counts are stated now or in a preregistration, before any tier they govern is run.
4. **Power.** Every planned test is sized from the measured spreads, labelled as derived where they come from superseded-design reports, and the proposal says what it cannot resolve.
5. **Main-track value per unit of work.**
6. **Within authority.** Owner acts are named with the exact change.

## Hard constraints

- No rule chosen because it makes one method win.
- No new search over instances, seeds, thresholds or metrics after seeing outcomes.

## What would settle parts of it empirically instead

Whether the cosine-kernel entropy is concave on the 22-drug matrix is a computation, not a judgement; a proposal may make its choice contingent on it. The same holds for whether greedy passes L1 in the rung-1 model, which is a Markov-chain computation.

## Material

`CLAUDE.md`; `arena/2026-11-23-*-outcome.md`; `decisions/2026-11-23.md`; `decisions/2026-11-24.md`; `inbox/` (all files); `references/`; `ocmdp/objectives/pushforward_entropy.py`; `ocmdp/agents/`; `experiments/sweep_phase_diagram.py`; `experiments/score_phase_predictions.py`; the plan revision named above.
