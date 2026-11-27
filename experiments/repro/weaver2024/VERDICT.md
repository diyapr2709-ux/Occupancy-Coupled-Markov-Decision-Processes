# Rung 1 verdict: Weaver et al., PNAS 2024 (DOI 10.1073/pnas.2303165121)

**Rung 1 not reproduced: the published result is not specified precisely enough to fix a reproduction target.**

Recorded 2026-11-26 under `arena/2026-11-23-reproduction-outcome.md`, on the owner's decision of that date. The evidence is the read-only look in `SPEC.md`. Nothing from the authors' repositories was installed, imported or run, and no reproduction run was made. This verdict says that a reproduction target could not be fixed from the published material and the pinned code. It says nothing about whether the published result is right.

## What is pinned and what is not

- **Pinned.** The reproduction outcome pins `evo_dm` at commit `16335ce` (16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0) and `rl_cycling` at commit `d68cf07`. Every file and line cited below is at those commits.
- **Not known.** Which `evo_dm` commit produced the published figures.
  - The batch scripts install `evo_dm` from its default branch at job time (`rl_cycling` `batch/test_harness_fit.sh:21`, and the same line in every `batch/*` file).
  - No commit hash is recorded with any result.
  - The result files and figure caches were not released.

## Blockers

Each blocker gives the figure panel, the files and lines, and the failure. Statements about R behaviour come from reading the code. They were not executed.

1. **Mixed scales in the base-case density panel: a stop for the two-drug comparison.**
   - **Panel.** Fig. 2 base-case density panel.
   - **The overlay.** RL-fit, RL-genotype, random and MDP are plotted beside every two-drug cycle on one x-axis, fixed at 0 to 3.3 (`rl_cycling` `R/figs.R:119-123`, axis at `R/figs.R:124`).
   - **At the pinned commits the two sides use different scales:**
     - the RL, random and MDP paths go through `evol_deepmind`, whose `normalize_drugs` defaults to True (`evo_dm` `exp.py:14`), and no harness overrides it;
     - the two-drug cycles go through `policy_sweep`, whose default is False (`exp.py:388`), and its caller does not override it (`R/prep_figs.R:663`).
   - **Normalisation.** It maps each drug's 16 values to (f − min)/(max − min) (`evol_game.py:428-435`).
   - **Consequence.** The reproduction outcome makes a two-drug comparison that mixes scales a stop. The comparison "RL beats every two-drug cycle" (gate L1's two-drug arm) has no single-scale target.
2. **The base-case scale cannot be identified.**
   - **Evidence for the normalised scale.** At `16335ce` the harness path is normalised (item 1 above), so no condition can average above 1.
   - **Evidence for the raw scale:**
     - the values 1.88 (random) and 1.41 (RL-fit) at noise modifier 40, reported by the authors (submission manuscript line 144), exceed 1;
     - the density axis runs to 3.3.
   - **What `evo_dm`'s history shows** for the measured-landscape branch of `evol_deepmind`:
     - before `9865458` (2023-02-23), normalisation was forced off;
     - from `9865458` to `7aa4fdc` (2023-06-05), it was forced on;
     - after `7aa4fdc`, it follows the argument, which defaults to on.
   - **Consequence.** Normalisation changes the RL reward and the MDP objective, so gates E2, L1 and L2 have no fixed target scale.
3. **The DQN's update budget cannot be identified.**
   - **At `16335ce`.** `train()` is called every 100 decision steps (`evo_dm` `learner.py:525-527`) and does nothing below 1 000 stored transitions (`learner.py:219-220`). That gives 91 gradient updates per replicate. This number is derived by arithmetic from code defaults (10 000 decision steps); it is not a measurement.
   - **Before `476922b` (2023-05-18, which introduced `train_freq`).** Every decision step from the warm-up on was an update: 9 001 per replicate, derived the same way.
   - **Consequence.** The two readings differ by a factor of about 99 (derived from the two counts above, not measured). "The unchanged published DQN" has no fixed definition, and neither does the budget a starved control would take 10% of.
4. **As read, the committed harness would not run to completion against the pinned `evo_dm`.**
   - **`R/test_harness.R:60`** passes a second named argument to the output-cleaning function, which takes one (`R/clean_evol_out.R:237`). In R that is an "unused argument" error.
   - **`R/test_harness_fit.R:48` and `R/test_harness_noise.R:51`** call the per-episode policy sweep.
     - At `16335ce`, `evol_deepmind` records implied policies only when asked (`exp.py:26`, `learner.py:525-531`, `learner.py:547-550`), so the sweep returns an empty list.
     - `R/clean_evol_out.R:56` then loops over 1:0 and indexes element 1 of that empty list.
   - **The figure code** reads policies from every base-case result file (`R/prep_figs.R:112-127`, default `policy_exists = TRUE` at `R/prep_figs.R:214`). Those exist only when policies were always recorded, which was true before `476922b`.
   - **Consequence.** The outcome's route, "the values on the code path the authors' harness executes", cannot be followed at the pinned commits.
5. **E3 drops out: the origin and scale of 1.88 cannot be identified.**
   - **No source for the values:**
     - the committed noise harness runs only modifier 100 (`R/test_harness_noise.R:127`), so no committed code produces a modifier-40 result;
     - the noise panel draws the difference random − RL-fit, not the two means (`R/figs.R:91-96`);
     - no committed code prints 1.88 or 1.41.
   - **How they were aggregated** (mean of episode means, or mean over replicates) is not found.
   - **The panel's letter** is cited three ways: the manuscript calls it Fig. 2D, the published text Fig. 2C, and the code's tag is B.
   - **Consequence.** As the reproduction outcome provides for this case, E3 drops out, recorded before any run. L3, the comparison against 1.41, becomes reported-only.

## Quantities

Nothing was run, because no target could be fixed. "Ours" is therefore "not measured" throughout, and there is no interval.

| Gate | Theirs (reported by the authors, as the reproduction outcome states them) | Ours | Tolerance (reproduction outcome) | Status |
| --- | --- | --- | --- | --- |
| E2, the MDP uses cefotaxime in more than half the steps | stated in the text | not measured | exact | target scale not fixed (blocker 2) |
| E3, random cycling against 1.88 | 1.88 | not measured | 0.005 + 3 σ_ep/√n | dropped (blocker 5) |
| L1, RL-genotype beats random and every two-drug cycle | 100/100 | not measured | Fisher exact | stopped: mixed scales (blocker 1); scale and budget not fixed (blockers 2, 3) |
| L2, RL-fit's measurable decrease against random | 98/100 | not measured | Fisher exact | scale and budget not fixed (blockers 2, 3) |
| L3, RL-fit at modifier 40 | 1.41 | not measured | ±0.10 | reported-only (blocker 5) |

## Consequences, as the owner decided on 2026-11-26

- Rungs 2 and 3 (the control on the published setup, and the bridge) are not claimed.
- Claims are worded on this project's own benchmark as "outperforms the strongest baselines that could be constructed, tuned at parity".
- Claim A's test launch is no longer waiting on this rung. It still needs its committed preregistration and the owner's `approvals/<slug>.ok`.
- The track stops here. Its authors' code is not debugged or run.
