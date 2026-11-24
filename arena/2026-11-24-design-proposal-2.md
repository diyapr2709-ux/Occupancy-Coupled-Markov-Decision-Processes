# Proposal 2

Every number below is either arithmetic on values the criteria file and plan already state, labelled "derived", or a configuration constant read from code. None is a new measurement.

**Two problems with the questions themselves:**
- **Decision 1's premise is only half right.** "If either agent's selected temperature is 0 … both agents optimise nearly the same objective" holds only when both bonuses are negligible. If one agent has α = 0 and the other has α > 0, their objectives differ and an equivalence test can fail. What that comparison loses is the ability to say anything about the entropy space. The rule below separates the two cases instead of treating them as one.
- **Decision 2 has a hidden tension.** Choosing a margin of 0.01 or more because it is affordable is a margin chosen for power, not for substance. A wide margin makes two one-sided tests (TOST) pass easily, so a null result turns into "equivalent", which runs against criterion 1. I therefore choose the widening option and fix the margin by a substantive rule.

---

## Decision 1: rules every entropy preregistration carries

### Recommendation

**(a) "Not tested" branch.**
- **The bound.** For agent i, let B_i = α_i (H_max,i − H_floor,i) / reward_scale.
  - Units: expected-failure-rate (EFR) units per quarter.
  - reward_scale = 10 (`OCSACConfig.reward_scale`; reward is −10 × mean failure probability).
  - H_floor and H_max come from `reachable_entropy_range`.
- **Why it is the right threshold.** A soft-optimal policy loses at most α(H − H_floor) of reward to its entropy bonus. So both agents' soft optima sit within max_i B_i of the reward optimum, and within that distance of each other. That bound is exact for the discounted objective each agent optimises; it is used here only as a screen on EFR.
- **Rule 1.** If max_i B_i < m (the preregistered equivalence margin), the tuned comparison is declared "not tested". The check runs twice:
  - from the α selected on val, before any test launch (no test seeds are spent if it triggers);
  - for learned temperatures, again from the median evaluation-time α realised on test.
- **Rule 2.** If min_i B_i < m ≤ max_i B_i (one agent effectively has no bonus), the tuned comparison is still reported as a parity result. It is labelled "entropy against effectively none". The entropy-space reading then rests only on the matched-temperature arm.
- **What this means on the current grid (derived).** At m = 0.01, a bonus counts only if α·ΔH ≥ 0.1:
  - SAC at K = 22 needs α ≥ 0.032 (ΔH ≤ log 22);
  - the mechanism-space agent at M = 6 needs α ≥ 0.056;
  - so the candidates 0.01 and 0.03, and the clamp floor (about 6e-6), fall under the bound.
- For B1, the plan's existing condition (model-predicted gap ≥ 3m) applies as well.

**(b) Matched-temperature arm.**
- At test, agent A also runs at α_B and agent B at α_A. Both are fixed temperatures; for a learned temperature, the val median of the realised evaluation α is used.
- The arm uses the same seeds, resets, replay order and evaluation episodes.
- It is skipped when α_A = α_B.
- The primary verdict comes only from the tuned comparison. The matched contrasts each get an interval and qualify the sentence; they never replace it. The interpretation table is written in the preregistration:
  - tuned effect present, both matched effects null → "temperature level, not space";
  - tuned effect null, matched effects present → reported as such, with no claim.
- One stated limitation: "matched" means the same price per nat, while the two agents' reachable ranges differ.

**(c) Selection statistic.**
- Each agent takes the candidate with the lowest arithmetic mean EFR over the val seeds and every preregistered cell, with all runs included. This is the per-agent component of the paired mean difference that the confirmatory paired t uses.
- The interquartile mean (IQM) and runs_learned are reported beside it, never selected on.
- Ties are broken as already adopted.
- Owner edit to the adopted CLAUDE.md replacement text: change "the best interquartile mean of the preregistration's primary metric" to "the best mean of the preregistration's primary metric, the statistic its confirmatory test uses, with the interquartile mean and runs learned reported beside it".

**(d) Non-learning runs.**
- **Definition, fixed in each preregistration.** A run has learned if its EFR is at or below EFR_uniform − 0.5 (EFR_uniform − EFR_ref).
  - EFR_ref is the dynamic-programming reference (DP) where it exists. Otherwise it is the planner at σ = 0, or the certainty-equivalent (CE) replanner under noise.
  - At K = 22 the planner may serve as reference only after planner restart agreement passes; otherwise the tuned rotation is the reference.
- **Never excluded.** Non-learning runs enter the primary mean at their actual values.
- **Reported:**
  - per agent, k/n with a Clopper–Pearson 95% interval;
  - for the pair, an exact McNemar test on discordant seeds, reported as a learning-reliability result that is not part of the primary verdict;
  - the contrast restricted to seeds where both learned, labelled descriptive only.

### Against the criteria
- **Severity.** "Not tested" is decided by α and computed ranges before test data exist, so it cannot be chosen after seeing a gap. Non-learning runs cannot be dropped to rescue a mean.
- **Comparators.** The matched arm stops a temperature-level advantage from being credited to the entropy space.
- **Fixed in advance.** Every threshold is a formula of quantities logged before test.
- **Power.** The matched arm doubles compute but uses the same n.
- **Main-track value.** These are the rules a hostile reviewer asks for first.
- **Authority.** The only owner act is the CLAUDE.md text edit above.

### Strongest objection
B is a bound on the optimum, but learned agents are not at the optimum. Agents with B < m can still differ through optimisation, for example in learning reliability, and "not tested" discards that.

Not fatal. That difference is still reported, through the McNemar test and the descriptive contrast. It just cannot be worded as an entropy-space effect, which is correct. Selecting on a mean over 10 bimodal val seeds is noisy, but the noise hits both agents equally.

### What would show this is wrong
- On dev, paired differences at B < m that routinely exceed m in the converged (rank 4) regime would mean the bound is not a useful screen.
- A dev check in which the mean and IQM pick different candidates far more often than seed noise predicts would mean the selection statistic needs more seeds.

### Prior work
- IQM, probability of improvement and non-learning reporting: NeurIPS 2021, arXiv 2108.13264.
- Seed power analysis: arXiv 1806.08295.
- Tuning and selection protocol: JMLR 2024, arXiv 2304.01315.
- Entropy coefficient as a confound: ICML 2019, arXiv 1811.11214.
- TOST: J Pharmacokinet Biopharm 15 (1987).

---

## Decision 2: test-tier capacity

### Recommendation
- **Widen the test tier (owner edit).** In `seeds.yaml`, change `test: [1000, 1099]` to `test: [1000, 1299]`.
  - Do it before any preregistration freezes and before any test launch.
  - The new range is disjoint from dev [0, 99] and val [100, 199]. The owner first confirms that no code derives a seed in 1100–1299 from a dev or val seed by an offset.
- **Fix margins by substance, not power.** m = 0.1 × G, where G = EFR_uniform − EFR_ref on the instance.
  - G is measured on dev episodes with reference policies only, so no agent outcome enters it.
- **Size n per preregistration.** n = ceil(8.56 σ_U² / m²), with a small-sample t correction.
  - The constant 8.56 is (z_0.95 + z_0.90)², for TOST at 80% power with a true difference of 0. The same n gives about 80% power for a two-sided test at a difference of m.
  - σ_U is the upper 95% chi-square bound of the paired SD from the rank-0 A/A test (20 dev seeds) on the current design. It is not taken from the superseded reports.
- **If n > 300,** that preregistration's equivalence branch reads "not tested at margin m", and only the two-sided test runs.
- **No restriction of the phase claim to σ ≤ 0.2.** The dev map already shows where crossings lie, so restricting the noise range now would set scope with outcomes in view.

### Against the criteria
- **Severity.** The margin is set by a reference gap that involves no agent outcome, so equivalence cannot be bought by widening it.
- **Power (derived from the superseded SDs, labelled as such):**
  - at 100 seeds the smallest resolvable m is about 0.0053 at σ = 0.018 and 0.0088 at σ = 0.030;
  - at 300 seeds it is about 0.0030 and 0.0051.
  - For Claim A, the test unit is evaluation episodes drawn from the same tier. At σ = 0.4 (paired SD 0.085, maximum 0.103, derived and not yet audited), the half-width falls from about ±0.017–0.020 at 100 episodes to about ±0.010–0.012 at 300. Planner episodes are cheap.
- **Main-track value.** This is the only option that adds power to both Claim A and the entropy tests without weakening either.
- **Authority.** One owner edit, named exactly above.

### What it cannot resolve
- Margins below about 0.005 when σ ≈ 0.03.
- The rung-3 interaction, whose SD is about √2 times a single contrast's (derived): at σ_U = 0.030 and m = 0.01, n ≈ 151.
- Anything at K = 22, where the SD is not measured.

### Strongest objection
Compute. B2 has about seven arms (the comparators plus the matched arms). At 300 seeds and about 15 minutes per job, that is several hundred CPU-hours (estimate).

Not fatal. The tier is a capacity ceiling, not a quota: each preregistration uses only the n its power formula gives.

### What would show this is wrong
- A rank-0 A/A σ_U well below 0.018 would make 100 seeds enough and widening unnecessary.
- A value of G so small that 0.1 G is below any resolvable margin would mean the equivalence claims are not viable on that instance at all.

### Prior work
- Seed power analysis: arXiv 1806.08295.
- Seed variance and reporting: AAAI 2018, arXiv 1709.06560.
- Equivalence testing and intersection-union tests: Statistical Science 11(4) (1996).

---

## Decision 3: the noise-optimal reference

### Recommendation
- **Location.** `experiments/dp_reference.py` (proceed authority), with tests in `tests/`. The Claim A preregistration records its SHA-256. The existing test-tier guard already requires a clean, committed tree. A move into `ocmdp/baselines/` (one owner-approved, byte-identical edit) is deferred to the release rung.
- **Scope.**
  - Backward induction over a discretised state: mechanism frequencies r (2 mechanisms), occupancy μ over 2 drugs, and the pending actions (delay 0–2), with multilinear interpolation.
  - Gauss–Hermite quadrature over the shock, including the environment's reflection at 0 and 1.
  - Finite horizon, DP-feasible cells only.
  - The sample-average open-loop schedule sits alongside it.
- **Verification, every check fixed before any held-out cell runs:**
  - (i) at σ = 0, small horizon, delay 0: the DP equals brute-force enumeration of all action sequences;
  - (ii) at σ = 0: the DP matches or beats the planner within a tolerance of 1e-3 EFR;
  - (iii) grid refinement: halving the spacing moves V(start) by less than 1e-3;
  - (iv) the DP's own computed value agrees with its simulated value within Monte Carlo (MC) error, which catches interpolation bias;
  - (v) with common random numbers, the DP does not lose to the CE replanner beyond a one-sided 95% MC error at any feasible cell;
  - (vi) the DP value lies between the CE replanner and the perfect-information relaxation bound.
- **Ordering.**
  - The perfect-information gate runs first.
  - Claim A is split into two preregistrations:
    - **A-boundary** tests the sign change of `planned_once_minus_greedy` against the theory-free null. It needs no normaliser and does not wait for the DP.
    - **A-decomposition** covers parts (ii) and (iii) at DP-feasible held-out cells. It freezes only after (i)–(vi) pass, using predictions hashed beforehand by the blind derivation.

### Against the criteria
- **Severity.** Checks (ii), (iv) and (v) can each fail.
- **Fixed in advance.** Both the tolerances and the hash are set before use.
- **Main-track value.** The spine is not held up by DP engineering.
- **Authority.** Only the deferred move needs the owner's approval.

### Strongest objection
A reference outside the gated directory can be edited before the freeze, after dev results are seen.

Not fatal. The hash plus the clean-tree guard freeze the exact code a confirmatory test uses, and edits before the freeze are visible in history. If the owner prefers the lock, the move into `ocmdp/baselines/` can happen at the freeze instead of at release.

### What would show this is wrong
- Check (iii) failing at affordable grid sizes would show the DP is too coarse to certify anything.
- Check (v) failing would mean either an interpolation-biased DP or a replanner that is closer to optimal than assumed.
- If the perfect-information gap is already inside the margin everywhere, the DP was unnecessary.

### Prior work
- Information relaxations: Operations Research 58(4) (2010).
- CE control and rollout as suboptimal control: standard dynamic-programming texts.
- The open-loop / CE / optimal decomposition template: arXiv 2610.33412 (RELATED).

---

## Decision 4: Claim B2

### Recommendation

**Primary metric: EFR.**
- It is the agents' own reward, so selection and test optimise the same quantity.
- Its spreads are the only ones logged (derived).
- It needs no efficacy threshold, a metric definition the owner has reserved.
- Efficacy at the horizon is logged as a secondary outcome that licenses no sentence.

**One entropy: the pushforward through row-normalised C.**
- Definition: H_C(π) = H(π C̃), with C̃[a, j] = C[a, j] / Σ_j C[a, j].
- Why this one:
  - it is concave by construction (entropy of a linear map). `reachable_entropy_range` assumes concavity, and so does the temperature target rule built on it;
  - it has no free bandwidth;
  - it reduces to drug Shannon entropy when C = I, and exactly to the class entropy when C is a hard 0/1 map, so B1 and B2 belong to one estimator family;
  - it lives in the space the dynamics act in (pressure = μC).
- **Validity check.** Any all-zero row of C is a stop.
- **The cosine-kernel H_Z.** Its concavity on the 22-drug matrix is a computation. It is run and reported, but it does not decide the choice. If it is concave, H_Z-cosine is an ablation; if not, it is excluded as an invalid temperature target. The Gaussian kernel is excluded from the primary arm because of its free bandwidth.
- **Naming.** The B2 sentence names H_C, not H_Z.

**C as input: an intersection-union test over two SAC arms.**
- Required comparators: drug-entropy discrete SAC with SD-SAC stabilisers, both (a) without C and (b) with the rows of C as fixed drug input features.
- "Lowers" is licensed only if the two-sided test rejects in the favourable direction against both arms. No multiplicity adjustment is needed under the intersection-union principle.
- "Equivalent" requires TOST to pass against both.
- The H_C agent keeps no C in its network input, as the scope arena adopted. C enters only through its entropy.

**Comparators.**
- Required (each with its own interval; the headline uses only the two SAC arms):
  - SAC without C and SAC with C;
  - MinRed SAC with a learned inverse model, the closest published graded-redundancy entropy;
  - α = 0 SAC. This arm is identical to tuned SAC if tuning picks α = 0.
- Ablations (intervals reported, no sentence licensed):
  - hard-group pushforward (the target-column map, n = (7, 5, 3, 3, 3, 1)), which tests whether the measured overlap adds anything beyond the grouping;
  - MinRed with known classes;
  - H_Z-cosine, only if the concavity check passes;
  - a misspecified-C arm: the entropy built from the other grouping rule, or from the 2014 cross-resistance-network matrix if its class list verifies, while the dynamics keep the primary C.
- PPO-entropy stays in the completeness batch.
- Every learned arm gets the same temperature candidate set, the same val seeds, the same evaluation episodes and the same configuration count.

**Reserve critic: cut.**
- It is removed from all B2 arms. The dual variable is 0 in every logged window, and an extra network shifts paired initialisation draws.
- With no preregistration using it, `ocmdp/agents/lagrangian.py`, the reserve critic component and the ξ term in the actor loss are deleted under the no-dead-code rule. They are not stubbed.
- This changes the method under test, so the owner adopts it (scope).
- It could come back only in a new preregistration whose budget is shown on dev to bind (ξ > 0), with PPO-Lagrangian as its comparator.

### Against the criteria
- **Severity.** Under the intersection-union test, the method must beat SAC even when SAC has the same information. Ablations cannot be promoted to the headline.
- **Comparators.** These are the strongest honest versions, including privileged-information parity.
- **Fixed in advance.** The entropy choice is made on concavity and parameter-freeness, not on outcomes.
- **Power.** Not measured at K = 22. Size n from the K = 22 dev A/A test using Decision 2's rule.
- **Authority.** C promotion and the NC licence remain owner items.

### Strongest objection
Circularity. The simulator and the entropy are built from the same matrix C, so the entropy is aligned with the simulator by construction. Row normalisation also discards how strongly each drug selects. A win might reflect the alignment rather than any property that transfers to real resistance data.

This hurts external validity but is not fatal to internal validity. The misspecified-C ablation measures how much of the effect survives a wrong matrix, and that number goes in the paper.

### What would show this is wrong
- The hard-group ablation matching H_C (the overlap adds nothing).
- MinRed with a learned inverse model matching H_C (knowing C adds nothing).
- SAC with C beating SAC without C by more than the H_C effect (the effect was information, not entropy).

### Prior work
- Order-1 similarity-sensitive diversity: Ecology 93(3) (2012).
- A counterexample to a recent conjecture on the concavity of similarity-sensitive entropy, with a Laplace-kernel class where concavity holds: arXiv 2601.03064 (abstract only).
- Graded redundancy (MinRed): UAI 2021, arXiv 2102.11329.
- Discrete SAC: arXiv 1910.07207 and arXiv 2209.10081.
- Data source for C: Mol Biol Evol 31(9) (2014).

---

## Decision 5: rung 1, reproducing the 2024 PNAS drug-cycling DQN

### Recommendation

**Minimum gate if 1.88 cannot be traced.**
- Gating items:
  - E2 (the decision-process policy uses cefotaxime in more than half the steps);
  - L1 and L2 (exact Fisher cut-offs);
  - both negative controls fail;
  - the decision-process positive control passes (below).
- L3 becomes a ratio test, but only if the read-only look shows that 1.41 and 1.88 come from the same harness and scale:
  - the replicate mean of RL-fit divided by the exact random expectation at noise 40 must lie with its 95% interval inside [1.31/1.88, 1.51/1.88] = [0.697, 0.803] (derived from the published ±0.10 band);
  - otherwise L3 is reported but not gated.
- The licensed sentence shrinks to the ordering claims, plus the ratio where it is gated. Rungs 2 and 3 still unlock, because they compare within this reimplementation.

**Gate-0 additions.** Gates are the published claims. The additions below qualify the sentence, and only the positive control can stop the rung.
- **Greedy.** Greedy's exact expectation is computed as a Markov chain against random and all 105 two-drug cycles before the freeze. If greedy passes L1, the rung-1 sentence must state that L1 does not separate the learner from myopic choice. A count of replicates beating greedy's exact value is reported as a characterisation, never as a gate.
- **Positive control.** The exact decision-process policy is run through the identical L1 and L2 procedure: same episode counts, same estimator, 100 harness replicates. It must pass both cut-offs. If it fails, the harness cannot detect a policy known to be better, and the rung stops and reports.
- **Starved-budget DQN.** Same configuration, with one-tenth of the full run's gradient updates; the update count is fixed by the read-only look, given `train_freq` and the warm-up. It is a dose point with a stated expectation, about 100 episodes per the published learning curve. If it passes L1 and L2, the sentence says those claims are reached at 10% of the budget.

**Rung-3 estimand.**
- θ = mean over seeds of D_s, where D_s = (Y_OC − Y_SAC) at κ* minus (Y_OC − Y_SAC) at κ = 0.
- Y is the published metric (mean population fitness), with paired-t intervals.
- One temperature per agent is selected on val by the mean over both κ conditions, so temperature cannot differ across the κ arms. The Decision 1 rules apply.
- κ* is fixed from the environment alone: the smallest grid κ at which the coupled decision-process optimum differs from the uncoupled optimum in at least 10% of reachable states. This is a computation that involves no agent.
- SDs on this metric are not measured. Size n from a dev A/A test on this setup.

### Against the criteria
- **Severity.** The positive control can stop the rung, and the starved and greedy checks can only narrow what is claimed.
- **Fixed in advance.** κ* is set without agents.
- **Authority.** The class map remains the owner's.

### Strongest objection
The ratio still leans on 1.88, so the minimum gate is weak.

Not fatal, because the licensed sentence weakens to match.

### What would show this is wrong
- A passing positive control at the Fisher cut-off would mean the harness discriminates; a failing one would mean the L1 and L2 cut-offs, at 100 harness replicates, could not detect even a policy known to be better.
- A κ* so large that D_s has no plausible room to be nonzero.

### Prior work
- Reproduction target: PNAS 2024, DOI 10.1073/pnas.2303165121.
- Decision-process cycling on measured profiles: PLoS Biol 17(10) e3000515 (2019).

---

## Decision 6: collateral sensitivity

### Recommendation
- **A stated limitation, not a core change.** The dynamics require C ∈ [0, 1] (`ocmdp/core/coupling.py`). A negative entry would break:
  - the pressure and failure-probability formulas;
  - the closed-form stationary point;
  - the mean-field identity behind B1;
  - the row-normalised pushforward;
  - every existing phase table.
- **Data step.** After the matrix is promoted and before the B2 preregistration, compute and report the floored share: the sum of |negative entries| divided by the sum of |all entries|, per selection strength.
- **Wording.** Every B2 sentence carries "with collateral sensitivity floored at zero".
- A dynamics change with negative pressure entering the reversion term would be a new scope through the owner, after B2 confirms.

### Strongest objection
The floored mass, for example from aminoglycosides under strong selection, is exactly the structure that published sequential-therapy strategies exploit. Reviewers will see the clipping as removing the main lever.

Not fatal for Claim A, which is defined on its own model class. It weakens B2's "measured matrix" selling point in proportion to the floored share, which is why that share is reported.

### What would show this is wrong
A large floored share would make B2's "measured" framing hard to defend.

### Prior work
- PLoS Biol 17(10) e3000515 (2019).
- PLoS Comput Biol 11(9) e1004493 (2015).

---

## Decision 7: sequencing

### Recommendation

**Run planner-only Claim A work alongside rung 1.**
- In scope: the perfect-information gate, the DP, the sample-average open-loop schedule, and the A-boundary preregistration and its test.
- No learned agent is involved, so this is not "new method work", which is the scope outcome's reading. The owner confirms that reading of the claims ladder.

**Restart the blind derivation now,** before any held-out cell, DP value or open-loop schedule is computed on held-out cells.
- The held-out cell list is hashed first.
- Deliveries are staged, and each is hashed on delivery:
  1. σ* and the sign beyond the crossing for `planned_once_minus_greedy`;
  2. the direction predictions;
  3. the part (ii) predictions at DP-feasible held-out cells.
- A stage not delivered within the spend cap is "not predicted", and that part is dropped from its preregistration. It is never filled in after data exist.
- The agent may read its own earlier partial notes, which were made under the same barriers.

**What the derivation agent may read and run.** It works in a separate export with no `.git`, containing only the allowed paths.
- May read:
  - `ocmdp/core/`, `ocmdp/theory/` and `ocmdp/envs/` source;
  - the source of `experiments/sweep_phase_diagram.py` and `experiments/score_phase_predictions.py`, for the contrast definitions and the scoring format;
  - the held-out list and the σ grid;
  - public literature.
- May run: closed-form and deterministic numerics only (mean-field, linearisation, moment equations, quadrature of formulas it derives itself).
- May not:
  - simulate shocks or import the environments or baselines at run time;
  - read `runs/`, `figures/`, `decisions/`, `arena/` (this file states the phase SDs), the plan doc (same), `provenance/`, `launches/`, the ledger, `tests/`, README results, or git history.

### Against the criteria
- **Severity.** The predictions are hashed before any measurement they predict exists, and enforcement is by filesystem rather than trust.
- **Main-track value.** The spine moves without waiting for rung 1.

### Strongest objection
Running Claim A alongside rung 1 stretches "work up in order". Separately, deterministic σ = 0 numerics give the derivation the noise-free contrast, which is part of the answer.

Not fatal. The noise-free value is a legitimate theory input. The test is of where the boundary lies in σ, which no σ = 0 computation reveals.

### What would show this is wrong
- The derivation's predictions matching the dev-map interpolation suspiciously closely (E − E_null ≈ 0 with tiny spread), which would suggest leakage.
- Rung 1 failing in a way that implicates shared environment code.

### Prior work
- arXiv 2610.33412 (RELATED).
- PLoS Comput Biol 7(3) e1001094 (2011), a required BASELINE.
- Search hits on feedback against open-loop control: arXiv 2603.11369, 2609.12264, 2502.09392.

---

## Owner acts named in this proposal
- `seeds.yaml`: `test: [1000, 1099]` becomes `test: [1000, 1299]`.
- The CLAUDE.md temperature-rule text: selection by the mean instead of the IQM (exact replacement in Decision 1(c)).
- Adopting removal of the reserve critic (Decision 4), which is a scope change.
- Confirming the ladder reading that lets planner-only Claim A work run alongside rung 1.
- Optional and deferred: moving the DP into `ocmdp/baselines/`.

- `arena/2026-11-24-design-criteria.md`
- `ocmdp/objectives/pushforward_entropy.py`
- `ocmdp/agents/entropy_range.py`
- `ocmdp/agents/oc_sac.py`
- `ocmdp/core/selection.py`
- `seeds.yaml`

Sources:
- [arXiv 2601.03064](https://arxiv.org/abs/2601.03064)
- [arXiv 2511.03849](https://arxiv.org/abs/2511.03849v1)
- [arXiv 1906.08325](https://arxiv.org/pdf/1906.08325)
