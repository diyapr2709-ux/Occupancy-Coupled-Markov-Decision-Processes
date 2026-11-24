# Proposal 3

**Where the criteria pull against each other** (flagged here, not resolved by a reading of my own):
- **Claim A against the claims ladder.** `CLAUDE.md` says rung 1 "comes before any new method work", and its rung 5 is "one run per preregistration on test seeds". The ladder was written for the entropy claim, and the scope arena read Claim A's planner work as "not new method work". So it is unclear whether Claim A's confirmatory run counts as rung 5, and therefore whether it is gated on rung 1. Only the owner can say. Decision 7 proposes a reading for the owner to confirm.
- **C as a network input.** Decision 4's privileged-information question collides with the adopted scope item "no true C in the networks". I propose a narrow reversal for B2 only, and it is an owner act.
- **Two parts are computations, not judgement.** One is whether the cosine-kernel entropy is concave on the 22-drug matrix; my choice for Decision 4 does not depend on it. The other is greedy's exact status on L1; my rule for Decision 5 is written to branch on it.

Every SD below is the brief's derived figure from superseded-design reports, carried as given: 0.018 pooled, 0.030 upper bound. Nothing was recomputed from `runs/`. All seed counts are power arithmetic, not runs.

---

### Decision 1. Rules every entropy preregistration carries

**1. Recommendation**
- **"Not tested" branch.** It is settled on val, before any test run, and written into the preregistration.
  - For each arm (the tuned arm and each matched arm), compute each agent's entropy share over the last training window on val: s = α·E[H_own(π)] / (α·E[H_own(π)] + E|r|), in scaled reward.
  - The entropy-space comparison in that arm is "not tested" if either agent has α = 0, or both agents have s < 0.05.
  - If exactly one agent selects α = 0 in the tuned arm, that arm is reported as "entropy against no entropy". The entropy-space verdict then comes only from the matched arm at the nonzero α.
  - If every arm is "not tested", the preregistered result is: "at parity tuning neither agent uses its entropy term". That is a reported finding, not a missing one.
  - For B1, add the plan's model condition: the mean-field predicted gap at the fixed α must be at least 3m.
- **Matched-temperature arm at test.** On the same test seeds, agent A also runs at α_B and agent B at α_A, giving a 2×2 of entropy space × temperature level.
  - A learned temperature is matched by a fixed α equal to that agent's median learned α over the last val window. That value is frozen in the preregistration.
  - The tuned comparison stays primary. The two matched contrasts are secondary, with a step-down multiplicity correction within the secondary family.
  - Wording rule, fixed now: if the tuned effect is significant but neither matched contrast is, the sentence must say the difference is not attributable to the entropy space at matched temperature.
- **Selection statistic.** Each agent takes the candidate with the best arithmetic mean of the primary metric over its 10 val seeds and cells. Every run counts at its realised value, non-learners included. The test's paired mean difference is the difference of these means, so selection and test use the same statistic. The IQM is reported beside it, never used to select. Ties go to the earlier candidate, as already adopted.
- **Non-learning runs.**
  - Definition, fixed now: a run has learned if its evaluation EFR improvement over uniform-over-drugs exceeds half the CE replanner's improvement on the same episodes. The replanner replaces the oracle, which is not a bound under noise and whose restart agreement at K = 22 is unverified.
  - Never excluded from the primary analysis.
  - Report per agent: non-learning count with an exact binomial 95% interval; the paired 2×2 learned/not-learned table with an exact test on the discordant pairs (secondary); IQM; probability of improvement.
  - The difference among seeds where both agents learned is descriptive only, because it conditions on an outcome of treatment.

**2. Against the criteria**
- **Severity.** "Not tested" is decided on val, so no test outcome can move a comparison into or out of it. The matched arm keeps the entropy-space question testable when the tuned temperatures differ. A null in a tested arm is reported as a null.
- **Comparators.** Each agent is tuned on its own candidate set at parity. The matched arm takes away the temperature-level advantage a tuned comparator might hold.
- **Fixed in advance.** The threshold 0.05, the "learned" definition and the statistic are all stated here.
- **Power.** The matched arms reuse the same seeds, so they cost runs but not seeds. Counting non-learners in the mean makes the bimodal SD the one that governs; that is why Decision 2 sizes from the upper bound.
- **Value.** Answers the reviewer's first question: whether the effect is the entropy space or just the temperature level.
- **Authority.** The temperature-arena text pending in `CLAUDE.md` says "best interquartile mean". The owner edit replaces it with "best mean of the primary metric over 10 val seeds, every run counted, interquartile mean reported beside".

**3. Strongest objection.** The 0.05 share threshold is a convention, not derived. Evaluation is greedy (argmax; the checkpoint test reproduces "per-episode argmax evaluation"), so entropy acts only through learning, and a small share can still reshape Q near ties. Not fatal: the threshold is fixed before val, it can only declare "not tested" (it never produces a win), and the matched arm still runs.

**4. What would show me wrong**
- On dev, the matched α = 0 pair should reproduce the A/A spread.
- If arms with s just above 0.05 show paired SDs indistinguishable from A/A, the threshold is too low. The fix would be a new dated rule before val, never after test.

**5. Prior work**
- Assay sensitivity in equivalence trials (ICH E10, 2000): an equivalence verdict is uninformative without evidence that the trial could detect a difference. This is the same logic as the "not tested" branch.
- NeurIPS 2021, arXiv 2108.13264 (IQM and probability of improvement). The IQM trims 25% at each end, which is why it hides non-learners.
- AAAI 2018, arXiv 1709.06560, and arXiv 1806.08295 (seed variance).
- JMLR 2024, arXiv 2304.01315 (empirical design, hyperparameter selection).
- I found no published matched-temperature design for comparing entropy spaces.

---

### Decision 2. Test-tier capacity

**1. Recommendation.** Set every entropy equivalence margin at m = 0.01 EFR. Do not widen the test tier, and do not restrict Claim A.
- Each preregistration may use all 100 test seeds (1000–1099).
- Seeds may be reused across preregistrations. Each preregistration records which other test outcomes existed when it was frozen.
- Claim A carries no equivalence claim. Its E-statistic interval reports what it cannot resolve as "inconclusive".

**2. Against the criteria**
- **Power** (derived; paired t, n = 100, α = 0.05):

| Test | At SD 0.018 | At SD 0.030 (upper bound) |
|---|---|---|
| Smallest margin TOST can confirm at 80% power, true Δ = 0 | 0.0053 | 0.0089 |
| Seeds needed for m = 0.01 | about 30 | about 80 |
| Seeds needed for m = 0.005 | about 113 | about 313 |
| Seeds needed for m = 0.0075 | — | about 139 |
| Smallest difference two-sided superiority detects, 80% | 0.0051 | 0.0085 |
| Same, with a three-contrast step-down correction (worst case) | 0.0059 | 0.0098 |

  At the upper bound, three-way-corrected TOST needs m ≥ 0.0103, so it is marginally under 80% (about 78%). The preregistration says so.
- **What it cannot resolve.** Differences below about 0.005–0.009. At the upper SD, a true |Δ| of 0.005 gives TOST about 50% power at m = 0.01.
- **Severity.** The margin is fixed now, before any val or test run of the current algorithm.
  - The dev A/A pilot (rank 0, 20 seeds, α = 0, at K = 22) sets only the reported power, never the margin.
  - The preregistration reports m as a fraction of the planner-only gap between replanner and uniform on the B2 instance. If that gap is small, the equivalence sentence says so. m is not moved.
- **Value.** Widening the tier to reach m = 0.005 needs about 113–313 seeds per arm, times 4–6 learned arms at K = 22. The extra work does not change the sentence much.
- **Authority.** No owner edit to `seeds.yaml` is needed.

**3. Strongest objection.** m = 0.01 may be coarse compared with entropy-space effects at K = 22, which are not measured, so equivalence could be easy to declare. Not fatal: the m-to-gap ratio is reported, and the superiority test resolves about 0.005–0.01.

**4. What would show me wrong.** The rank-0 A/A SD at K = 22. Above 0.030, the tests are underpowered at m = 0.01, and that is stated rather than repaired.

**5. Prior work.** Equivalence-test sizing (Social Psychological and Personality Science 2017, 8(4)); assay sensitivity (ICH E10).

---

### Decision 3. The noise-optimal reference

**1. Recommendation**
- **Scope.** Finite-horizon backward induction:
  - 2 mechanism classes (tied drugs within a class), delay 0, 1 or 2;
  - the dev map's memory values {0, 0.5, 0.8, 0.95} and noise grid {0, 0.1, 0.2, 0.4};
  - the phase horizon of 40 quarters;
  - state = per-mechanism resistance levels × the 2^d pending stack × the occupancy memory;
  - Gaussian quadrature over the shock, interpolation on the grid.
  - It also gives the best open-loop schedule by sample-average approximation, so decomposition (i)–(iii) is available at the DP cells.
- **Verification, every item a test.**
  - V1, grid convergence: halving the grid changes the initial-state value by less than τ = 0.001 EFR.
  - V2, at σ = 0: matches the deterministic planner within τ and is never worse by more than τ.
  - V3, at σ > 0: on common episodes, the one-sided paired 95% bound says it does not lose to the CE replanner or to the SAA open-loop schedule.
  - V4, duality sandwich: the perfect-information relaxation with a value-function penalty (Operations Research 58(4), 2010). The gap between the DP policy's simulated value and the penalised bound is below τ plus Monte Carlo error. This check is independent of the discretisation.
  - V5, exact case: at horizon 3, exhaustive enumeration of the policy tree with the quadrature nodes reproduces the DP to 1e-9.
- **Placement.** Build and verify in `experiments/` on dev, under proceed authority. Then one owner-approved edit moves it to `ocmdp/baselines/` (the reference module plus its tests) before the Claim A preregistration freezes, and the preregistration pins its hash. It goes in `baselines/` rather than `theory/` because the DP policy is itself a comparator in decomposition (ii).
- **Before the phase-boundary preregistration?** Yes, unless the perfect-information gate passes. The gate runs first and takes minutes: if the replanner is within τ of the perfect-information bound at every DP-feasible cell, the replanner is the certified reference there and the DP is not built. The reason for "yes" otherwise: a decomposition prediction frozen after the held-out test cells had been run would no longer be blind. So both go in one preregistration.

**2. Against the criteria**
- **Severity.** V2–V5 can each fail, and a failure stops the decomposition.
- **Fixed in advance.** τ is stated here.
- **Value.** The certified optimum is what separates Claim A from the 2011 informed-switching baseline and from arXiv 2610.33412. Without it, the boundary compares two heuristics.
- **Authority.** One named owner edit.

**3. Strongest objection.** Cells with 2 mechanisms and delay ≤ 2 may not represent the held-out cells, so the decomposition holds only where the DP exists. Not fatal: Claim A's main test (E against log √2, and E − E_null) needs no DP, and the decomposition sentence is restricted to DP cells.

**4. What would show me wrong.** A V4 gap well above τ, which means the discretisation is biased. Or the replanner already within τ of the perfect-information bound everywhere, which means the DP was unnecessary work.

**5. Prior work.**
- Information-relaxation duality (Operations Research 58(4), 2010).
- The standard textbook result that open-loop feedback control is never worse than optimal open-loop (Athena Scientific, *Dynamic Programming and Optimal Control*, Vol. I).
- The PNAS 2024 paper uses a perfect-information backward-induction reference of the same kind.

---

### Decision 4. Claim B2

**1. Recommendation**
- **Primary metric: EFR.** Efficacy at the horizon becomes a preregistered secondary with no claim attached. Its threshold θ is the owner's, set in the preregistration before val.
- **Entropy: the pushforward through row-normalised C, used unconditionally.**
  - A = diag(C·1)⁻¹ C and H(πA).
  - A drug whose clipped row is all zero is assigned one-hot to its Table-1 target group.
  - The cosine kernel and the Gaussian kernel are dropped from the confirmatory test. The cosine-concavity computation on the 22-drug matrix is still run and reported, but the choice does not depend on it: sampling can only find counterexamples, never prove concavity.
- **C as input: yes, for every learned B2 agent.** The row of the measured C is a fixed per-drug input feature, identical across agents. The no-C pair (both agents) becomes an ablation, run only if the rank-4 settling launch shows the agents learn without C.
- **Required comparators.** Each is tuned at parity, with the same candidate count, val seeds and evaluation episodes:
  1. drug-entropy SAC (SD-SAC stabilisers);
  2. MinRed SAC with a learned inverse model (from its published Appendix B configuration);
  3. hard-class pushforward SAC on the Table-1 target groups, which is MinRed's known-class form and tests whether the measured overlap adds anything over textbook grouping;
  4. the α = 0 control.

  The primary contrast is pushforward-SAC against drug-entropy SAC. The contrasts against 2 and 3 are co-primary, with a three-way step-down correction, each two-sided plus TOST at m = 0.01.
- **Ablations:** no-C inputs; the cosine similarity entropy (dev only); two-level SAC; the class-size-corrected variant.
- **Non-learning references, reported only:** CE replanner, greedy, uniform over drugs and over groups, the tuned rotation, informed switching.
- **Completeness batch, outside the B2 verdict:** PPO-entropy.
- **Reserve critic: removed from the B2 agents.** In `ocmdp/agents/oc_sac.py`, ξ multiplies the reserve term in the actor loss, the embeddings are fixed buffers, and ξ is 0 in every logged window. So the critic is inert apart from initialisation random-number draws, which is dead weight. It comes back only under a new preregistration whose constraint is shown on dev to bind.

**2. Against the criteria**
- **Severity.**
  - H(πA) is concave for every A, because entropy is concave and the pushforward is linear. That gives the unique soft-greedy policy the regularised-MDP theory assumes (ICML 2019, arXiv 1901.11275). A non-concave regulariser could turn optimisation failure into a spurious null or win.
  - It has no free bandwidth, so it adds no tuning axis to one agent only.
- **Comparators.**
  - C as input to everyone removes the privileged-information confound.
  - MinRed with a learned inverse model is the published method for the same failure mode (UAI 2021, arXiv 2102.11329).
  - The hard-class arm attacks the "measured matrix" novelty directly.
- **Fixed in advance.** Everything above is stated here.
- **Power.** EFR is the metric with measured spreads; efficacy-at-horizon has none. With the correction, detectable differences are about 0.006–0.010 (derived; Decision 2).
- **Value.** It matches the claim's own wording, "entropy over resistance mechanisms".
- **Authority.** Owner acts:
  - reverse the scope-arena item "no true C in the networks" for B2;
  - drop the reserve critic, which is dropping scope;
  - set θ;
  - promote the 22-drug matrix.

**3. Strongest objection.** Moving from the similarity-sensitive H_Z to a soft pushforward brings B2 closer to MinRed's known-class form, which thins B2's novelty. That could be fatal to B2's main-track value, though not to its severity. It is acceptable only because Claim A carries the paper, and the hard-class and MinRed arms will show whether anything remains.

**4. What would show me wrong.**
- The 22-drug cosine-concavity computation finds no counterexample under a dense search plus local maximisation of the Hessian's largest eigenvalue on the simplex tangent space.
- Pushforward-SAC's learned α sits at a clamp in the rank-5 dev check.
- Most of C's mass is negative before flooring (see Decision 6).

**5. Prior work.**
- UAI 2021, arXiv 2102.11329 (MinRed).
- Ecology 93(3), 2012 (order-1 similarity-sensitive diversity).
- arXiv 2511.03849 (comparison of similarity-sensitive entropies; abstract only, no concavity result found).
- TMLR, arXiv 2209.10081 (SD-SAC).
- arXiv 2509.02930 (similarity-kernel entropy over skills, not actions).

---

### Decision 5. Rung 1, the PNAS 2024 drug-cycling DQN

**1. Recommendation**
- **Minimum gate if 1.88 cannot be traced.**
  - "Traced" means the read-only look identifies, before any of our values exists, the one code path and arguments that print the panel holding 1.88. Several candidate readings never get tried against the number.
  - If it is not traced, record before any run that E3 drops out. L3 drops to reported-only, because 1.41 sits on the same untraced scale.
  - What remains gates:
    - E2 (the MDP uses cefotaxime in more than 50% of steps, exactly);
    - L1 and L2 with the exact cut-offs already fixed (k ≥ 95 and k ≥ 91 of 100);
    - all discrimination controls below;
    - E3′, the exact Markov-chain random expectation against our simulator's Monte Carlo within 3 SE. This is a simulator check, not a reproduction claim.
  - Licensed sentence: "reproduces the published ordinal outcomes L1, L2, E2; the noise-panel values were not traced and are shown beside ours ungated."
- **Gate-0 additions.**
  - **(a) Greedy's exact status on L1.** Greedy is defined as: genotype-aware, minimises expected fitness after one SSWM step, ties go to the lowest drug index. Compute its exact expectation against random and all 105 two-drug cycles.
    - If greedy passes exactly: L1 stays as a reproduction gate, the sentence adds "which a myopic rule also passes", and the RL-against-greedy share is reported ungated.
    - If greedy fails exactly: it becomes a third negative control that must also fail the empirical 100-replicate pipeline. If it passes empirically, the pipeline is too noisy and the gate is non-discriminating.
  - **(b) Positive control.** The backward-induction MDP policy runs through the identical L1 pipeline, including the base-case sensor noise as the look identifies it, and must reach the L1 cut-off (k ≥ 95/100). Its exact value must be at least that of every cycle and random. Failing either means the gate is broken: stop.
  - **(c) Starved-budget DQN.** The published configuration is stopped after its first 10% of gradient updates, counted from the confirmed train_freq and warm-up, and evaluated greedily the same way. It must fail L1 and L2. If it passes, the gate is declared not discriminating training length, and the sentence drops "learned".
- **Rung-3 estimand.**
  - I = [Y(drug-SAC, κ₁) − Y(class-SAC, κ₁)] − [Y(drug-SAC, 0) − Y(class-SAC, 0)].
  - Y is the paper's mean population fitness per episode. It is computed per seed, with the same seed labels in all four cells. The interval is a paired-t 95% interval on per-seed I.
  - Each agent is tuned at parity separately at κ = 0 and at κ₁, and carries Decision 1's matched arm.
  - κ₁ is fixed in the rung-3 preregistration by a planner-only rule, before any agent run.
  - The class map comes from the WHO ATC index (all 15 drugs face one TEM enzyme, so classes are a proxy).
  - SD not measured. The seed count is sized from a 20-seed dev pilot that uses the variance only, and is fixed in the preregistration.

**2. Against the criteria**
- **Severity.** Three controls must fail and one must pass, and a non-discriminating gate is itself reported.
- **Fixed in advance.** The traceability test is decided before any number exists, so this is not a search over readings.
- **Value.** The interaction estimand is the only rung that compares against prior work on shared ground.
- **Authority.** None needed beyond the approvals already named (the authors' code runs only if a gate fails).

**3. Strongest objection.** Without E3 and L3, what remains is ordinal and possibly lenient, since beating random 100/100 may be easy. Not fatal: the greedy, starved and untrained controls are there to reveal leniency, and their outcome is reported.

**4. What would show me wrong.** Greedy passes L1 exactly and the starved DQN passes too. Then rung 1 cannot certify learning, and the reproduction sentence collapses to "environment reproduced".

**5. Prior work.** PNAS 2024, e2303165121 (target); PLoS ONE 2015, e0122283 (landscapes); PLoS Biol 2019, e3000515 (MDP baseline in the same field).

---

### Decision 6. Collateral sensitivity

**1. Recommendation.** A stated limitation, not a change to `ocmdp/core/`. Before the B2 preregistration, report from the promoted S2 matrix:
- the count of negative entries;
- the share of total absolute mass they carry;
- which source drugs carry it.

Fixed now: if that share exceeds 20%, B2's sentence names the instance "the cross-resistance part of the 22-drug matrix".

**2. Against the criteria**
- **Severity and fixed in advance.** The wording rule is set before the data are examined.
- **Value.** A core change would invalidate the dev phase map and every instance, which costs the spine for a B2 detail. The chosen entropy also needs nonnegative rows (a signed row cannot be normalised into a distribution), so the limitation is coherent with Decision 4.
- **Authority.** No locked-path edit.

**3. Strongest objection.** The source study's headline finding is collateral sensitivity under strong aminoglycoside selection (MBE 2014). Flooring it removes the structure that collateral-sensitivity cycling exploits, and a reviewer can call the "measured matrix" half-measured. Not fatal if quantified and worded as above. It would be fatal to any claim about cycling strategy.

**4. What would show me wrong.** A large negative-mass share concentrated in the aminoglycoside rows.

**5. Prior work.** Sci Transl Med 2013, 5:204ra132 (collateral-sensitivity cycling); PLoS Biol 2019, e3000515; MBE 2014, 31(9).

---

### Decision 7. Sequencing

**1. Recommendation**
- **Claim A alongside rung 1, with limits.** Claim A's planner-only work runs alongside rung 1: perfect-information gate, DP build and verification, derivation, preregistration draft. Claim A's test-tier launch waits for rung 1's verdict, whichever way it goes; it is not conditional on a pass. Claim A uses no learner, so a failed reproduction does not touch it. This reading needs the owner's confirmation in `CLAUDE.md`.
- **Derivation restart: now, first in the queue.** Its predictions are hashed before any new phase, perfect-information-gate or DP number is generated. Partial notes from the earlier attempt are reused only if an audit shows they contain no measured value. The owner sets a spend allowance sized to finish.
- **What the derivation agent sees.** It works in a sanitised bundle: a separate directory with a manifest hash.
  - **May read:**
    - the source of `ocmdp/core/`, `ocmdp/envs/` and `ocmdp/baselines/`;
    - the contrast definitions;
    - the held-out (κ, δ, H) lines and the σ grid;
    - the predictions schema;
    - public literature.
  - **May compute:** closed forms and deterministic σ = 0 quantities at any cell.
  - **May not:**
    - simulate the stochastic system (σ > 0) at any cell, or run any Monte Carlo;
    - read `runs/`, `figures/`, `decisions/`, `arena/`, the plan document, `README` results or git history.

**2. Against the criteria.** Ordering in time is a stronger guarantee than a read bar. The σ > 0 rule draws a checkable line between derivation and measurement.

**3. Strongest objection.** Allowing σ = 0 planner values at held-out cells lets the theory borrow numbers. Not fatal: they are model properties, not measured crossings, and the dev-map null is the comparator.

**4. What would show me wrong.** The bundle's manifest audit turns up a measured value. Or the predictions sit unusually close to the barred dev map compared with the null.

**5. Prior work.** Blind analysis (Annu. Rev. Nucl. Part. Sci. 55, 2005).

---

### Owner acts, with the exact change
1. **`CLAUDE.md`, the pending temperature bullet:** "best interquartile mean" becomes "best mean of the primary metric over 10 val seeds, every run counted; interquartile mean reported beside".
2. **`CLAUDE.md`, the claims ladder:** state that Claim A's test launch waits for rung 1's verdict but not its pass.
3. **`ocmdp/baselines/`:** one edit adding the verified DP reference and its tests.
4. **Reserve critic:** approve dropping it (scope).
5. **B2 inputs:** approve C as an input feature for all learned B2 agents.
6. **B2 data and θ:** promote the 22-drug matrix and set θ.
7. **Derivation:** set a spend allowance for the restart.

No change to `seeds.yaml` or `ocmdp/core/`.

Sources:
- [arXiv 2209.10081 (SD-SAC)](https://arxiv.org/abs/2209.10081v4)
- [arXiv 1901.11275 (regularised MDPs, ICML 2019)](https://arxiv.org/abs/1901.11275v2) · [PMLR listing](https://proceedings.mlr.press/v97/geist19a.html)
- [arXiv 2511.03849 (similarity-sensitive entropies)](https://arxiv.org/abs/2511.03849v2)
- [arXiv 2012.02113](https://arxiv.org/pdf/2012.02113v2)
- [arXiv 2102.11329 (MinRed, UAI 2021)](https://arxiv.org/abs/2102.11329)
