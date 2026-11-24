# Proposal 1

Two notes before the answers:
- **Arithmetic, not results.** Every power figure below is a calculation on the pooled A/A paired SDs the criteria give as fact: 0.018 pooled, 0.030 upper bound, both from superseded-design reports. They are labelled "derived". No figure here is a result.
- **A framing problem in decision 2.** It asks for "one" option, but the 100-seed cap binds two different things: training seeds for learned agents, and evaluation episodes for the planner-only Claim A. I answer each separately and say so, rather than picking one reading.

---

## Decision 1. Rules every entropy preregistration carries

### 1a. The "not tested" branch

**Recommendation.** For each agent i, compute the bias capacity from its selected candidate:

b_i = α_i · (H_max,i − H_floor,i) / reward_scale

- `reward_scale` is 10. Per-quarter reward is −10 × mean failure probability (`ocmdp/agents/oc_sac.py`), so b_i is in failure-rate units.
- For a learned candidate, α_i is the final scalar.

The entropy-space comparison is **not tested** if:
- b_i < m/2 for either agent, where m is the preregistered equivalence margin; or
- either agent selects α = 0.

When it is not tested:
- No test seeds are spent on the entropy-space contrast.
- The preregistration's result is the sentence "not tested: agent X's selected temperature cannot move its EFR by more than m/2 from the unregularised optimum."
- No re-selection is allowed. Dropping grid points to force a test is forbidden.

**Why it meets the criteria.**
- **Severity.** At the optimum, each agent's EFR is within b_i of the reward-optimal EFR. The soft-optimal bias bound is α·ΔH per step.
  - If both b_i < m/2, an equivalence verdict cannot fail, so it is not tested.
  - If only one is below m/2, that agent is effectively α = 0. The contrast is then "entropy against none", not "entropy space against entropy space".
- **Fixed in advance.** b_i depends only on the grid value and the reachable range from `entropy_range.py`. The whole list of candidates that would trigger "not tested" can therefore be printed before val runs.
- **Within authority.** It is preregistration text and needs no owner act.
- **Not tilted.** It is symmetric in the two agents.

**Strongest objection.** b_i is a bound on the objective at the optimum. With function approximation, two agents with small b can still differ through optimisation dynamics. Conversely, a large b does not guarantee the entropy term is active.
- Not fatal. The rule only stops vacuous tests; it never licenses a claim.
- Report the realised share α_i·(mean H_eval − H_floor)/reward_scale beside it, descriptively.

**What would show it wrong.** On dev, both agents at a candidate with b < m/2 produce a paired difference whose interval excludes ±m. The bound would then be missing an effect, most likely an optimisation artefact.

**Prior work.**
- The entropy-regularisation bias bound V* − V*_τ ≤ τ log|A|/(1−γ): Operations Research 2022, arXiv 2007.06558; regularised MDPs, ICML 2019, arXiv 1901.11275; arXiv 1705.07798.
- Equivalence testing (TOST): J. Pharmacokinetics and Biopharmaceutics 15, 1987.

### 1b. The matched-temperature arm at test

**Recommendation.** Let θ_OC and θ_SAC be the selected grid candidates: a fixed α value or a learned fraction c, since candidates are defined identically for both agents. Every test seed runs four configurations: (OC, θ_OC), (SAC, θ_SAC), (OC, θ_SAC) and (SAC, θ_OC).
- **Primary:** the tuned contrast.
- **Secondary, Holm-corrected as a pair:**
  - E1 = OC(θ_OC) − SAC(θ_OC);
  - E2 = OC(θ_SAC) − SAC(θ_SAC).
- **When an entropy-space sentence is allowed:**
  - "The difference is due to entropy space" only if E1 and E2 share the tuned contrast's sign and both exclude 0.
  - "Equivalent in entropy space" only if both are within ±m.
  - Otherwise: "entropy space and temperature level not separated."
- If θ_OC = θ_SAC, the matched arm coincides with the tuned contrast and is not rerun.
- "Not tested" applies to each matched cell separately.

**Why it meets the criteria.**
- It separates the space from the level on paired seeds.
- Matching on the grid entry is defined before val. Matching on the realised α would be an outcome.
- It doubles test runs on the same 100 seeds, with no new seed demand.

**Strongest objection.** A learned fraction c means different realised α in different entropy spaces (log 22 against log 6 nats). "Matched" is therefore matched in the protocol's terms, not in reward per nat. Not fatal: report the realised α of every cell, and match fixed-α candidates exactly.

**What would show it wrong.** E1 and E2 have opposite signs, which would mean an interaction between space and level that the 2×2 cannot summarise.

**Prior work.**
- Tuning confounds in algorithm comparison: ICML 2020, arXiv 2006.16958; ICML 2023, arXiv 2306.01324.
- Temperature sensitivity of discrete SAC: arXiv 2209.10081.

### 1c. The selection statistic

**Recommendation.** Each agent selects the candidate with the lowest **mean** EFR over the same 10 val seeds and every preregistered cell. Every run is counted, including non-learning runs at their realised EFR. Ties go to the earlier candidate, as already adopted.
- This is the statistic the confirmatory paired t-test estimates, because the mean paired difference equals the difference of means.
- IQM, probability of improvement and runs_learned are still reported beside it, never used for selection.

**Owner act.** In the temperature-arena replacement text for CLAUDE.md, change "the candidate with the best interquartile mean of the preregistration's primary metric over 10 val seeds" to "the candidate with the best mean of the preregistration's primary metric over 10 val seeds, every run counted".

**Strongest objection.** A 10-seed mean on bimodal outcomes is a noisy selector, and the winner's curse can pick a lucky configuration.
- Not fatal for validity, because test seeds are fresh. It only costs comparator strength, and equally for every agent.

**What would show it wrong.** The val mean ranking and the test mean ranking of the two selected configurations disagree in sign more often than seed noise predicts. Not measured; record it.

**Prior work.** The IQM, PoI and mean trade-off: NeurIPS 2021, arXiv 2108.13264.

### 1d. Non-learning runs

**Recommendation.**
- Keep the code's existing definition of `runs_learned`: more than half the reference's gain over uniform. Name the reference in each preregistration: the noise-free planner at σ = 0, the replanner under noise.
- Never exclude non-learning runs. They enter the primary paired mean at their realised EFR.
- Report:
  - learned k/n per agent, with exact (Clopper–Pearson) 95% intervals;
  - the paired 2×2 table of learned status (both, only OC, only SAC, neither);
  - an exact McNemar test as a preregistered secondary;
  - the difference among both-learned pairs, as descriptive only and labelled as conditioning on a post-treatment variable.

**Why.** Learning failure is part of an agent's effect. Excluding it would be an exclusion threshold that can turn a null into a win.

**Objection.** The 0.5 cut is arbitrary. Not fatal: it is used only for counting, never for exclusion.

**Measure.** The sensitivity of the counts to cuts of 0.3 and 0.7, reported, not used.

**Prior work.** arXiv 1709.06560 (AAAI 2018); arXiv 2304.01315.

---

## Decision 2. Test-tier capacity

**Recommendation.**
- **Learned-agent equivalence (B1, B2, rung 2): set margins at m = 0.01 EFR. Do not widen the training seeds.** Use 100 test seeds per preregistration. The same seeds may be reused across different preregistrations, since tuning never touches them.
- **Claim A, which is planner-only:**
  - The owner widens the test tier for evaluation episodes. Exact edit in `seeds.yaml`: `test: [1000, 1099]` becomes `test: [1000, 1399]`.
  - Claim A uses 400 episodes per cell. Learned-agent preregistrations still state n = 100 training seeds.
  - If the owner declines, the fallback is fixed now: only grid points with σ ≤ 0.2 enter the crossing test, and crossings above 0.2 are censored as "above grid".

**Why it meets the criteria. Power, all derived** (normal approximation; TOST at 5% per side, true difference 0):
- n = 100, m = 0.01: power about 1.00 at SD 0.018, 0.97 at 0.026, 0.91 at 0.030. Power falls below 0.8 once the SD exceeds about 0.034.
- n = 100, m = 0.005: about 0.74 at SD 0.018, and close to 0 at 0.030. Margins below 0.01 are therefore not resolvable without seeds the budget cannot carry; the plan's own calculation puts m = 0.005 above 100 seeds.
- Two-sided difference test at n = 100: the minimum detectable effect at power 0.8 is about 0.005 (SD 0.018) to 0.008 (SD 0.030). With Holm over three secondary contrasts, about 0.0097 at SD 0.030.
- **What it cannot resolve:** equivalence tighter than 0.01, and differences below about 0.008–0.010.
- **Claim A:** 4× the episodes halves the derived half-widths at σ = 0.4 (±0.017–0.020 at 100 episodes, unaudited). No training is involved, and cost is linear in episodes.

**Severity and authority.**
- The margin is fixed now, before any val run.
- The preregistration recomputes power from the val paired SD of the selected configurations and states it before test. An inconclusive TOST is reported as inconclusive.
- The `seeds.yaml` edit is the owner's.

**Strongest objection.** The SDs are A/A spreads at other instances from a superseded design. Agents optimising genuinely different objectives at K = 22 may spread more, and outcomes are bimodal. A margin chosen by what the design can resolve, rather than by clinical relevance, is also backwards in principle.
- Not fatal. Inconclusive is a stated outcome. Each report also expresses m as a fraction of the uniform-to-reference gain on its instance, so readers can judge relevance.

**What would show it wrong.** A val paired SD at the selected configurations whose upper 95% bound exceeds 0.034 would mean the equivalence branch is underpowered at n = 100.

**Prior work.** Seed counts in deep RL: arXiv 1806.08295. Equivalence margins: the TOST reference in 1a.

---

## Decision 3. The noise-optimal reference

**Recommendation.**

**Placement.**
- Build it in `experiments/dp_reference.py`, with tests in `tests/`, under proceed authority.
- The Claim A decomposition preregistration pins its SHA-256 and the hash of its verification report in `runs/`.
- The test-launch guard already requires a clean, committed tree.
- Moving it to `ocmdp/baselines/` is an owner act at rung-6 release only.
- The hash pin freezes it as firmly as per-edit approval would, without making the owner approve every fix during verification.

**Scope.**
- Two mechanisms, delay up to 2, the dev-grid memory values and σ grid, and any held-out cells that satisfy these limits.
- State: resistance and occupancy per class on a grid, the pending occupancy exactly, and time to go.
- Action set identical to the CE replanner's, so every comparison stays within one policy class.
- Gauss–Hermite quadrature over the shock.
- Call it the "DP reference", never V*. Its simulated value is a feasible lower bound; the perfect-information relaxation is the upper bound.

**Verification. Tolerances fixed now; each check can fail.**
- V1: at σ = 0, the DP value simulated on the continuous environment is within τ = 0.001 EFR of the planner's.
- V2: halving the grid spacing changes the start-state value by less than τ. So does going from 5 to 9 quadrature nodes.
- V3: under noise, on dev episodes, the DP does not lose to the CE replanner or to the SAA open-loop schedule beyond Monte Carlo error (one-sided paired t at 5%).
- V4: the DP's simulated value does not exceed the perfect-information bound. A violation is a bug.
- V5: unit tests: exact agreement with brute-force enumeration on a tiny cell, and with a textbook MDP.
- "Certified" is said only where the DP-to-PI gap is below τ_cert, fixed in the preregistration.

**Must it exist before the phase-boundary preregistration?** Split Claim A into two preregistrations.
- **A1** (the boundary of `planned_once_minus_greedy` against the null): a raw paired contrast that needs no DP. It can freeze without it.
- **A2** (decomposition (i)–(iii) and the (ii) prediction): freezes only after V1–V5 pass.
- The perfect-information gate runs first. Where the CE replanner already sits within τ_cert of the PI bound, no DP is needed for (ii) and (iii).

**Strongest objection.** Code under `experiments/` that decides verdicts is less protected than code under `ocmdp/baselines/`.
- Not fatal. The hash pin, the clean-tree guard and preregistered tolerances protect exactly the runs that carry claims. The owner can still require `ocmdp/baselines/`; nothing else here changes if so.

**What would show it wrong.** V3 fails at the finest feasible grid. The discretisation would then be too coarse to serve as a reference at all.

**Prior work.**
- Information relaxations and duality: Operations Research 58(4), 2010.
- CE and MPC as suboptimal control: European Journal of Control 11, 2005.
- The open-loop, CE and optimal regime map in LQ systems: arXiv 2610.33412 (RELATED).

---

## Decision 4. Claim B2

### Primary metric: expected failure rate

**Recommendation.** EFR is primary. Mean horizon efficacy, 1 − failure probability averaged over drugs at the horizon and threshold-free, is a preregistered secondary. Both are already stored per rollout.

**Why.**
- It is what both agents optimise, so the test does not favour either.
- It is the only metric with measured spreads, so it can be powered.
- It is threshold-free.
- It matches Claim A.

Horizon efficacy is the metric entropy is hypothesised to help, and no agent optimises it. Making it primary would choose the metric for its expected winner. It is also sensitive to end-of-horizon drug-burning by any near-optimal finite-horizon policy.

**Objection.** The CLAUDE.md claim says "preserves drug efficacy", which reads closer to horizon efficacy. Not fatal, but the owner should change that sentence to the EFR wording the scope arena already adopted.

### Entropy and kernel

**Recommendation.** The single entropy is the pushforward through row-normalised C: H(π C̃), with C̃[a, ·] = C[a, ·] / Σ_j C[a, j] over the six target groups.
- Assert that no row sums to 0 after clipping.
- It is concave in π by construction, since entropy composed with a linear map is concave. It has no free parameter.
- It reduces exactly to the hard-class form when C is a hard assignment.
- **Drop the Gaussian kernel.** Its bandwidth is a tuning dimension only one agent would have, which breaks parity.
- **Cosine H_Z becomes an ablation only if** a Hessian check on the simplex shows it concave on the 22-drug matrix. That is a computation run before any agent run, not a judgement.

**Objection.** Row normalisation discards the magnitude of cross-resistance: weak and strong profiles with the same shape are identical. The regulariser is also not strictly convex, since C̃ has rank at most 6, so the actor is unregularised along within-group directions.
- Not fatal. The second property is the intended one, and the B1 identity relies on it. The first is a stated limitation.

### Does drug-entropy SAC also get C as input?

**Recommendation.** Yes, and so does the H_C̃ agent. Both networks take the rows of C as their fixed action-embedding buffer, so the agents differ only in the entropy term. The pair without C is an ablation.

**Why.** It removes the confound that H_C̃'s gain is simply access to C. It is also the strongest honest version of drug-entropy SAC.

**Objection.** The scope arena chose no true C in networks. B2 is the case it reserved, since the plan's comparator list already names drug-entropy SAC with and without C. Rank 4's runs_learned-without-C check is reported but does not reopen this.

### Comparators

**Required, tuned at parity on the same grid and val seeds:**
- Drug-entropy SAC with C as input. This is the single primary contrast: two-sided test plus TOST at m = 0.01.
- Holm-corrected secondaries:
  - the α = 0 control (with C);
  - MinRed SAC with a learned inverse model (UAI 2021 Appendix B configuration as the starting point);
  - mechanism-level SAC with hard Table-1 groups. This tests whether partial overlap adds anything over hard classes.

**Ablations (no claim):**
- both agents without C;
- MinRed with known classes, which is near-identical to the hard-class pushforward under deterministic transitions;
- cosine H_Z, conditional as above;
- PPO-entropy, in the completeness batch.

### The reserve critic

**Recommendation.** Remove it from OC-SAC and its training loop, as a logged refactor. A component that never enters a claim is dead code under the repository rules.
- It is inert: ξ = 0 in every logged window, and the embeddings are a buffer the reserve loss cannot train.
- Removing it shifts initialisation draws. Pre-change runs already do not pool with new ones.
- An option-depletion constraint returns only as its own preregistered claim, with a budget shown to bind on dev (ξ > 0 in the last window).

**What would show the B2 choices wrong.**
- The cosine-kernel check passes, and the cosine and pushforward ablations disagree in sign. Entropy choice would then drive B2.
- Drug-SAC learns markedly better with C than without. The confound would have been real.

**Prior work.**
- MinRed: UAI 2021, arXiv 2102.11329.
- Similarity-sensitive diversity: Ecology 2012; its maximisation, arXiv 1512.06314; the axiomatic treatment, arXiv 2012.02113. My search found no source establishing concavity of the order-1 form.
- Kernel entropy of samples: arXiv 2210.02410.
- Which similarity-sensitive entropy to use: arXiv 2511.03849.
- The data: Mol Biol Evol 2014, 22-drug cross-resistance matrix.

---

## Decision 5. Rung 1

### Minimum gate if 1.88 cannot be traced

**Recommendation.** Record before any run that E3 drops out. The minimum gate is then:
- E2;
- the MDP positive control below;
- an internal check: the reimplementation's random-cycling value computed exactly by Markov chain and by 10^5-episode Monte Carlo agree within MC error;
- L1 and L2 unchanged;
- **L3 as a ratio.** The 95% interval of RL-fit / random (exact) must lie inside 1.41/1.88 ± 0.10/1.88, which is ±0.053 (derived). 1.41 and 1.88 come from the same panel, so the ratio is invariant to the unknown multiplicative scale.

If the noise definition cannot be pinned either, L3 drops out. Rung 1 then licenses only the L1 and L2 sentences.

### Gate-0 additions

- **Greedy's exact L1 status.** A Markov-chain computation before freezing, so the preregistration is contingent on a fact, not an outcome.
  - If greedy passes L1 exactly, L1 stays as the published claim but is labelled "not discriminating against myopic feedback". RL-genotype against greedy is reported as a paired diagnostic.
  - If greedy fails L1, L1 is labelled discriminating.
- **Positive control.** The backward-induction MDP policy, evaluated by the authors' comparison procedure and episode counts, must reach k ≥ 95 of 100 on L1. If it fails, the procedure cannot discriminate: stop.
- **Starved-budget DQN.** The same DQN with 10% of the full run's gradient updates and the warm-up kept. The update count is taken from the read-only look and fixed in the preregistration. It is expected to fail L1 and L2.
  - If it passes, L1 and L2 are labelled "not discriminating learning amount".
  - If greedy passes, the starved DQN passes, and E3 is out, rung 1 is reported as "reproduced only non-discriminating claims" and stops for the owner.

These rules can only weaken the licensed sentence, never strengthen it.

### Rung-3 estimand

θ = [EFR(drug-entropy) − EFR(class-entropy)] at κ > 0, minus the same contrast at κ = 0.
- Computed per test seed, with all four runs on the same seeds. Exact t 95% interval over n = 100.
- Each agent's temperature is selected once on val at κ > 0 and held fixed at κ = 0, so θ isolates coupling.
- One κ, fixed in the preregistration from the instance family's default before any coupled run on the published landscape. No κ search.
- The two-sided test on θ is primary. There is no equivalence claim on θ: the minimum detectable effect is about 0.007–0.012 (derived, interaction SD √2·s). The not-tested rule and the matched arm apply.

### Objection, measurement and prior work

**Strongest objection.** The ratio form of L3 assumes the unknown scale is purely multiplicative. An additive offset, such as sensor noise inside the reported means, would break it. Not fatal: the read-only look either pins the noise definition or L3 drops out.

**What would show it wrong.** The MDP fails the positive control under the authors' procedure, which would make the whole gate design moot.

**Prior work.**
- The target: PNAS 2024, DOI 10.1073/pnas.2303165121.
- The landscapes: PLoS ONE 2015, e0122283.
- An MDP over resistance profiles: PLoS Biol 2019, e3000515.

---

## Decision 6. Collateral sensitivity

**Recommendation.** Keep it as a stated limitation. No `ocmdp/core/` change for this paper.
- Every B2 preregistration and the claim sentence say "on the non-negative part of the measured matrix (collateral sensitivity set to 0)".
- The data-promotion script writes the count and magnitude of the clipped entries to `runs/` so they can be reported.

**Why.**
- A core change alters Claim A's model class. It would invalidate the dev map, the blind derivation's model and every logged report, all on the spine's critical path.
- Collateral-sensitivity dynamics would need their own validation against data. That is a separate study.

**Strongest objection.** In collateral-sensitivity-rich data, concentrating on paired drugs (low entropy) can be optimal, so flooring may remove exactly the structure that would make structure-aware entropy lose.
- Not fatal to the RL claim, which is defined on a given kernel. It does bound the application sentence, and the paper must say that the direction of B2 under collateral sensitivity is not tested.

**What would show it wrong.** The clipped mass is a large share of the off-diagonal entries. "Measured matrix" would then be a misleading description.

**Prior work.**
- Collateral-sensitivity cycling: Sci Transl Med 2013; Cell 2018 (P. aeruginosa).
- PLoS Biol 2019, e3000515.
- Trends in Microbiology 2015 (review).

---

## Decision 7. Sequencing

### Claim A planner work and rung 1

**Recommendation.** Run them in parallel, including A1's test run once it is approved.
- Claim A's comparators are planners, heuristics and stateless bandits, with no learned agent. Rung 1 validates the RL pipeline Claim A does not use, so rung 1's outcome cannot change Claim A's verdict.
- Rungs 2, 3, B1 and B2 still wait for rung 1.

### When the blind derivation restarts

**Recommendation.** Restart as soon as the owner fixes the held-out (κ, δ, H) cell list in the A1 draft. Do not wait for the DP or for rung 1.
- **Deliverable first:** the scorer-format predictions JSON (`sigma_star`, `sign_at_max_sigma`, directions), then the (ii) predictions at DP-feasible held-out cells, then the reasoning.
- **Spend-limit protection:** each version is timestamped and archived. The last one delivered before freeze is the one used. All versions are kept, and none is chosen after any held-out measurement.
- The predictions' hash goes into A1 before any held-out cell runs.

### What the derivation agent may read

Enforce by an allowlisted sandbox checkout, not by instruction.

**May read:**
- `ocmdp/core/` and `ocmdp/envs/` (the model equations);
- the policy code of the contrast's policies;
- `CONTRASTS` and `cell_config` in `experiments/sweep_phase_diagram.py`;
- the docstring and format of `experiments/score_phase_predictions.py`;
- the held-out cell list;
- public literature, including arXiv 2610.33412 and the 2011 informed-switching study.

**May compute:** analytic work, including perturbation expansions in σ with derivatives evaluated at σ = 0, and deterministic σ = 0 quantities of its own reduced models.

**May not read or run:**
- `runs/`, `figures/`, `decisions/`, `arena/`, the plan document, the README results section, inbox or memory notes that quote project measurements, or git history;
- any sampled noise path, Monte Carlo run or stochastic DP;
- any execution of `ocmdp` rollouts.

The plan and this arena's facts are on the forbidden list because they quote the contrast's SDs by σ, which is measurement leakage.

The transcript is archived, and its hash goes into A1.

### Objection, measurement and prior work

**Strongest objection.** Allowing σ = 0 deterministic computation lets the agent "measure" one end of the contrast. It is a model input a theory legitimately uses, though, and the crossing under noise stays unmeasured. Not fatal.

**What would show it wrong.** The transcript audit finds a forbidden read, or its predictions match the dev-map crossings far more closely than at held-out cells. Either would point to leakage.

**Prior work.** None found on blind-derivation protocols in RL. The derive-first rule here follows standard preregistration practice.

---

## Owner acts named above

1. In the temperature-arena replacement text for CLAUDE.md: "best interquartile mean" becomes "best mean ..., every run counted".
2. In `seeds.yaml`: `test: [1000, 1099]` becomes `test: [1000, 1399]`, for Claim A episodes.
3. Optionally, reword the CLAUDE.md claim sentence to the EFR wording.
4. Fix the held-out cell list.
5. Move the DP reference to `ocmdp/baselines/` only at release.

## Two items settled by computation, run before any preregistration they govern

- The cosine-kernel concavity check on the 22-drug matrix.
- Greedy's exact L1 status.

- arena/2026-11-24-design-criteria.md
- arena/2026-11-23-scope-outcome.md
- arena/2026-11-23-temperature-outcome.md
- arena/2026-11-23-reproduction-outcome.md
- decisions/2026-11-23.md
- decisions/2026-11-24.md
- ocmdp/objectives/pushforward_entropy.py
- ocmdp/agents/oc_sac.py
- ocmdp/agents/lagrangian.py
- experiments/score_phase_predictions.py
- seeds.yaml
- references/2026-11-16-weaver-2024-drug-cycling-dqn.md
- the 2026-11-24 plan revision

Sources:
- [arXiv 1512.06314](https://ar5iv.labs.arxiv.org/html/1512.06314)
- [arXiv 2511.03849](https://arxiv.org/html/2511.03849v4)
- [arXiv 2012.02113](https://arxiv.org/pdf/2012.02113)
- [arXiv 0910.0906](https://ar5iv.arxiv.org/html/0910.0906)
- [arXiv 2601.03064](https://arxiv.org/pdf/2601.03064)
