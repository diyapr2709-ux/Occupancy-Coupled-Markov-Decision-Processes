# Proposal 1

# Proposal: put the control characterisation first, cut the entropy claim to a theory check plus one test on partial cross-resistance, and remove the true C from the networks

## 1. Recommendation

### Sub-question 1: narrow the entropy claim

Stop pursuing the claim as it is written. Under exact classes it is already covered twice:
- **By the logged identity.** With tied networks, SAC equals OC-SAC plus a reward bonus alpha(s) log n_class(a) (decisions/2026-11-23.md).
- **By prior work.** That is the deterministic, known-class form of MinRed SAC (Baram et al., UAI 2021).

Two narrower claims replace it. Each is preregistered separately.

- **B1, a theory check rather than a claim of benefit.** "With exactly interchangeable drugs and networks that cannot tell class members apart, the efficacy gap between drug-entropy and mechanism-entropy SAC at a fixed temperature alpha equals the gap between the soft-optimal schedules of the two objectives. `experiments/predict_entropy_gap.py` predicts that gap on the mean-field model, and it is zero at alpha = 0."
  - Test: fixed (not auto-tuned) alpha at a few values set in advance. Measured gap minus predicted gap must fall inside an equivalence margin.
  - This can fail, for example through optimisation effects or critic bias.
- **B2, the only part with new content.** "On an instance whose cross-resistance matrix is measured and not block-diagonal, SAC with the similarity-sensitive entropy H_Z, using a drug-similarity kernel taken from that matrix, either lowers expected failure rate against drug-entropy SAC tuned at parity, or is equivalent to it within margin m."
  - Both outcomes are pre-declared. This is a two-sided test plus TOST.
  - Drop the phrase "at equal treatment cost". `ToyConfig` has no treatment cost, so the phrase means nothing unless you define a cost.

### Sub-question 2: what carries the paper, in order

**Claim A carries the paper.** It is the control characterisation of OC-MDPs, measured as a decomposition of the value of acting on the observed state, at fixed (delay, memory, noise), into three parts:
- **(i) Cost of certainty equivalence in the schedule:** the noise-free plan committed in advance, minus the best open-loop schedule under noise.
- **(ii) True value of feedback:** the best open-loop schedule minus the optimal closed-loop policy. Only this part is a property of the problem.
- **(iii) Suboptimality of certainty-equivalent replanning:** the CE replanner minus the optimal closed-loop policy.

On top of that sits the blind-derived boundary for the contrast that can flip sign (`planned_once_minus_greedy`). It is tested at held-out (kappa, delta, horizon) against the theory-free null, using the analysis code already written (`experiments/score_phase_predictions.py`, `experiments/null_phase_predictions.py`).

The decomposition matters for a theoretical reason:
- Open-loop feedback control re-solves the optimal stochastic open-loop problem at each step. It is never worse than the optimal open-loop policy; I recall this result from Bertsekas, Dynamic Programming and Optimal Control (DPOC) Vol. I, §6.2, but did not re-verify it this session.
- So "feedback beats a committed schedule" is a theorem when both sides are optimised under the noise. It can only fail when CE planners are compared.
- A hostile reader will raise this. Without (ii) the phase map compares two suboptimal planners.

**Order:**
0. **Owner steps** (listed at the end), and a gatherer pass that files the new prior work in §5 to inbox/. Maynard-Zhang et al. 2026 needs your PRE-EMPTS call.
1. **Rung 1:** an independent reimplementation of Weaver et al. 2024 (15 drugs, 16 genotypes, SSWM, random cycling, 105 two-drug cycles, backward-induction MDP, DQN), with a tolerance fixed beforehand. The paper gives no base-case means, so the targets are the stated quantities only: 100/100 and 98/100 replicate counts; 1.41 vs 1.88 at noise 40; policy correlations 0.96 and 0.82 (all "reported by" the authors, inbox file). Distribution-level matching only, because no seeds are set there.
   - In parallel, planner-only work on Claim A. See the ladder note in §2.
2. **Noise-optimal reference for small cells** (sub-question 5), then the Claim A confirmatory preregistration on held-out cells, reporting (i) to (iii).
3. **Rung 2 control:** OC-SAC vs SAC on the reproduced Weaver setup. Expected result is a tie, tested as equivalence.
4. **B1** (cheap; one fixed-alpha grid).
5. **Rung 4 parity tuning, then B2 on test seeds** (one preregistration).
6. **Late completeness batch.**
7. **Rung 3 bridge** only if time remains; it is optional for this paper.

The plan's rank-3 synthetic two-factor KL sweep is dropped as the spine. Under exact classes its outcome follows from the identity, which is what B1 tests directly. Under partial classes, hand-built class vectors are exactly the instance design a reviewer calls arbitrary.

### Sub-question 3: comparators

Every learned comparator gets the same dev seeds, the same number of tuning configurations, the same evaluation episodes, and is logged in `baselines/registry.md`.

**Entropy tests (B1, B2):**

| # | Comparator | Strongest published version | Reference code |
|---|---|---|---|
| 1 | Drug-entropy discrete SAC | SD-SAC stabilisers (Zhou et al., arXiv 2209.10081): averaged twin-Q, Q-clip, entropy penalty. Decoupled-regularisation dynamic target, alpha·log\|A(s)\| (Mohammadpour et al., ICLR 2024), per state under masks. Fixed alpha (SD-SAC uses 0.05) and an annealed target (arXiv 2112.02852) as members of the tuning grid. | SD-SAC repo; SobhanMP/decoupled-soft-RL; CleanRL sac_atari.py update rule. No code for the annealed target. |
| 2 | MinRed SAC | Baram et al., UAI 2021, in two forms: known-class ARS form, and learned inverse model q(a\|s,s'). SB3 hyperparameters from their Appendix B. | None; implement from the paper. |
| 3 | Mechanism-level SAC (M-way head, uniform within class) and two-level SAC (learned within-class choice, temperature on the class level only) | Hybrid-SAC factorisation (Delalleau et al., arXiv 1912.11077) | None; build from 1. |
| 4 | Drug SAC with class-size-corrected entropy | Derivation only | Not applicable. Equal to OC-SAC under tied networks, so it is informative only without C in the networks. |
| 5 | Redundant-action masking, "No Prior Mask" | AAAI 2024, arXiv 2312.06258 | Code status not checked; include only if code exists, and say so either way. |
| 6 | Non-learning references | Uniform mixing over drugs (the clinical "mixing" of Bergstrom et al. 2004, i.e. maximal drug entropy); uniform over mechanisms; cycling; tuned rotation; myopic greedy; CE replanner; DP reference where tractable | Already in the repo except DP. |

**Late completeness batch:**

| Comparator | Source | Code |
|---|---|---|
| PPO with entropy bonus | CleanRL ppo.py; RL Zoo ppo.yml as the nearest tuned configurations; ent_coef in the grid | CleanRL commit e421c2e; RL Zoo eec15dc |
| PPO-Lagrangian | OmniSafe PPOLag; a PID-Lagrangian variant (Stooke et al., ICML 2020) as the stronger multiplier rule | OmniSafe 15603dd. That it includes a PID variant is from memory, not verified. |
| DQN | Weaver configuration and CleanRL dqn.py; also needed for rungs 1 to 3 | evo_dm (GPL-3.0, review only); CleanRL 004f8a0 |
| SW-UCB / D-UCB | Garivier and Moulines 2011, labelled as stateless | SMPyBandits 012fc13 (MIT) |
| Delay-aware RL | Derman et al., ICLR 2021. Note it addresses execution delay, not a delayed kernel response. | galdl/rl_delay_basic (not opened) |
| Recurrent SAC without the pending-occupancy feature | Unsourced | Not applicable |
| Rebounding / memory bandit planners | Leqi et al. 2021 cyclic-greedy; Clerici et al. 2023 OFUL-memory. These are policies that use own-history memory, for Claim A. | None found |

### Sub-question 4: no true C in the networks for the main experiments

In the main experiments, agents learn without the true cross-resistance matrix in their networks:
- drug embeddings are one-hot or learned;
- the observation is the raw drug-indexed occupancy plus the pending stack, asserted present (not `MechanismObservation`).

C enters only where a method is defined by it: OC-SAC's pushforward map psi, H_Z's kernel, and MinRed's known-class form.

The reason is that with C in both the embeddings and the observation map, drugs in a class are indistinguishable. The comparison is then structurally tied by the identity. That configuration is kept only as B1's arm, where the identity is the thing being tested.

This is settled empirically first, by one dev launch (see "What experiments settle" below).

### Sub-question 5: build the noise-optimal reference, for small cells only

- **Closed-loop optimum.** Finite-horizon backward induction on TwoDrug (K = 2, M = 2).
  - State: resistance (r1, r2) on a grid, mu (one dimension for K = 2), and the pending stack. The stack has 2^d values under a one-drug-per-quarter policy class, or 5^d under a mixing-fraction grid {0, .25, .5, .75, 1} for d ≤ 2.
  - Horizon 40. The reflected noise shock is integrated by Gauss-Hermite quadrature with interpolation.
  - The grid is refined until the start-state value moves by less than a tolerance fixed in advance.
- **Best open-loop schedule under noise.** Sample-average approximation: optimise the committed schedule over N sampled noise paths, evaluate on fresh paths.
- **Verification check that can fail.** Within its policy class, DP must not lose to the CE replanner beyond Monte Carlo error on any cell. If it does, that is a bug.
- **Normalisation.** DP becomes the sigma > 0 normaliser only on the cells it covers. The CE replanner stays the normaliser elsewhere, as CLAUDE.md requires.
- **Placement.** This is a new planner, so it belongs in `ocmdp/baselines/` or `ocmdp/theory/`, which makes it an owner step.

### Sub-question 6: the instance family for B2

- **Primary instance:** Oz et al. 2014 (E. coli, 22 drugs, measured normalised MIC matrix, Table S2).
- **Grouping rule:** "target column as printed" in the paper's Table 1, giving K = 22, M = 6, n = (7, 5, 3, 3, 3, 1). The rule is the paper's own label; the other three rules in the inbox file are not used.
- **C from the data:** C[a, j] = mean normalised resistance to the drugs of group j among populations evolved on drug a, clipped to [0, 1]. Collateral sensitivity is floored at 0, which is a stated limitation because `CrossResistanceCoupling` requires non-negative entries.
- **H_Z kernel:** Z = cosine similarity of the rows of C.
- **Secondary replication, fixed now and descriptive only:** Lazar et al. 2014 (CC BY, 12 drugs), conditional on its Table 1 class list being verified.
- **Rejected sources:**
  - Imamovic 2018: CC BY-NC-ND blocks release of a derived matrix (rung 6), and it has one strain per drug.
  - Sakenova 2025: calls are inferred, categorical and undirected.
  - Maeda 2020: mostly non-antibiotic stressors.
  - Weaver/Mira landscapes: a single drug class.
- **Licence caveat on Oz:** its licence is CC BY-NC 3.0, so the rung-6 release would carry NC. That is your call.
- **Fixed with these choices:** kappa and delta stay at their defaults; the floor/ceiling exclusions and margin m are fixed in the preregistration before any dev gap on this instance is looked at.
- **Prerequisite:** you must first promote these data from inbox/.

### What experiments settle, instead of arguing

- **Sub-question 4: one dev launch.** PairedDrugs, sigma 0, dev seeds 0 to 4, both agents with one-hot embeddings and raw observations; one launch, roughly 15 minutes on Modal given the 747 to 916 s per job logged on 2026-11-16. Read off:
  - `runs_learned` per agent;
  - `policy_entropy.within_class` (does SAC break the log 2 symmetry?);
  - the temperature path (does SAC's alpha leave the clamp floor?).
  - If neither agent learns without C, sub-question 4 flips, and B2 has to give C to every agent equally.
  - Needs: a flag in `experiments/train_agent.py` and an embedding option in `ocmdp/agents/` (both within proceed authority), plus an assertion that the raw observation carries the pending stack (`markov_observation`).
- **Sub-question 6, partly: planner tractability at K = 22.** Run `experiments/planner_restarts.py` on the Oz-derived instance before it is used as a denominator.
- **Sub-question 5, partly: is DP tractable?** Time one cell (d = 2) after it is built.
- **Sub-question 1, partly:** B1 itself is cheap. It needs one fixed-alpha grid on dev before any test-seed preregistration.

## 2. Criterion by criterion

**1. Novelty against the filed prior art.** Each kept claim, against its closest paper:
- **Claim A vs Tennenholtz et al. 2023 (DCMDP):** they bound regret for learning with a history-drawn latent context. A instead splits the value of state feedback over committed schedules into its certainty-equivalence and true-feedback parts, as a function of delay, memory and noise, and predicts the sign-change boundary at held-out parameters.
  - Against Rank et al. 2024 the response here is within one trajectory, with an explicit delay, rather than across deployments.
  - Against rebounding bandits (Leqi et al. 2021), which show cyclic play optimal for identical deterministic arms, A asks when feedback beats cycling under noise with cross-coupling.
  - The qualitative point that feedback matters under noise is not new (Fischer et al. 2015, "to control one must monitor"; Bertsekas). Only the decomposition and the predicted boundary are claimed.
- **B2 vs Baram et al. 2021 (MinRed):** MinRed counts actions as redundant only when they lead to identical next states. With partial cross-resistance every drug has a distinct row, so MinRed sees no redundancy and reduces to SAC. H_Z gives partial credit. B2 tests whether that partial credit changes efficacy.
- **B1** claims nothing new. It is labelled as a check of a known decomposition (the chain rule, MinRed's form) on this dynamics.
- **Per-agent target scaling** is credited to Mohammadpour et al. 2024 and not claimed.

**2. Falsifiability and severity.**
- **A:** predictions are made at held-out (kappa, delta, horizon) cells and scored against the dev-map carry-over null. The DP check can fail.
- **B1:** a point prediction, with equivalence against the predicted value.
- **B2:** two-sided, with TOST at margin m.
- **Power for B2 at 100 test seeds.** Logged paired OC-SAC minus SAC standard deviations at sigma 0 range from 0.0023 (`runs/train/train_20261116_233400_32604.json`, `runs/train/train_20261123_190750_20538.json`) to 0.0177 (`runs/train/train_20261123_190717_20498.json`); at sigma 0.2 it is 0.0092 (`runs/train/train_20261026_145126_52656.json`). By arithmetic (2.8·sd/√100), the minimum detectable effect at 80% power, 5% two-sided, is about 0.0006 to 0.005 expected failure rate. A margin m ≥ 0.005 is therefore resolvable even at the worst logged spread. These are five-seed SD estimates, and measuring the across-agent correlation on val comes first.
- **Budget.** One launch carries all seeds of a pair on Modal. Tuning at parity, about 6 comparators × 8 configurations, is about 48 launches, a little over one day at 40 per day.

**3. Value per unit of work.**
- The planner and DP work needs no training.
- B1 is one grid of runs.
- B2 is one instance and one test run.
- The synthetic KL sweep is dropped, removing a grid of hand-built instances whose exact-class outcome is derivable.
- The largest single cost is rung 1, which CLAUDE.md requires anyway.

**4. Survives nulls.**
- If A's prediction fails, the decomposition still states how much of the measured value of feedback is a CE artefact.
- If B1 holds, it explains the logged tie as structural. If it fails, that is an optimisation finding.
- A B2 null within the margin is an equivalence result: action entropy suffices even with partial cross-resistance. That is the characterisation CLAUDE.md asks for.

**5. Comparator strength.**
- Every entropy comparator is listed with its strongest published form, including MinRed in both forms and the mechanism-level and two-level SAC a reviewer will ask for.
- None is weakened. Where no code exists (MinRed, Hybrid SAC), that is stated and the comparator is built from the paper.

**6. CLAUDE.md compliance.**
- Rung 1 comes before any rung-2+ agent claim.
- Instances, grouping rule, C mapping, kernel and margins are fixed before any gap is seen.
- Test seeds are used once per preregistration.
- Locked-path items go to you.
- **Tension I cannot resolve:** CLAUDE.md puts reproduction "before any new method work". I read Claim A (planners, heuristics, DP) as not being method work, so it runs alongside rung 1. If you read it otherwise, Claim A waits for rung 1.

## 3. Strongest objection, and whether it is fatal

A hostile reader can say:
- the OC-MDP is a DCMDP or augmented-state MDP;
- "feedback beats open-loop under noise" is textbook (Bertsekas) and already argued for evolving populations (Fischer et al. 2015);
- a predicted boundary on a two-drug toy is a workshop figure, not main-track work.

Two further facts make it worse:
- Maynard-Zhang et al. (arXiv 2610.33412, 24 Oct 2026) analyse open-loop, CE and optimal gaps by regime. It is not in the filed prior art.
- With noise-optimal references, the sign of (ii) is fixed by theorem. That leaves only magnitudes and the sign of the CE/greedy contrasts, which are partly properties of the planners.

Not fatal, in my judgement, if two things hold:
- the decomposition shows a true value of feedback that the derived theory predicts at held-out parameters better than the null;
- the paper words A as a characterisation of this class, not as a new control principle.

It may still fall short of NeurIPS main-track novelty. That is a risk to the venue, not to the soundness of the work.

The secondary objection concerns sub-question 4: without C, learning may fail. The dev launch catches that before any test seed is spent.

## 4. What would show I was wrong

- **Dev launch without C:** if `runs_learned` collapses for either agent, or SAC never breaks within-class symmetry, the answer to sub-question 4 is wrong.
- **DP vs CE replanner on small cells:**
  - if they are equal within Monte Carlo error everywhere, the DP adds certification only, and (iii) is empty;
  - if DP is far better on cells of the existing dev map, those contrasts are planner artefacts. That would be a finding that invalidates an existing table, which is a stop condition.
- **Claim A scoring:** if the score E is not better than E_null at the preregistered margin, A fails and is reported as a failure.
- **B1:** if the measured gap at fixed alpha leaves the equivalence band around the prediction, the identity does not govern trained agents.
- **B2 temperatures:** if every agent's evaluation temperature on the Oz instance sits at the clamp floor on dev, the entropy objective is inactive at evaluation. B2's null is then structural, and the preregistration should say so in advance.
- **Literature:** a PRE-EMPTS hit on a delayed, occupancy-coupled kernel with a feedback-vs-schedule analysis would mean A's novelty sentence is wrong.

## 5. Prior work

**Already filed:**
- Baram et al. UAI 2021 (MinRed, the closest to B);
- Tennenholtz et al. ICML 2023 and Rank et al. UAI 2024 (closest to the formulation);
- Leqi et al. NeurIPS 2021; Clerici et al. 2023;
- Mohammadpour et al. ICLR 2024 (target scaling);
- Lee and Blumberg 2026 (abx_amr_simulator: no delay, no RL numbers);
- Weaver et al. PNAS 2024 (rung 1).

**Not yet filed** (for a gatherer to file to inbox/; tags are my assessment):
- [Maynard-Zhang, Choi, Sattar, Jedra, Dean, Fazel, "Performance Gaps Between Open-Loop, Certainty-Equivalent, and Optimal Control Under Action-Dependent Observations", arXiv 2610.33412 (Oct 2026)](https://arxiv.org/abs/2610.33412). LQ with bilinear action-dependent observations; three regimes by an information parameter q. RELATED, and the closest on the control question. Abstract page only.
- [Fischer, Vazquez-Garcia, Mustonen, "The value of monitoring to control evolving populations", PNAS 112:1007 (2015)](https://arxiv.org/pdf/1406.6957). Stochastic optimal control of evolving populations; "to control one must monitor". RELATED.
- [Bergstrom, Lo, Lipsitch, PNAS 101:13285 (2004)](https://pmc.ncbi.nlm.nih.gov/articles/PMC516561). Mixing beats cycling in a hospital model; mixing is the clinical counterpart of maximal drug entropy. RELATED, and a heuristic comparator.
- [Lässig and Mustonen, "Eco-evolutionary control of pathogens", PNAS 117:19694 (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443876). RELATED.
- [No Prior Mask, AAAI 2024, arXiv 2312.06258](https://arxiv.org/pdf/2312.06258v1). Redundant-action elimination. BASELINE candidate.
- Bertsekas, Dynamic Programming and Optimal Control Vol. I, §6.2 (open-loop feedback control and certainty-equivalent control). Cited from memory; the "at least as good as optimal open-loop" statement was not verified this session ([search](https://arxiv.org/pdf/2002.09478) did not confirm it).
- Stooke, Achiam, Abbeel, ICML 2020 (PID Lagrangian). From memory, not verified.

## Owner steps (locked paths, CLAUDE.md, promotion)

1. Promote the Oz 2014 data (and Lazar 2014, if used) and decide on the NC licence.
2. Decide where the DP and SAA reference lives (`ocmdp/baselines/` or `ocmdp/theory/`).
3. Rule on whether Claim A may run alongside rung 1.
4. Decide whether CLAUDE.md's target rule gets the published decoupled form, per-state under masks. It is the same as c·log K when there is no mask.
5. Make the PRE-EMPTS call on arXiv 2610.33412.
6. The t-table in `ocmdp/eval/vv.py`, already noted in decisions.

## Relevant files

- arena/2026-11-23-scope-criteria.md
- decisions/2026-11-23.md
- inbox/2026-11-16-mechanism-level-sac.md
- inbox/2026-11-16-oz-2014-ecoli-22-drug-cross-resistance.md
- inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md
- experiments/predict_entropy_gap.py
- experiments/score_phase_predictions.py
- ocmdp/envs/wrappers.py (where the true C reaches the observation)
- ocmdp/agents/oc_sac.py (where the true C reaches the embeddings)
- runs/train/train_20261123_190717_20498.json, runs/train/train_20261123_190750_20538.json, runs/train/train_20261116_233400_32604.json, runs/train/train_20261026_145126_52656.json (source of the paired SDs used for power)

Sources:
- https://arxiv.org/abs/2610.33412
- https://arxiv.org/pdf/1406.6957
- https://pmc.ncbi.nlm.nih.gov/articles/PMC516561
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7443876
- https://arxiv.org/pdf/2312.06258v1
- https://ar5iv.labs.arxiv.org/html/2102.11329
- https://arxiv.org/pdf/2002.09478
