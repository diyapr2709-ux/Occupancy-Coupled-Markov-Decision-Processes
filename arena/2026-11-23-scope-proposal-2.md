# Proposal 2

## 1. Recommendation

**Summary.** The paper is built on the problem class: the theory of occupancy-coupled MDPs and a phase diagram, with its boundary derived in advance, of when acting on the observed state beats a committed schedule. The entropy claim is cut down to a one-section description of an exact identity and a test of what it predicts. It is not presented as a method that wins.

**Sub-question 1: narrow the entropy claim.** Drop "entropy over mechanisms preserves drug efficacy better than entropy over drugs" as a headline. Replace it with:

> Claim E: "When drugs within a mechanism class are interchangeable, drug-entropy SAC is class-entropy SAC plus the reward bias alpha(s) log n_class(a). The paired efficacy gap between the two agents equals, within a margin set in advance, the gap the mean-field model predicts between the two soft-optimal schedules at the temperatures the agents actually reached."

- **Primary rule:** two one-sided tests (TOST). The 90% t-interval of the measured gap minus the predicted gap must lie inside ±m on each instance of the family. The equivalence test option (`--equivalence-margin`) and `experiments/predict_entropy_gap.py` already exist.
- **Secondary, dev only, no claim:** partial cross-resistance with the similarity-sensitive entropy H_Z. This is the one setting where the objective, not the parametrisation, could make a difference, because no mechanism-level twin exists there. It is promoted only through a new dated preregistration.
- Changing a claim is the owner's call. This is proposed, not adopted.

**Sub-question 2: what carries the paper, in order.**
- **(0)** Rung 1, the reproduction (decided by the separate reproduction arena), runs in parallel. Every agent claim waits for it.
- **(1)** The confirmatory phase-diagram test at held-out settings (other kappa, delta, horizon), with the derivation, the theory-free competitor (`null_phase_predictions.py`) and the scorer fixed in a preregistration first.
- **(2)** A check that the planner is good enough to be the reference under noise (sub-question 5).
- **(3)** The Claim E test.
- **(4)** Rungs 2 and 3 on the reproduced setup.
- **(5)** The completeness batch.

The paper is framed as "occupancy-coupled MDPs: when feedback beats commitment", with the entropy identity as one section.

*Reading to flag:* CLAUDE.md requires rung 1 before "new method work". The phase diagram compares existing planners and heuristics, so I read it as not method work. The owner should confirm that reading, not have it assumed.

**Sub-question 3: comparators.**

For the entropy test:
- **Discrete SAC** (Christodoulou, arXiv 1910.07207; code: p-christ repository and CleanRL `sac_atari.py`).
  - Arm 1: the decoupled target rule (Mohammadpour, Frejinger and Bacon, ICLR 2024; code: SobhanMP/decoupled-soft-RL).
  - Arm 2: a fixed temperature tuned at parity, with the SD-SAC stabilisers (arXiv 2209.10081; code: coldsummerday/SD-SAC).
- **MinRed SAC in its learned-inverse-model form** (Baram et al., UAI 2021). No reference code; the paper's Appendix B Stable-Baselines3 settings are the starting point. Its deterministic form with the class map given is identical to OC-SAC, so it is stated as the same objective, not listed as a separate comparator.
- **Two-level (Hybrid) SAC** with a learned within-class choice at zero within-class temperature (Delalleau et al., arXiv 1912.11077; no code). Mechanism-level SAC that samples uniformly within a class is its special case.
- **Zero-temperature control:** same networks, entropy off. Required because the optimum is close to deterministic.
- **Target-entropy annealing** (arXiv 2112.02852; no code).
- **DQN / Double DQN** (CleanRL `dqn.py`). It is the algorithm of the reproduction target, so rungs 2 and 3 need it.
- "SAC with action entropy minus expected log class size" is OC-SAC by the identity. It is named as such and not counted as a comparator.

For the late completeness batch (plan rank 7):
- **PPO with an entropy bonus** (Schulman et al. 2017; CleanRL `ppo.py`; RL Zoo configurations as the starting grid).
- **PPO-Lagrangian** (safety-starter-agents; OmniSafe PPOLag), with a PID-controlled multiplier (Stooke et al., ICML 2020; code location unverified). Include it only if a preregistered constraint can actually bind. Otherwise drop it and say why, since the reserve clause currently cannot fail.
- **Recurrent SAC without the pending-occupancy feature** (no source found).
- **Non-learning comparators:**
  - SW-UCB and D-UCB (Garivier and Moulines; SMPyBandits, MIT licence) and EXP3.S;
  - informed switching (Kouyos et al., PLoS Comput Biol 2011), which is the literature's counterpart of myopic greedy;
  - cycling and mixing (Beardmore et al., Mol Biol Evol 2017);
  - the certainty-equivalent replanner;
  - the noise-optimal reference from sub-question 5.

**Sub-question 4: true cross-resistance matrix?** The primary condition does not give it to the networks.
- **Primary:** both agents get drug-indexed observations (no MechanismObservation wrapper, which is a change in experiments/ only), one-hot drug embeddings and K-way heads. OC-SAC uses C only inside its entropy term. Under this setup the two agents differ only in where entropy is measured. With C given to the networks, the tie is structural by the identity.
- **Secondary:** the C-given condition, where the identity predicts the gap exactly.
- The observation must still carry the pending occupancy, and that is asserted, not assumed.
- **Settled by one dev launch, not argued:**
  - both agents, both conditions, PairedDrugs plus one unequal-class instance, σ = 0, dev seeds 0–9, temperatures logged;
  - outputs: runs_learned, the paired gap against the predicted gap, and within-class entropy.

**Sub-question 5: a reference that is optimal under noise.** Yes, but behind a gate.
- **Gate (minutes, planner only):** on the dev cells with σ > 0, compute the perfect-information relaxation bound (Brown, Smith and Sun, Operations Research 2010). This means planning each evaluation episode with its noise path revealed, which gives an upper bound on any policy that cannot see the future. Compare the certainty-equivalent replanner with it.
  - If the gap is inside the phase test's margin, the current planner is certified good enough and no new reference is built.
  - Otherwise the gate is inconclusive (that bound is known to be loose) and the next step runs.
- **Next step:** dynamic programming on a discretised state for M = 2 and delay ≤ 2. The state is (r1, r2, mu, the pending queue, t), with quarter-level class shares as actions and Gauss-Hermite quadrature for the noise. It is verified two ways: (a) the value converges as the grid is refined, and (b) its policy, simulated in the real environment, sits between the replanner and the bound.
- **Rejected as the reference:** scenario-tree MPC. It is not optimal and comes with no certificate. It may still appear as a comparator.
- The reference belongs in ocmdp/baselines/, so building and placing it is an owner step.

**Sub-question 6: instance family.**
- **Source:** Oz et al. 2014, Table 1. Its class vectors were read from the paper itself, the MICs are measured and directed, and its CC BY-NC 3.0 licence allows non-commercial release. Imamovic 2018 is CC BY-NC-ND, so a derived instance could not be released. Sakenova 2025's class vector was not obtained. Lazar 2014's classes are nearly equal.
- **Rule fixed now:** keep all 22 drugs, no subsetting, and run every grouping the table prints. There is no choice among them afterwards.
  - drug target: n = (7,5,3,3,3,1)
  - class crossed with target: n = (5,4,3,3,3,3,1)
  - cellular process: n = (12,3,3,3,1)
  - plus the PairedDrugs (2,2) anchor
- **Guard:** the planner's restart-agreement check at K = 22 must pass before anything runs.
- **Soft instance (secondary):** the drug-by-drug mean of Oz's Table S2, clipped to [0,1] because the locked coupling cannot represent collateral sensitivity. That clipping is a stated limitation, and Table S2 has not been downloaded (unverified).

## 2. Criterion by criterion

**1. Novelty against the filed prior art.** One sentence per kept claim:
- *Phase diagram vs Tennenholtz et al. 2023 (closest):* their paper gives learning algorithms and regret bounds for history-dependent contexts, but no condition, as a function of delay, memory and noise, under which a committed schedule is optimal, and no boundary derived in advance and tested at held-out parameters.
- *Phase diagram vs Rank et al. 2024:* their response plays out over deployment rounds with no delay and studies stability. Here the coupling acts within one trajectory and the question is optimal closed-loop control.
- *Phase diagram vs rebounding bandits and Kouyos 2011:* those papers establish the σ = 0 face (cycling is optimal) and "feedback wins in a ward", not the interior boundary.
- *Claim E vs Baram et al. 2021:* MinRed shows that action entropy over-weights redundant actions. It does not predict the efficacy cost of that bias under delayed occupancy coupling or test such a prediction. The novelty here is admittedly thin, which is why Claim E is a section and not the contribution.
- *Release:* abx_amr_simulator pre-empts "first benchmark". Rung 6 can claim only "an instance family with delay, exponential memory, a planner and frozen seed tiers".

**2. Falsifiability and severity.**
- The phase test can fail in both directions through the `planned_once_minus_greedy` contrast, and the theory must beat the theory-free competitor (E − E_null). It needs planners only; the logged two-cell pilot gives about 80 episodes per cell for a half-width near 0.007, so it fits inside 40 launches a day.
- Claim E can fail. A measured gap outside the predicted band means the bias explains nothing, which is itself a finding.
- Measured noise between seeds (5-seed paired SDs at σ = 0):

  | Report | Paired SD | Note |
  |---|---|---|
  | train_20261116_233400_32604 | 0.0023 | local flat target |
  | train_20261123_184321_12020 | 0.0049 | Modal, unpinned |
  | train_20261123_184953_15052 | 0.0047 | Modal, unpinned |
  | train_20261123_190717_20498 | 0.0177 | Modal, pinned; one OC-SAC seed did not learn (4/5) |
  | train_20261123_190750_20538 | 0.0024 | zero target |

- With n ≈ 8(SD/Δ)² + 2, an SD of 0.0177 needs about 28 seeds for Δ = 0.01 but about 106 for Δ = 0.005, which is more than the 100 test seeds. This is planning arithmetic, not a result. So the Claim E margin cannot go below about 0.005–0.01 unless the 10-seed dev launch shows a smaller SD. That limit should be stated in the preregistration.

**3. Value per unit of work.** The phase diagram needs no training. The identity is already verified. The prediction tool and the scorer exist. The one large new build, the dynamic-programming reference, is gated behind a check that takes minutes.

**4. Survives null results.**
- If the phase test fails or ties the theory-free competitor, the result reads "the boundary is not predictable beyond interpolation", with certified references.
- If Claim E is confirmed, the entropy-space difference is fully explained by a known bias.
- If Claim E fails, there is a parametrisation or optimisation effect beyond the objective, which is a finding.

**5. Comparator strength.** Every entropy comparator has a published configuration or the nearest available one, with code status stated. The temperature rule that CLAUDE.md fixes makes SAC's temperature inert, a weakening imposed by protocol (see the conflict below).

**6. CLAUDE.md compliance.**
- Rung 1 comes before any agent claim.
- No search over instances: every printed grouping rule is run.
- The margins, cells and seed counts are preregistered by the owner.
- Locked directories are touched only as owner steps.

**Conflict to flag, not resolve:** CLAUDE.md's Protocol fixes c log M and c log K, while criterion 5 demands the strongest comparator. The logged temperatures at the clamp floor mean the protocol weakens SAC. Only the owner can resolve this, through the temperature arena and an edit to CLAUDE.md.

## 3. Strongest objection

"The value of closed-loop over open-loop control under noise is textbook stochastic control, and a phase diagram on a toy model is not NeurIPS main-track work."

It is not fatal if the derived boundary is non-trivial in (delay, memory) and beats the theory-free competitor at held-out settings. It is fatal to the main-track framing, though not to publication, if E − E_null ≈ 0. A second risk: if the gate fails and the dynamic-programming reference is never built, a boundary measured with certainty-equivalent planners may be an artefact of those planners.

## 4. What would show this proposal wrong

- E − E_null and pass / fail / inconclusive on the held-out cells.
- The gap between the certainty-equivalent replanner and the perfect-information bound, against the phase test's margin.
- From the dev launch for sub-question 4: runs_learned without C, the measured gap minus the predicted gap, and the paired SD at 10 seeds. An SD above about 0.02 makes Claim E unpowered and argues for dropping it entirely.
- Planner restart agreement at K = 22.
- A PRE-EMPTS hit from the application searches that were never run (inbox/2026-11-23-application-queries.md).

## 5. Prior work

- Baram, Tennenholtz and Mannor, UAI 2021, https://proceedings.mlr.press/v161/baram21a.html — the follow-up Zhong et al., AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/29652 , was seen only in a search snippet and is unverified.
- Tennenholtz, Merlis, Shani, Mladenov and Boutilier, ICML 2023, https://proceedings.mlr.press/v202/tennenholtz23a.html — the inbox file's author list should be corrected to this one.
- Brown, Smith and Sun 2010, information relaxations, https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/Information_Relaxations_and_Duality_in_Stochastic_Dynamic_Programs.pdf
- Kouyos, Abel zur Wiesch and Bonhoeffer, PLoS Comput Biol 2011, informed switching beats cycling and mixing: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3048378/
- Allocation-protocol criteria, 2021, a near-optimal switching time: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8941386/ — snippet only, unverified.
- Beardmore et al. 2017: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5400377/
- Stooke, Achiam and Abbeel, ICML 2020: https://proceedings.mlr.press/v119/stooke20a.html — code location unverified.
- All other sources are as filed in the inbox files under inbox/.

Key repository files:
- arena/2026-11-23-scope-criteria.md
- experiments/predict_entropy_gap.py
- experiments/score_phase_predictions.py
- experiments/null_phase_predictions.py
- ocmdp/core/coupling.py (C is restricted to [0,1], so collateral sensitivity cannot be represented)
