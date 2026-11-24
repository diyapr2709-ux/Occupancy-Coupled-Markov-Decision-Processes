# Proposal 1

## Proposal: rung 1 reproduces Weaver et al. 2024 by reimplementing it here, checking the environment exactly before training any agent, and using the authors' code only to diagnose a failure

No number below comes from runs/. Every figure is either "reported by" the paper's authors or arithmetic used to set a threshold. Read-only sources I looked at this session: the PMC full text, `evodm/evol_game.py` and `evodm/exp.py` at commit 16335ce, and `R/test_harness_noise.R` and `R/prep_figs.R` from rl_cycling. All of these were read through an extraction tool, so they still need to be confirmed against the raw source.

### 1. Recommendation

**Q1. Target paper and quantities.** Weaver et al., PNAS 2024 (DOI 10.1073/pnas.2303165121), base case: 15 drugs, N = 4, the Mira 2015 landscapes, start at genotype 0, 20-step episodes, 500 training episodes. The checks fall into two layers: the environment alone (P1, P2), then the trained agents (P3, P4 and the secondary checks S1, S2).

Primary checks (these decide whether rung 1 has been reached):
- **P1. Planner check, no learning.** Solve the MDP by backward induction with horizon 20 and discount 0.99 (the code's value). Compute exactly how often the optimal policy prescribes cefotaxime (CTX), weighted by how often each state is visited. Reported: "more than 50% of time steps".
- **P2. Random-cycling baseline, no learning.** The exact expected mean population fitness when a drug is drawn uniformly at random each step. Reported: 1.88 at noise modifier 40. The noise is added only to the sensor reading (`sensor_fitness`); the recorded fitness is the true one. A random policy ignores observations, so this number depends only on the landscapes, the dynamics, the start state and the landscape scale.
- **P3. RL-genotype count.** Number of replicates in which RL-genotype beats random cycling and all 105 two-drug cycles. Reported: 100 of 100.
- **P4. RL-fit count.** Number of replicates in which RL-fit beats random cycling. Reported: 98 of 100.

Secondary checks (reported, not gating):
- **S1. Policy similarity.** Pearson correlation of edge weights between each agent and the MDP. Reported: 0.96 for RL-genotype and 0.82 for RL-fit.
- **S2. Noise panel.** RL-fit mean fitness under noise. Reported: 1.41 against 1.88 for random.

**Q2. Route: reimplement in this repository, with the authors' code used only for diagnosis.**
- **Landscape values.** Transcribe them from Table 4 of Mira et al. 2015 (CC BY), not from the GPL source. Two independent transcriptions are diffed, then compared against the list form of `define_mira_landscapes` at 16335ce. Every disagreement is listed, including the known ZOX one (0.5 vs 0.805).
- **Configuration.** Values the paper leaves out are taken from the pinned commit 16335ce and cited by file and line. These are facts, so no code is copied:
  - hyperparameters and the network, including Dropout 0.2;
  - plain DQN targets;
  - `train_freq` 100;
  - the epsilon schedule;
  - fitness recorded before the transition.
- **Learner.** The DQN is written in torch to match what the code does, including Keras defaults: glorot_uniform initialisation and Adam epsilon 1e-7. It is not strengthened. This is a reproduction, not a comparator; the strongest honest DQN belongs to rung 4.
- **Seeds and evaluation.** Replicates use dev seeds 0–99, with every random source seeded. Each replicate gets 500 greedy evaluation episodes (epsilon 0), matching the `ep_number > 500` filter in `prep_figs.R`. Agents and comparators share the same environment seeds in evaluation.
- **Authors' code.** It is not run in the primary route. It may be run only after a failure, as described under Q4.
- **First step, before the preregistration is frozen: a half-day read-only look** at 16335ce and the R harness. It answers factual questions; it does not tune anything:
  - (a) the landscape scale each condition ran on. Code defaults suggest the learner normalises (`evol_deepmind normalize_drugs=True`, never overridden by the harness) while `policy_sweep` defaults to False. That would make 1.88, which is above 1, impossible for a normalised random policy, and it could mean the two-drug comparison mixed scales;
  - (b) which column and episode count produced 1.88 and 1.41. The harness at HEAD sweeps only `noise_vec <- c(100)` with `iter = 10`;
  - (c) the transition rule when no fitter neighbour exists;
  - (d) the starting genotypes in the two-drug sweep. `policy_sweep` loops over all 16;
  - (e) the `done`/`DONE` flag;
  - (f) the meaning of `train_freq`.
  
  Run the queries in inbox/2026-11-23-application-queries.md that were cut short in the same half-day. My two searches found no other RL drug-cycling paper with numeric results. A 2025 bioRxiv Q-learning paper models copper stress, not antibiotics.

**Q3. Tolerances, fixed in the preregistration before any run.**
- **P1 passes** if the visitation-weighted CTX fraction is above 0.5. The unweighted share over the (state, time) table is reported alongside.
- **P2 passes** if |E − 1.88| ≤ 0.005 + 3·σ_ep/√100.
  - E is the exact expectation from the Markov chain.
  - σ_ep is the exact standard deviation of one episode's mean fitness, from the same chain.
  - 100 is the smallest episode count anywhere in the authors' harness, so it gives the widest band, which is deliberately conservative.
  - Rule fixed now: P2 is primary only if the read-only look identifies how 1.88 was computed and on which scale. Otherwise it is recorded as "not identifiable" before any run, not after.
- **"Beats" for one replicate** means its mean true fitness over its 500 evaluation episodes is lower than the comparator's mean on the same episode seeds. All conditions are compared on one landscape scale, the learner's. If the look shows the published comparison mixed scales, the mixed-scale count is reported as a description only, and that finding is itself a stop.
- **P3 and P4 pass** if a two-sided Fisher exact test of our k/100 against the published count gives p ≥ 0.05.
  - For P3 against 100/100, my hypergeometric arithmetic gives p ≈ 0.059 at k = 95 and p ≈ 0.029 at k = 94, so the pass mark is k ≥ 95.
  - For P4 against 98/100, a rough approximation puts the pass mark near k ≥ 91.
  - A script computes the exact cut-offs and they are frozen in the preregistration. My numbers are approximations.
- **S1 passes** if the difference in Fisher-z is at most 0.2. That is about [0.94, 0.97] for 0.96 and about [0.74, 0.88] for 0.82. Edge weights are transition fractions pooled over all replicates' evaluation trajectories. The MDP's weights come from its exact visitation distribution. The correlation is taken over the union of edges that are nonzero in either.
- **S2 passes** if, with noise sd = 0.05 × 40 applied to the sensor only (the code's definition, which contradicts the paper's 0.05·σ²) and 10 replicates (the figure caption), the 95% t-interval over replicate means contains 1.41. This test is weak, and the preregistration says so.

**Q4. If the numbers do not reproduce.**
- Stop. Report each discrepancy with its size: k with its Clopper-Pearson interval and the Fisher p; E − 1.88 with σ_ep; r with its z-difference.
- Do not tune the reimplementation.
- Check order is P1 and P2 first, then P3, P4, S1 and S2. A failure in the environment checks stops everything before any training.
- Diagnosis is a separate step and needs your approval, because it means installing external code. Run evo_dm at 16335ce in a throwaway container outside the repository:
  - python 3.10, TensorFlow 2.11 CPU, wheels pinned by hash;
  - the repository not mounted;
  - network cut off after installation;
  - the code reviewed before it runs.
  
  The purpose is to tell whether their code reproduces their own number. That locates the fault in our reimplementation or in the publication; it does not repair anything.
- Any change to the reimplementation afterwards needs a new dated preregistration and is labelled post hoc. It is never presented as the reproduction.
- Rungs 2 and 3 do not start on an unmatched reproduction.

**Q5. Path to rungs 2 and 3.**
- **Rung 2.** OC-SAC and SAC on the reproduced environment, unchanged, with the genotype one-hot as observation.
  - Equal training budget, tuned at parity on dev seeds.
  - Entropy targets c·log M and c·log K, with c chosen on val seeds.
  - Proposed drug-to-mechanism map, taken from WHO ATC groupings so that results cannot have chosen it:
    - CA: AMP, AM;
    - CR: AMC, SAM, TZP;
    - DC: CEC, CXM, CTT;
    - DD: CTX, ZOX, CRO, CAZ, CPD;
    - DE: FEP, CPR.
    
    That gives M = 5 with class sizes (2, 3, 3, 5, 2), versus K = 15 drugs. Adopting it is a scope decision for you.
  - Expected result: equivalence, judged by an equivalence test (TOST) with a margin set in the rung-2 preregistration. The reproduced DQN and the MDP are reported as reference points.
- **Rung 3.** The same landscapes, transition rule, 15 drugs and horizon, plus one change: each drug's fitness is modulated by the delayed, exponentially weighted occupancy of its mechanism class.
  - A test asserts that with coupling strength κ = 0, trajectories are identical to rung 1 under the same seeds.
  - The observation carries the pending occupancy, and this is asserted.
  - "The published approach" is the reproduced DQN without changes, alongside a strongest-honest DQN that sees the occupancy.
  - The MDP is no longer exact on the augmented state, so it is called the noise-free planner.
  - Effect delay must be kept distinct from the paper's measurement delay.
  - The new environment code goes in ocmdp/envs/ and the DQN in ocmdp/baselines/. Both are stop-for-you edits.

### 2. Against each criterion
1. **Shared ground.** Every gating quantity is one the authors state in the text. Nothing is digitised from figures.
2. **A tolerance fixed now that can fail.**
   - P2 fails on a wrong scale (a normalised policy cannot reach 1.88), a wrong landscape or wrong dynamics.
   - P3 fails at k ≤ 94 out of 100.
   - P1 is binary but exact.
   - Replicate counts are fixed: 100 dev seeds per agent, 10 for the noise panel.
3. **Auditability.** The rebuild uses CC-BY data and facts read from GPL source, so no GPL code enters this repository. Every source of randomness is seeded. P1 and P2 are deterministic and need no seeds. Every configuration value cites the commit and line it came from.
4. **Cost.** About 5 working days:
   - 0.5: read-only look and finishing the queries;
   - 1: double transcription of the landscapes;
   - 1: environment checks P1 and P2;
   - 1.5: DQN and evaluation harness;
   - 0.5: preregistration;
   - one launched command per condition.
   
   Stall risks: where 1.88 came from (handled by the conditional rule for P2), how readable Table 4 is, and Keras/torch differences (handled by matching the defaults).
5. **Use for rungs 2 and 3.** The environment is ours, so OC-SAC, SAC and the coupling plug straight into it, and κ = 0 is the bridge back to rung 1. Running the authors' code could never host OC-SAC without GPL consequences.
6. **Security.** The primary route runs no downloaded code. The diagnostic run is gated on your approval and isolated.

### 3. Strongest objection
The reported quantities barely test an RL reproduction.
- 100/100 and 98/100 against random and two-drug cycles are near the ceiling. Almost any competent policy on a 16-genotype landscape may pass, so a wrong DQN could still "reproduce".
- The only continuous numbers (1.41 and 1.88) come from a panel with conflicting replicate counts, a noise definition that differs between paper and code, and a scale that is unclear.

**Not fatal.** The criteria rank shared ground first, and this is everything the paper reports. P1, P2 and S1 test policy structure and the environment more sharply. The sentence this licenses stays narrow: "reproduces the published replicate-count outcomes and environment baseline". It does not say the DQN was reproduced.

There is a tension between criteria 1 and 2 rather than a contradiction: shared ground restricts the tests to quantities that can fail but separate good agents from bad ones poorly.

Part of the decision is factual, not a matter of design: whether 1.88 can be identified at all. The read-only look settles that.

### 4. What would show this proposal is wrong
- **A myopic policy also passes P3.** Compute, without learning, a one-step greedy policy that picks the drug with the lowest immediate expected fitness. If it beats random and all two-drug cycles, P3 cannot tell good RL from a simple rule. This is a diagnostic in the preregistration, not a gate.
- **P3 is a coin flip.** If the pass rate for k sits near the cut-off, the conclusion depends on seed luck.
- **The look cannot pin down the scale or where 1.88 came from.** Then P2 and S2 drop out and shared ground narrows to counts.
- **The remaining queries find a better target**: an RL drug-cycling paper with numeric tables and seeds.
- **The work takes more than about 8 days.**

### 5. Prior work
- **Weaver et al. 2024**, published version and preprint. I found no independent reproduction of it and no follow-up paper.
- **Data and same-landscape models:**
  - Mira et al. 2015, the source data;
  - Nichol et al. 2015, PLoS Comp Biol, which models sequential therapy as a Markov chain over the same landscapes. It could serve as an independent environment check, but only if you widen the scope.
- **Related decision-process work:**
  - Maltas and Wood 2019, PLoS Biology, an MDP over collateral sensitivity;
  - Engelhardt 2020, JMLR.
- **Reproducibility practice:**
  - Henderson et al., AAAI 2018, "Deep RL that Matters" (arXiv 1709.06560), on seed variance;
  - Agarwal et al., NeurIPS 2021, rliable (arXiv 2108.13264).

Sources:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9882109
- https://www.biorxiv.org/content/10.1101/2023.01.12.523765.full.pdf
- https://raw.githubusercontent.com/DavisWeaver/evo_dm/16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0/evodm/evol_game.py
- https://raw.githubusercontent.com/DavisWeaver/evo_dm/16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0/evodm/exp.py
- https://raw.githubusercontent.com/DavisWeaver/rl_cycling/master/R/test_harness_noise.R
- https://raw.githubusercontent.com/DavisWeaver/rl_cycling/master/R/prep_figs.R
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567305/
- https://www.biorxiv.org/content/10.64898/2025.12.18.695270.full.pdf

Repository files read:
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/arena/2026-11-23-reproduction-criteria.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-23-application-queries.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-23-reviewer-leads.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-16-prior-art-antibiotic-rl.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/seeds.yaml
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/budget.yaml

Nothing was written and no code was run.
