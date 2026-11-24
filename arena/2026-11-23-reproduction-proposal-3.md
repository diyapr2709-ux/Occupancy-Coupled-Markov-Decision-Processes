# Proposal 3

I recommend reproducing Weaver et al. (PNAS 2024) by reimplementing it from the paper. The gate has two parts, both fixed before any run: exact checks on the environment, then the authors' few stated numbers using all 100 dev seeds. Their released code would be run only to diagnose a mismatch, in isolation and only with your approval.

I read the arena file and all the listed material. I also opened the published article on PMC to check the stated numbers verbatim, and looked at Mira et al. 2015 and one other candidate. I wrote nothing and ran nothing.

## 1. Recommended option

### Target paper and quantities (sub-question 1)
- **Paper:** Weaver, King, Maltas, Scott, "Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance", PNAS 121(16): e2303165121 (2024).
- **No better candidate turned up.** One more search found only arXiv 2510.01808, which is optimisation over switching periods in a birth-death model, not RL, and has no code. abx_amr_simulator reports no RL results. The application search is still incomplete (inbox/2026-11-23-application-queries.md), so this is "best found", not "best existing".

**Gate 0: environment checks.** These are exact Markov-chain calculations with no learner, run before any training.
- **E1, landscape.** Transcribe the 240 growth rates from Mira et al. 2015 (PLoS ONE 10(5): e0122283, CC BY), Table 4, in two independent passes that must agree exactly. Use the S1/S2 TXT supporting files if they hold the values.
  - Compare against the list form of `define_mira_landscapes` at commit 16335ce, by reading the code only. Record every disagreement, including the known ZOX cell (0.5 vs 0.805).
  - Fixed now: the runs use the values on the code path the authors' harness actually executes (the list form, `as_dict=False`, if the read confirms it). Disagreements with Table 4 are reported. They do not count as a reproduction failure.
- **E2, MDP uses cefotaxime.** Solve the 20-step backward-induction MDP (discount 0.99, start genotype 0, tie-breaking as in their `backwards_induction`). The expected fraction of steps on CTX must be above 0.5 (paper Sec. 2.2).
- **E3, random cycling.** The exact expected mean population fitness under uniform random cycling (start genotype 0, 20 steps) is compared with the reported 1.88.
  - The policy never observes fitness, so measurement noise does not change this expectation.
  - Pass if |E − 1.88| ≤ 0.005 + 3·s/√5000. Here 0.005 is rounding, s is the exact standard deviation of the per-episode mean fitness from the same chain, and 5000 is their 10 replicates × 500 episodes.
  - This is the only absolute number the authors give for a non-learning policy. It ties our landscape and dynamics directly to their figure.

**Gate 1: learners.** Use dev seeds 0–99, one seed per replicate. Train 500 episodes × 20 steps, then run 500 greedy evaluation episodes. The metric is mean population fitness per episode, matching their `prep_figs.R` filter `ep_number > 500`.
- **L1.** RL-genotype has a lower mean than random cycling and than the best two-drug cycle, reported in 100/100 replicates.
- **L2.** RL-fit shows a "measurable decrease" against random, reported in 98/100.
  - The definition is taken from the authors' figure code for SI Fig. S1A.
  - If that code does not define it, the fallback, fixed now, is a one-sided Welch t-test over the 500 evaluation episodes against random's episodes at p < 0.05.
- **L3.** RL-fit at noise modifier 40 has mean population fitness 1.41, reported.

**Secondary quantities** (reported against fixed tolerances):
- **C1, C2.** Edge-weight Pearson correlation with the MDP: 0.96 for RL-genotype, 0.82 for RL-fit.
  - These become gating only if the read-only look (below) finds the edge-weight construction unambiguous in their figure code. That decision is recorded in the preregistration before any run.
- **D, delay.** Delay d ∈ {0, …, 5}, Fig. 6. Descriptive only, because "similar to random" has no stated bound.

### Route (sub-question 2)
**Main route: our own reimplementation.** It is written in this repository from the paper, with configuration values read from the pinned public code. No GPL code text is copied.
- **Components:** SSWM equal-probability transitions on 16 genotypes; random cycling; the two-drug cycles (all pairs, enumerated as their sweep does); finite-horizon backward induction; and the two DQNs as coded.
  - Plain DQN target.
  - RL-genotype network: Conv1D(64,3)×2, max-pool, flatten, dropout 0.2, dense 28.
  - RL-fit network: dense 64, dense 28. Its input is current fitness plus the one-hot previous action.
  - Adam, MSE loss.
- **Hyperparameters:** the values that are actually in effect when their R harness calls `evol_deepmind` at commit 16335ce. That is the last commit before publication, and it contains every feature the paper uses (delay was added in 46ec357).
  - These include learning rate, minibatch size, target period, gamma, epsilon schedule, `train_freq` and warm-up.
  - The base case is run with the harness's `noise TRUE`, modifier 1, sigma 0.5, if the read confirms that is what ran. The paper calls it noise-free, so the preregistration states the conflict and follows the code.
- **Location and approvals:**
  - Code goes in a new directory, for example `experiments/repro/weaver2024/`. Putting it under `ocmdp/envs/` or `ocmdp/baselines/` instead is a stop item for you.
  - Reported numbers go into `baselines/registry.md`, labelled "reported by".
  - The transcribed landscape is stored as a data file with its provenance (Mira 2015, CC BY).
  - Before any of this, you would need to promote inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md to references/, because nothing in inbox/ may be imported or cited.

**Short read-only look, done before freezing the tolerance.** It reads code and documents only, never outcomes. It settles:
- the effective hyperparameters and what `train_freq` means;
- the `self.DONE` / `self.done` flag;
- normalisation;
- which form of the ZOX value is used;
- how the noise modifier maps to f' = f + N(0, 0.05σ²);
- whether the reported fitness is the true or the noisy value;
- how the two-drug cycles are enumerated and estimated (100 episodes per pair);
- how the S1A "measurable decrease" and the edge weights are computed (figure code in rl_cycling at d68cf07, which has no licence, so read only);
- the SI Appendix PDF. My fetch of the PMC supplement returned only a download stub, so any hyperparameter table there is still "not found".

**Diagnostic arm, only if Gate 0 or Gate 1 fails, and only with your approval** (installing external code is a stop item):
- Run the authors' evo_dm at 16335ce, cloned outside the repository and reviewed first (setup.py, imports, the driver path).
- Run it in a container with pinned dependencies (Python 3.10, TensorFlow ~2.11) and `--network none`. Mount the source read-only and give it one output directory.
- Drive it with a small Python driver that passes the harness arguments, so R and reticulate are not needed.
- The repository holds only the Dockerfile, which fetches by commit hash, plus the driver. No GPL code enters the repository.

### Tolerance (sub-question 3)
All of this is written into a new dated preregistration before any run.
- **L1:** pass if our count k1/100 is not different from 100/100 by a two-sided Fisher exact test at α = 0.05. By my hand calculation, k1 = 95 gives p ≈ 0.06 (pass) and k1 = 94 gives p ≈ 0.03 (fail), so the threshold is about k1 ≥ 95. The preregistration records the exact threshold from `scipy.stats.fisher_exact`, computed before any run.
  - Comparators follow the authors' procedure, read from code. If that is unreadable, the fallback is the exact expectation.
- **L2:** the same test against 98/100. The hand-computed threshold is about k2 ≥ 91, to be computed exactly and recorded.
- **L3:** run all 100 dev replicates at noise modifier 40 and pass if the 100-replicate mean is within 1.41 ± 0.10.
  - Reasoning: the authors' 1.41 is a mean over 10 replicates with no dispersion reported. ±0.10 still requires reproducing at least about 79% of their reported 0.47 gap to random.
  - The report gives our interval and between-replicate standard deviation, and the false-discrepancy probability those imply.
- **C1, C2:** within ±0.30 in Fisher-z units, which is about [0.93, 0.98] and [0.70, 0.90].
- **Verdict:** REPRODUCED only if E2, E3, L1, L2 and L3 all pass (plus C1 and C2 if they were made gating). Anything else is NOT REPRODUCED, reported quantity by quantity.
- **Stated cost of these choices:** if their code and ours behave identically, the gate still shows a false discrepancy up to about 10–15% of the time. I accept that because a false failure costs a stop and a conversation, while a false pass contaminates rungs 2 and 3.

### If it does not reproduce (sub-question 4)
1. The script writes the run report to runs/. It gives each quantity with ours, theirs, the tolerance, the gap and the interval.
2. I stop and tell you. No hyperparameter, seed, landscape value or metric is changed.
3. With your approval, the diagnostic arm runs their code under the same configuration on the same seeds. That tells us which case we are in:
   - **Their code matches the paper and ours does not:** we have a bug. It is found by comparing behaviour and code, never by searching for settings that close the gap. Every fix must point to a documented divergence from the paper or their code. Then a new dated preregistration and a full rerun.
   - **Their code does not match the paper either:** the published numbers do not reproduce from the released code. That is reported as a finding, with the size of the gap.
4. Rung 2 does not start on an unmatched reproduction. Whether "Gate 0 passed, Gate 1 failed" is enough shared ground for rung 3, with that wording, is your decision.

### Path to rungs 2 and 3 (sub-question 5)
**Rung 2:** the same environment, seeds and evaluation episodes.
- Add discrete SAC (target c·log 15) and OC-SAC (target c·log M). Keep the reproduced DQN, the MDP, random cycling and the best two-drug cycle as comparators.
- Pair seeds and use equal training budgets (500 episodes) with equal tuning budgets.
- Preregister an equivalence margin. A tie is the expected result.
- **Prerequisite, a scope decision for you:** a mapping from the 15 β-lactams to M mechanism classes. Weaver's environment has none. It should come from a cited source, not be chosen by us.

**Rung 3:**
- Add only a d-delayed, exponentially weighted occupancy over the mechanism classes that modulates the landscape. The exact functional form is a design decision for an arena or a preregistration.
- The observation carries the pending occupancy, and this is asserted.
- **Bridge test:** with coupling strength 0, the rung-3 environment must reproduce the rung-1 trajectories exactly on the same seeds. That one test is what makes the ground shared.
- The published DQN gets the augmented observation in its published configuration, so it is the strongest honest version of it.
- The planner on the augmented state (continuous occupancy) needs discretisation or the mean-field planner. Its quality has to be checked before it is used as a normaliser.

## 2. Against each criterion
1. **Shared ground.** Every gated quantity is one the authors state in text or that is computed directly from their stated setup: 100/100, 98/100, 1.41, 1.88, CTX above 50%, and 0.96/0.82. Nothing is digitised from figures. Digitising Fig. 2B is rejected as a gate because it adds a measurement error of our own to a quantity the authors never gave as a number.
2. **A tolerance fixed now that can fail.**
   - Every threshold is either numeric or a formula whose inputs are known before any training (model-derived standard deviations, exact Fisher thresholds).
   - The replicate count is 100, which is exactly the dev tier (seeds.yaml dev: [0, 99]).
   - Each check can fail: k1 ≤ 94, k2 ≤ 90 (approximately), a mean outside 1.31–1.51, an exact random value off 1.88, or the MDP not favouring CTX.
3. **Auditability and licences.**
   - The main route is entirely our code and data under a pinned commit and a preregistration. A clean-room reimplementation does not copy GPL expression (this is my reading, not legal advice). Numbers and the algorithm description are facts.
   - The rl_cycling repository has no licence, so it is read only.
   - The diagnostic arm keeps GPL code outside the repository and fetches it by hash, so it can be rerun.
4. **Cost.** About 4–6 working days for the main route:
   - landscape transcription and diff, about 1 day;
   - environment, MDP and baselines, about 1 day;
   - DQN, about 1 day;
   - preregistration and the code look, about 1 day;
   - runs: minutes to hours of CPU, because the network is tiny with 10k steps per replicate.
   Stall risks are the two layers of hyperparameter defaults and the Table 4 image. The first is bounded by pinning the effective values at 16335ce. The second is bounded by double entry. If triggered, the diagnostic arm adds about 2–4 days with a high stall risk (fragile TensorFlow 2.11), which is why it is conditional.
5. **Use for rungs 2 and 3.** The environment is ours and Gymnasium-shaped, so SAC, OC-SAC and the occupancy wrapper attach directly. The zero-coupling bridge test ties rung 3 to rung 1.
6. **Security.** The main route executes no downloaded code. The diagnostic arm runs only after review and your approval, in a container with no network access.

## 3. Strongest objection to my own proposal
**The objection:** the reported quantities are weak tests of fidelity. Counts near 100/100 are close to their ceiling, so almost any working DQN beats random. Only L3 is quantitative, it rests on 10 replicates with no dispersion reported, and it sits in a noise regime that depends on implementation details.

**Why I don't think it is fatal:**
- Rung 1 licenses only "our reimplementation reproduces the published result", and that is what this gate tests.
- What rungs 2 and 3 actually depend on is environment fidelity. Gate 0 tests that tightly (E3 to Monte Carlo precision, plus E1 and E2).
- L1 also requires beating the best of all two-drug cycles, which is close to the MDP and much harder than beating random.

**The reporting consequence:** the report should say plainly that the learner part of the gate has low resolving power. It should not present a pass as strong evidence that our DQN behaves like theirs.

**A smaller objection:** the commit that produced the figures is unknown. If 16335ce differs from that commit in code these numbers depend on, a true reproduction could fail. The diagnostic arm is what would show that.

## 4. What would show me wrong
- **E3 fails even though our Table 4 transcription matches the hard-coded values.** Then the model, the start state or the metric differs from what I have assumed, and the gate design needs revisiting before any learner runs.
- **The read-only look shows the harness configuration cannot be pinned**, for example sweep outcomes overriding the defaults. Then the route stalls and this cost estimate is wrong.
- **The between-replicate standard deviation for RL-fit at noise 40 is above about 0.15.** Then ±0.10 has a high false-discrepancy rate and L3 is mostly noise.
- **The diagnostic arm, once run, fails its own gates by a wide margin.** Then the target itself is not reproducible, and the bar for rung 1 needs a new decision from you.
- **The rest of the application literature search (queries not yet run) turns up an RL drug-cycling paper** that reports a numeric table with dispersion and released seeds. That would be a better target.

## 5. Prior work
- **Mira et al. 2015**, the source of the landscape, reports exact equal-probability path probabilities (for example, the optimised range 0.375–1.0; it also says 0.38 elsewhere, an inconsistency in the paper itself). These could serve as an extra transcription check, but I would not gate on them. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0122283
- **Nichol et al. 2015, PLoS Comput Biol**, ran MDP sequencing on the same landscapes. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567305/
- **Weaver et al. 2024**, the published version with the stated numbers verified verbatim: https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/ ; preprint https://pmc.ncbi.nlm.nih.gov/articles/PMC9882109/
- **Reproducing Weaver et al.:** none found. Two searches, including this session's, found no follow-up or replication.
- **Methodology:**
  - Henderson et al., "Deep RL that Matters" (arXiv 1709.06560), on seed variance and why only distribution-level agreement is possible without seeds.
  - Agarwal et al., rliable (arXiv 2108.13264), for intervals and IQM.
  - Simonsohn 2015, "Small Telescopes" (Psychological Science), and Lakens 2017 on equivalence testing, as precedents for fixing a replication criterion in advance and for equivalence-margin reasoning.
  - I cite these last four from memory; I did not open them this session.
- **Not RL, no code:** arXiv 2510.01808. https://arxiv.org/abs/2510.01808

## Notes for the arena
- **Contradictions in the criteria:** none. Criteria 1 and 2 pull against each other, because the only shared quantities are near-ceiling counts and one mean, so tolerances that can fail have limited resolving power. I state that above rather than resolve it.
- **Empirical parts:** what their code actually ran is a factual question, answered by the read-only look. The choice of target itself is not empirical.

Files read:
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/arena/2026-11-23-reproduction-criteria.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-16-weaver-2024-drug-cycling-dqn.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-23-application-queries.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-23-reviewer-leads.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/inbox/2026-11-16-prior-art-antibiotic-rl.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/seeds.yaml
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/budget.yaml
- /private/tmp/claude-501/-Users-diyap-Occupancy-Coupled-Markov-Decision-Processes/dd0e1ca7-75ac-4fdd-9fa8-5a9faedfa4d9/scratchpad/plan-doc.txt

Sources:
- [Weaver et al. 2024, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/)
- [Weaver et al. preprint, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9882109/)
- [Mira et al. 2015, PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0122283)
- [Nichol et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567305/)
- [arXiv 2510.01808](https://arxiv.org/abs/2510.01808)
- [Maltas and Wood 2019, PLoS Biol](https://pmc.ncbi.nlm.nih.gov/articles/PMC6834293/)
