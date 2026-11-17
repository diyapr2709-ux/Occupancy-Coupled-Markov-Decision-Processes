# Comparators a hostile reviewer would ask for that are not on the current list

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. Adding any of these is a scope decision that is not a
gatherer's to make. Lower-priority pass: most sources below were read at abstract
level only, as stated per row. Rows marked "no source opened" are suggestions without
a citation and must be sourced before use. No code was cloned or run.

| # | Comparator | Why a reviewer asks | Source opened | Code |
|---|---|---|---|---|
| 1 | Mechanism-level SAC, uniform within class (already required) | If it ties the proposed agent, the contribution is a parametrisation. | See the mechanism-level-SAC inbox file. | Not found; build from discrete SAC. |
| 2 | Two-level SAC with learned within-class choice and temperature only on the class level | Same objective and same policy class as a drug-level agent with pushforward entropy; the cleanest test of objective versus parametrisation when within-class drugs differ. | Hybrid SAC, https://ar5iv.labs.arxiv.org/html/1912.11077 (two-level entropy, separate temperatures). | Not released. |
| 3 | Drug-level SAC with class-size-corrected entropy (action entropy minus expected log class size) | Removes the trivial explanation that drug entropy is inflated by class size. | Derivation only (mechanism-level-SAC inbox file). | n/a |
| 4 | MinRed SAC (redundancy-corrected maximum entropy) | Published method for the stated problem that action entropy over-counts equivalent actions; a reviewer who knows it will ask why it was not compared. | https://ar5iv.labs.arxiv.org/html/2102.11329 (UAI 2021). | Not found. |
| 5 | DQN, epsilon-greedy | It is the algorithm of the reproduction target; the control and bridge rungs need it on shared ground. | PNAS 2024 paper and its repository (reproduction-target inbox file). CleanRL cleanrl/dqn.py last touched at commit 004f8a086a892a2a180f4dd332b90d83a968aa7a (commit list opened, file not opened). | Released (both). |
| 6 | SAC with a fixed, tuned temperature | Separates "which entropy space" from "automatic temperature dynamics", which are confounded when the target entropy drives the result. | SD-SAC configuration uses fixed alpha 0.05: https://arxiv.org/html/2209.10081v4 | Released (SD-SAC repository, not inspected). |
| 7 | SAC plus an explicit reward term for mechanism diversity (no change to the entropy term) | Tests whether the effect is available by reward shaping alone. | No source opened. | n/a |
| 8 | A general-utility / convex-MDP method that optimises a concave function of the occupancy directly | The coupling makes the objective a function of the action occupancy, which is the convex-MDP setting; a reviewer will ask why an entropy bonus rather than the established formulation. | "Reward is enough for convex MDPs", https://arxiv.org/abs/2106.00661 (abstract only): goals that are convex in the stationary state-action distribution "cannot be formulated using stationary reward functions"; solved as a policy-versus-cost game. | Not found. |
| 9 | Delay-aware baseline (state augmentation with pending actions, or a non-stationary delayed-Q method) | The kernel responds after a delay; a reviewer will ask whether gains come from handling delay rather than from the entropy space. | "Acting in Delayed Environments with Non-Stationary Markov Policies", ICLR 2021, https://arxiv.org/abs/2101.11992 (abstract only). Note the setting there is execution delay, not delayed kernel response. | Links on the abstract page: github.com/galdl/rl_delay_basic and github.com/galdl/rl_delay_atari (repositories not opened). |
| 10 | Recurrent or frame-stacked SAC/PPO without the pending-occupancy feature | Shows whether the hand-built observation is doing the work. | No source opened. | n/a |
| 11 | Recovering-bandit planner (reward of an arm depends on time since last played) | Drug efficacy that recovers when a drug is rested is that model; it is the bandit literature's direct answer to cycling. | "Recovering Bandits", NeurIPS 2019 spotlight, https://arxiv.org/abs/1910.14354 (abstract only): expected reward depends on "the time since the arm was last played"; Gaussian-process estimation with optimistic planning. | Not found. |
| 12 | EXP3.S (adversarial, switching) | The non-stationarity is caused by the learner, so stochastic-bandit assumptions fail; the original SW-UCB paper itself benchmarks against it. | https://ar5iv.labs.arxiv.org/html/0805.3415 (simulation section). | SMPyBandits (EXP3 family present per its commit log; file not opened). |
| 13 | Restless-bandit index policy (one arm per drug or class, resistance as arm state) | Standard operations-research formulation of "resting" resources. | No source opened. | n/a |
| 14 | Stewardship heuristics from the clinical literature: periodic cycling, random mixing, and a heterogeneity-maximising allocation | Mixing is uniform randomisation over drugs and is treated in that literature as maximal heterogeneity, so "maximum entropy over drugs" and "over classes" both have clinical-heuristic counterparts that need no learning. | "Antibiotic Cycling and Antibiotic Mixing", Mol Biol Evol 2017, https://pmc.ncbi.nlm.nih.gov/articles/PMC5400377/ : reported by its authors, "neither cycling or mixing is a priori better". | None named. |
| 15 | Exact or near-exact planner on the true model (finite-horizon dynamic programming, with replanning under noise) | Every prior drug-sequencing paper reports the MDP optimum as the reference; needed to express results as a fraction of the attainable gain. | PNAS 2024 paper (backward induction); PLoS Biol 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6834293/ (value iteration, gamma = 0.99, Fig 5F). | evo_dm dpsolve.py (GPL-3.0); PLoS Biol code not found in the portion read. |
| 16 | Myopic greedy on the current resistance estimate | The cheapest state-feedback policy; if it matches the learned agents under noise, learning adds nothing. | No source opened. | n/a |
| 17 | Annealed-target SAC | A known stronger schedule for discrete SAC; relevant because a rising or falling target can change which agent wins. | https://arxiv.org/abs/2112.02852 (abstract only). | Not found. |

Notes
- Rows 14 to 16 may already exist in this repository's heuristics and oracle modules;
  they are listed because a reviewer will expect them named with their literature
  counterparts, not because they are known to be absent.
- Row 4 doubles as a prior-work question, not only a comparator; see section 3 of the
  mechanism-level-SAC inbox file.
- Not searched in this pass: model-based RL on the known mean-field model, offline RL
  on logged prescribing, multi-objective (Pareto) formulations of cost versus
  resistance, and information-theoretic diversity indices used in antimicrobial
  consumption monitoring.
