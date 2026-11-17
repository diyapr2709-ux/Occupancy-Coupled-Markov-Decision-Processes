# Reproduction target: RL drug cycling on measured E. coli fitness landscapes (PNAS 2024)

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. Nothing here is a result of this project.

How this was gathered: pages were opened through a fetch tool that returns a
model-written extraction of the page, not the raw page. Short quotes below are as
returned by that tool. Commit hashes come from the GitHub REST API. No code was
cloned, installed or run. Items marked "not opened" could not be read and must not be
assumed.

## Paper

- "Reinforcement learning informs optimal treatment strategies to limit antibiotic
  resistance". Proc Natl Acad Sci U S A, 2024 Apr 12; 121(16): e2303165121.
  DOI 10.1073/pnas.2303165121. Article licence: CC BY-NC-ND 4.0.
- Published version (opened): https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/
  (first request returned a verification page; the retry with a trailing slash opened).
- Preprint v2 (opened): https://pmc.ncbi.nlm.nih.gov/articles/PMC9882109/
- Publisher page https://www.pnas.org/doi/10.1073/pnas.2303165121 : HTTP 403, not opened.
- bioRxiv full text: HTTP 429, not opened.
- SI Appendix (cited for Figs S1, S2 and possibly tuning details): NOT OPENED. Any
  hyperparameter table that may exist there is "not found", not "absent".

## 1. Reference implementation

Simulation and RL package ("evodm"):
- URL: https://github.com/DavisWeaver/evo_dm
- Licence: GPL-3.0 (LICENSE.md). Copyleft: review before any reuse of code.
- No releases, no tags. setup.py version 1.1.2. 234 commits on main.
- HEAD of main when accessed: d975114b8cd01dad8036a21dba2b086cee149157 (2024-08-23,
  "update_target_counter back").
- Last commit before the PNAS publication date: 16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0
  (2023-12-09, "fix prev"). A query for commits up to 2024-04-30 returned this as newest.
  The paper does not state which commit produced its figures: NOT FOUND. The preprint
  experiments (January 2023) predate this commit; the delay feature was added 2023-07-28
  (46ec35711b2fe51a8b90eddcee67c6c2b695ed93).
- Post-publication commits change code a reproduction touches:
  73358131244cc86dd0f7f3e36cd93f8819a673e2 (2024-08-14, "fix policy and value
  iteration") rewrites value_iteration and policy_iteration in dpsolve.py as wrappers
  around pymdptoolbox and edits evol_game.py (+62/-31). backwards_induction, the solver
  the paper names, is not touched by that patch.
- Dependencies (setup.py at HEAD): tensorflow~=2.11.0, numpy, scipy, networkx, pytest,
  tqdm, pymdptoolbox, pandas, matplotlib, ruptures. No python_requires. The commit log
  around 2023-07-17/19 records repeated TensorFlow/scipy/networkx version changes for a
  cluster, so the environment is fragile.
- README contains a title, authors and badges only. No install, usage or reproduction
  instructions.
- Repository also tracks build/, egg-info, __pycache__, .coverage and a profiler dump.

Analysis / figure code:
- URL: https://github.com/DavisWeaver/rl_cycling
- HEAD of master: d68cf0773ece598082268faddc3776f9f3a7ed71 (2023-08-20, "fix readme").
  Repository created 2023-08-20, i.e. it predates the published version by about eight
  months and was not updated for it.
- Licence: none listed (GitHub API licence field is null). No licence means no reuse
  rights by default.
- Language R, driving the Python package through load.py. Contents: R/ harness scripts
  (test_harness.R, test_harness_fit.R, test_harness_noise.R, learning_rate.R,
  batch_size.R, gamma.R, update_target_every.R, prep_figs.R, figs.R), batch/ cluster
  scripts, a manuscript PDF dated 2023-02-23, and a wet-lab validation/ folder.
- README states the purpose only (companion to the bioRxiv v1 preprint). No commands,
  dependencies, compute notes or data locations.

## 2. Data release

- Fitness landscapes: released, but only as numbers hardcoded in source.
  evodm/evol_game.py, function define_mira_landscapes: 15 drugs x 16 genotypes (N = 4),
  drug order AMP, AM, CEC, CTX, ZOX, CXM, CRO, AMC, CAZ, CTT, SAM, CPR, CPD, TZP, FEP.
  No csv/npy data file exists in either repository.
- Upstream source: PLoS ONE 10(5): e0122283 (2015), "Rational Design of Antibiotic
  Treatment Plans", https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0122283
  (opened). CC BY. Data statement as extracted: "All relevant data are within the paper
  and its Supporting Information files." The growth rates are Table 4 ("Average Growth
  Rates (x 10-3)"), 12 replicates per genotype-drug pair. Table 4 is an image on the
  page; its values were NOT read and the hardcoded values were NOT checked against it.
- DISCREPANCY inside the released code (present at both 16335ce and d975114): the dict
  form and the list form of define_mira_landscapes disagree for ZOX, second genotype:
  dict 0.5, list 0.805. Default is as_dict=False (list form). Which value matches
  Table 4, and which form the paper's runs used, is NOT FOUND.
- Per-replicate training outputs behind the figures (the data/results/*.Rda and
  data/test/*.Rda files the R scripts write and read): NOT RELEASED. The rl_cycling
  tree has no data/ directory. The paper's statement is "All other data are included
  in the manuscript and/or SI Appendix."
- Trained agents / weights: not found.

## 3. Numbers reported by the original authors

All of the following are "reported by" the authors of the PNAS 2024 paper. None is a
result of this project. The paper has NO numeric results table: Table 1 lists drug
codes, Table 2 lists example drug sequences. The headline comparison is a density plot.

| Quantity | Reported by the authors | Where |
|---|---|---|
| RL-genotype vs random cycling and all two-drug cycles | better in all 100 replicates | text Sec. 2.1; Fig. 2B (published), Fig. 2C (preprint) |
| RL-fit vs random cycling | "measurable decrease in population fitness" in 98/100 replicates | text Sec. 2.1; SI Fig. S1A |
| RL-fit vs two-drug cycles | average replicate outperforms all two-drug cycles | Fig. 2B |
| RL-genotype vs MDP optimum | performance distribution "nearly overlapped" | Fig. 2B; SI Fig. S1D |
| RL-fit under noise modifier 40 | average population fitness 1.41 vs 1.88 for random | text Sec. 2.1; Fig. 2C |
| Policy similarity to MDP | Pearson 0.96 (RL-genotype), 0.82 (RL-fit) on edge weights | Sec. 2.3; SI Fig. S2 |
| Learning speed | effective after about 100 episodes, little change to 500 | Fig. 2A |
| MDP policy | cefotaxime in more than 50% of time steps | Sec. 2.2; Fig. 3A |
| Measurement delay (RL-genotype) | d <= 1 matches base case; d > 1 no better than, or worse than, random | Sec. 2.5; Fig. 6 (text says 6C, caption says 6B) |
| Landscape size N = 4..10 | RL-genotype > RL-fit > random at every size | Fig. 6A |

Not reported as numbers anywhere in the opened text: mean population fitness of
RL-genotype, RL-fit, random, the MDP optimum or the best two-drug cycle in the
noise-free base case; any interval or dispersion statistic for them.

Internal inconsistency in the published version: the text says 100 replicates per
condition, while the noise-panel caption says "10 fully trained RL-fit replicates"
(the R code wraps noise replicates at 10).

## 4. Algorithm, protocol and configuration

Stated in the paper:
- Deep Q-learning with a target network "only updated rarely"; MSE loss.
- RL-genotype: one-hot genotype (state vector) input, two 1D conv layers, max pooling,
  dense 28, linear output. RL-fit: input is current population fitness plus the one-hot
  previous action; dense 64, dense 28, linear output.
- Environment: 15 drugs, N = 4 loci (16 genotypes); strong selection weak mutation;
  equal probability to every fitter single-mutant neighbour; one drug per step; reward
  R = 1 - f(s|a).
- 20 steps per episode, 500 episodes per replicate (10,000 observations), 100
  replicates per RL condition. "We chose 500 episodes as the training time after
  hyperparameter tuning"; no tuning grid or SI pointer was found in the opened text.
- Baselines: random cycling; all two-drug cycles; finite-horizon MDP solved by
  backward induction with perfect information.
- Noise: f' = f + N(0, 0.05 x sigma^2), sigma from 0 to 100. Delay d from 0 to 5.

NOT stated in the paper text (opened versions): learning rate, discount, epsilon
schedule, batch size, replay size, target-update period, optimiser, seeds.

Recovered from the released code (values to be confirmed by reading the pinned source
directly, since they came through an extraction tool):
- hyperparameters class defaults (learner.py, same at 16335ce and d975114):
  LEARNING_RATE 0.0001, DISCOUNT 0.99, REPLAY_MEMORY_SIZE 10000,
  MIN_REPLAY_MEMORY_SIZE 1000, MINIBATCH_SIZE 100, UPDATE_TARGET_EVERY 310 (comment
  says 500), EPSILON_DECAY 0.95, MIN_EPSILON 0.001, RESET_EVERY 20, EPISODES 500.
- evol_deepmind (exp.py) overrides them with its own signature defaults:
  learning_rate 0.0001, minibatch_size 60, update_target_every 310, gamma 0.99,
  min_epsilon 0.005, EPSILON_DECAY = MIN_EPSILON ** (1/EPISODES), train_freq 100,
  win_threshold 200, win_reward 0.
- The R harness (rl_cycling R/test_harness.R, R/test_harness_fit.R) calls it with
  episodes 500, reset_every 20, iter 100, mira TRUE, num_drugs 15, N 4,
  min_epsilon 0.005, noise TRUE with noise_modifier 1, sigma 0.5, win_threshold 50,
  win_reward 0, starting_genotype 0, train_freq 100. It does not pass learning rate,
  batch size, target period or gamma, so the evol_deepmind defaults apply at whatever
  commit was installed. Separate sweep scripts exist for learning rate, batch size,
  gamma and target period; their outcome is not released.
- Network as coded: Conv1D(64, k=3, relu) x 2, MaxPooling1D(2), Flatten, Dropout(0.2),
  Dense(28, relu), linear output; Adam; MSE. Dropout is not mentioned in the paper.
- Plain DQN target (max over target network), not double DQN.
- train_freq is environment steps between single gradient steps: with train_freq 100
  and 10,000 total steps, a replicate performs on the order of 90 gradient updates
  after the 1,000-transition warm-up. Worth verifying; it determines how sensitive the
  result is to initialisation.
- Evaluation: there is no separate greedy evaluation phase. practice() returns an empty
  reward list; performance is reconstructed from the agent's master_memory by the R
  code. prep_figs.R keeps, for the learner, rows with ep_number > 500, which matches
  the harness re-running the trained agent with pre_trained=TRUE (epsilon 0) for a
  further 500 episodes. The metric is mean population fitness per episode.
- Random baseline: uniform redraw every step. Two-drug sweep: policy_sweep with 100
  episodes per pair, starting genotype 0. MDP: backwards_induction with horizon
  RESET_EVERY and discount 0.99.
- The base-case harness passes noise TRUE with modifier 1, so the "noise-free" base
  case as run may include sensor noise of scale 0.05. Needs confirmation.
- Seeds: none set anywhere in learner.py, exp.py or the R harness. Bit-level
  reproduction is impossible; only distribution-level agreement can be targeted.
- Extraction-tool observations that should be checked by reading the source: the done
  flag is set as self.DONE but read as self.done; convert_two_drug builds 0-based drug
  indices where other code uses 1-15.

## 5. What a reimplementation needs that is not stated or released

1. The commit that produced the published figures.
2. Confirmation of the DQN hyperparameters actually used (paper silent; code has two
   layers of defaults and tuning sweeps with unreleased outcomes).
3. The correct ZOX value and which landscape form was used; verification of all 240
   hardcoded values against Table 4 of the 2015 source (an image).
4. Whether drugs were normalised (NORMALIZE_DRUGS is True by default in the class but
   mdp_mira_sweep and policy_sweep use False; the reported 1.41 and 1.88 exceed 1, so
   un-normalised growth rates are implied for that panel).
5. Numeric targets. The base-case comparison is published only as figures, so a
   tolerance fixed beforehand has to be defined against the few stated numbers
   (98/100, 100/100, 1.41 vs 1.88 at noise 40, correlations 0.96 and 0.82) or against
   values digitised from Fig. 2. The unreleased .Rda outputs would settle this.
6. The SI Appendix, which was not opened.
7. A working TensorFlow 2.11 environment, if the released code is to be run rather
   than reimplemented (installing external code is a stop-and-ask item here).

## 6. Verdict on feasibility

- Running the authors' pipeline end to end: code for the simulator, agent, baselines
  and harness is public, and the landscape values are embedded. Per-replicate outputs,
  seeds, the figure-producing commit and stated hyperparameters are not. Expect days
  for environment setup and a 100-replicate rerun, plus an unknown amount to reconcile
  defaults with the paper.
- Independent reimplementation of the environment and baselines (SSWM Markov chain on
  16 genotypes, random, 105 two-drug cycles, 20-step backward induction): small and
  fully specified once the landscape values are verified. Days.
- Matching the RL numbers: the qualitative claims (all replicates beat random and
  two-drug cycles; near the MDP optimum) are checkable. A quantitative match "within a
  tolerance fixed beforehand" is only possible against a handful of stated numbers,
  none of which is the noise-free base-case mean. This is the part that can turn days
  into weeks, and it should be decided before any run what counts as a match.

## 7. Related work with or without released code (opened pages only)

- PLoS Biol 17(10): e3000515 (2019), "Pervasive and diverse collateral sensitivity
  profiles inform optimal strategies to limit antibiotic resistance".
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6834293/ (first half of page read). MDP over
  discretised resistance profiles, 6 drugs, value iteration; reported by its authors:
  the gamma = 0.99 policy keeps resistance below all tested 2-4 drug collateral
  sensitivity cycles (Fig 5F). Data in "S1 Data". MDP code release: not found in the
  portion read; data availability statement not reached.
- JMLR 21(203), 2020, "Dynamic Control of Stochastic Evolution: A Deep Reinforcement
  Learning Approach to Adaptively Targeting Emergent Drug Resistance".
  https://jmlr.org/papers/v21/19-546.html (opened). Reported by its author: "100%
  success rate" at suppressing proliferation in a sample scenario. Code link on the
  page: none.
- Mol Biol Evol 34(4): 802-817 (2017), "Antibiotic Cycling and Antibiotic Mixing: Which
  One Best Mitigates Antibiotic Resistance?" https://pmc.ncbi.nlm.nih.gov/articles/PMC5400377/
  (opened, last 8% not read). Reported by its authors: "neither cycling or mixing is a
  priori better". Mixing is random allocation and is treated as maximal drug
  heterogeneity; an information-entropy index of antibiotic use is attributed there to
  a 2006 monitoring method. No code or data repository named.
- Cancer Research 84(11), 2024, "Mathematical Model-Driven Deep Learning Enables
  Personalized Adaptive Therapy": found by search only, article page NOT opened; code
  availability not established.
- A direct follow-up to the PNAS paper with released code: NOT FOUND in two searches.
  The evo_dm repository itself gained seascape and horizontal-gene-transfer modules
  after publication (seascapes_exp.py, hgt_exp.py), which suggests follow-up work, but
  no paper was located.
