# Rung 1 reproduction: specification draft (DOI 10.1073/pnas.2303165121)

Status: draft from a read-only look, 2026-11-25. Nothing here is a result of this project. Every number below is either "reported by the authors" or "code default" (a literal in the authors' code), or a count derived by arithmetic from code defaults and labelled as such. Nothing was executed, imported or installed from the clones.

## Sources and precedence

| Tag | Source | How it was read |
| --- | --- | --- |
| P | Published text, PNAS 2024 | Not opened in this look. Only the short quotes in `references/2026-11-16-weaver-2024-drug-cycling-dqn.md` are available, and they came through an extraction tool. |
| SI | SI Appendix of the published version | Not opened. Nothing in this file comes from it. |
| E | `evo_dm` at 16335ceaf834e6a84dd0fa8b3dff3ea4df963bc0 (GPL-3.0), cloned read-only to a directory outside the repository | Read with Read/Grep, plus `git log`/`git show` on its own history. `build/lib/evodm/` is byte-identical to `evodm/` for the five modules used here. |
| R | `rl_cycling` at d68cf0773ece598082268faddc3776f9f3a7ed71 (no licence), cloned read-only to a directory outside the repository | Read with Read/Grep, plus `git log`/`git show`. |
| S | `rl_cycling/manuscript/PNAS/PNAS_submission_02_23_23.pdf`, a February 2023 submission manuscript with its supplement. It is not the published text. | Its text was taken from the PDF's compressed streams by a parser written in the session scratchpad. The PDF is treated as data. It ranks with R, below E. Line numbers are the manuscript's own. |

Citations in the form `file:line` refer to E (paths under `evodm/`) or to R (paths under `R/`, `batch/`, `load.py`). E history is cited by commit hash.

Two facts govern how far E and R can be trusted as "the code that made the figures":

- **H1. The batch scripts install an unpinned evo_dm.** Every cluster script installs evo_dm from the repository's default branch at job time (`batch/test_harness_fit.sh:21`, and the same line in every `batch/*` file). The evo_dm commit that produced any figure is not recorded anywhere in R.
- **H2. The committed harnesses do not run to completion against E.**
  - `R/test_harness.R:60` passes a second named argument to the output-cleaning function, but `R/clean_evol_out.R:237` takes one argument. Under R semantics that is an "unused argument" error.
  - `R/test_harness_fit.R:48` and `R/test_harness_noise.R:51` call the per-episode policy sweep. At E, `evol_deepmind` does not record implied policies by default (`exp.py:26`, `learner.py:525-531`, `learner.py:547-550`), so the sweep returns an empty list. The loop at `R/clean_evol_out.R:56` then runs over 1:0 and indexes element 1 of an empty list.
  - `R/test_harness_fit.R` and `R/test_harness_noise.R` date from R commit 38725ad (2022-10-28) and were not changed afterwards.
  - The figure code reads policies and implied-policy memories from every base-case result file (`R/prep_figs.R:112-127`, called with its default `policy_exists = TRUE` from `R/prep_figs.R:214`). Those exist only when policies were recorded on every run. At E that is opt-in, behind the flag added in E commit 476922b (2023-05-18). Before 476922b it was always on.
  - The figure data were therefore produced by an evo_dm version other than E. That is an inference from code structure, not a settled fact.

Where E and the figure-era code differ, the reading below is E's (by precedence), and the difference is listed under "Conflicting reading".

---

## 1. Environment

### 1a. SSWM transition rule

- **Reading.**
  - A genotype is an integer 0..15, read as a 4-bit pattern. Its neighbours are the four single-bit flips.
  - Under drug a, a neighbour counts as fitter when its fitness exceeds the current genotype's by more than the code default 1e-5.
  - Each fitter neighbour gets equal probability. If there is none, the population stays where it is with probability 1.
  - Each step samples one genotype from that row, so the trajectory is a single population. Averaged outcomes are off.
- **Source.**
  - The environment builds every drug's matrix with the "phenom" variant at exponent 0 (`evol_game.py:152`; `phenom` default 0 at `exp.py:28`, passed through at `learner.py:159`).
  - The variant raises fitness gains to the power 0 and normalises them, which gives the uniform split (`landscapes.py:162-178`). Staying put is at `landscapes.py:170-171`.
  - Sampling: `evol_game.py:439-450` and `evol_game.py:471-482`. The harness leaves averaged outcomes at their default, False (`exp.py:17`).
  - The matrix is stored column-to-row: entry [next, current]. The solver transposes it to rows indexed by the current state (`dpsolve.py:115-130`).
- **Conflicting reading.**
  - S lines 57-60 say only "probability of mutation between adjacent genotypes" under SSWM. The published text, as quoted in the gatherer note, says "equal probability to every fitter single-mutant neighbour". The two agree.
  - The plain `get_TM` (`landscapes.py:84-121`) gives the same rule, but the environment does not use it.
- **Note for the implementation.** Normalisation is a positive affine map per drug, so it cannot change which neighbours are fitter, except through the absolute 1e-5 threshold. Assert in a test that the 15 transition matrices are identical on the raw and the normalised scale. Do not assume it.
- **Status:** settled.

### 1b. Starting genotype

- **Reading.**
  - Every episode starts as a point mass on genotype 0 (code default: `starting_genotype = 0` at `exp.py:25`; harness `starting_position = 0` at `R/test_harness.R:145`, `R/test_harness_fit.R:114`, `R/test_harness_noise.R:131`).
  - **Before the first decision, the loop takes one forced step with drug 1 (AMP).** The reset sets the action to 1 (`evol_game.py:376`). The training/evaluation loop runs RESET_EVERY + 1 iterations, and iteration 0 calls the environment step and skips everything else (`learner.py:475-478`).
  - That step is not stored: replay and the logging memory record only steps with action number above 1 (`learner.py:205`). The first agent, random, MDP or two-drug decision is therefore taken at the genotype reached by one SSWM step under AMP from genotype 0.
  - This applies to every condition that goes through `practice` (RL training, RL evaluation, random, MDP, two-drug sweep).
- **Source.** E as cited. The forced step entered E in commit 9306148 (2022-11-10) and is present at E.
- **Conflicting reading.**
  - P (gatherer note) and S line 116 state 20-step episodes "starting at genotype 0", with no forced first drug.
  - The design outcome's target line ("start at genotype 0, 20-step episodes") does not mention it either.
- **Status:** settled (E). The forced AMP step is part of the specified environment.

### 1c. Episode length and the done flag

- **Reading.**
  - One episode is 21 environment steps: the forced AMP step plus 20 decision steps. 20 transitions are recorded.
  - RESET_EVERY is the code default 20 (`learner.py:84`; harness `reset_every = 20`).
  - The terminal flag never fires:
    - the reward function writes an upper-case DONE attribute (`evol_game.py:300`, `:303`, `:310`, `:313`);
    - the loop tests a lower-case done attribute (`learner.py:532`);
    - the lower-case attribute is only ever set False (`evol_game.py:92`, `:381`).
  - The conditions that would set DONE are unreachable under the harness arguments anyway:
    - the win threshold is 50 consecutive steps (harness), against 21 steps per episode;
    - the consecutive counters reset every episode (`evol_game.py:379-380`);
    - the cutoffs are the `evol_deepmind` defaults, −1 and 2 (`exp.py:15`);
    - the win reward is 0 (harness).
  - There is no terminal masking in the TD target either (`learner.py:249-254`). Every transition bootstraps, including the last step of an episode.
- **Source.** E and R as cited. The gatherer's suspicion is confirmed.
- **Conflicting reading.** None.
- **Status:** settled.

### 1d. Reward

- **Reading.**
  - The reward at a decision is 1 − f_a(g): the true, un-noised fitness under the chosen drug a of the genotype g occupied **before** that step's transition.
  - The fitness is recorded before the population evolves (`evol_game.py:476-479`). The reward uses the true fitness (`evol_game.py:181`, `:223`, `:315`).
  - The total-resistance branch is never taken, because the call passes no such flag (`evol_game.py:223`, `:230`).
  - The logged "population fitness" metric is the same quantity f_a(g), also un-noised: `learner.py:211`, `learner.py:213`, and `R/clean_evol_out.R:78-85`, which takes column 5 or 4 as `average_fitness`.
  - The scale is whatever the environment's drugs are: normalised on the harness path at E (item 1e).
- **Source.** E. P (gatherer note) and S eq. 1 (line 77): "R = 1 − f(s|a)". The MDP uses the same reward (`dpsolve.py:94-113`).
- **Conflicting reading.** `dp_env.define_P` (`dpsolve.py:62-92`) would reward the post-transition genotype, but nothing calls it (`dpsolve.py:55`).
- **Status:** settled.

### 1e. Normalisation of drugs on each code path

Normalisation maps each drug's 16 values to (f − min)/(max − min) over that drug (`evol_game.py:428-435`). It is applied in the environment when the flag is set (`evol_game.py:144-147`).

| Code path | Flag at E | Source |
| --- | --- | --- |
| RL training (`evol_deepmind`, harness) | True. The default is True and no harness passes it. | `exp.py:14`, `exp.py:98`; R harness calls at `R/test_harness.R:49-59`, `R/test_harness_fit.R:32-40`, `R/test_harness_noise.R:35-43` |
| RL evaluation (pre-trained rerun) | Same agent, same environment, so True | `exp.py:124-127` |
| Random baseline | A deep copy of the same agent, so True | `exp.py:134`, `exp.py:141-143` |
| MDP baseline | A deep copy, solved on that copy's drugs, so True | `exp.py:136-138`, `learner.py:385-389` |
| Two-drug cycles (`policy_sweep`) | False (its default), not overridden by the caller | `exp.py:388`, `exp.py:401`; caller `R/prep_figs.R:663` |
| MDP discount sweep (S Fig. S9) | False | `R/prep_policy_sweeps.R:28-36`, `:161`; `exp.py:261` |
| Signal-to-noise panel | False | `exp.py:453`; caller `R/prep_figs.R:757-765` |

- **Conflicting reading (E history, which also bears on H2).** In `evol_deepmind`, the measured-landscape branch:
  - forced normalisation **off** before E commit 9865458 (2023-02-23; see E 61ee56d `exp.py:123`);
  - forced it **on** from 9865458 to 7aa4fdc (2023-06-05);
  - follows the argument, default on, from 7aa4fdc to E.
  - The reported 1.41 and 1.88 (P, S line 144) exceed 1, so they cannot be on the normalised scale.
  - The density panel's x-axis is fixed at 0 to 3.3 (`R/figs.R:124`), which fits the raw scale for every condition.
- **Status:** settled for E. Which scale the published figures used is not found (see item 1f and Stop S3).

### 1f. Landscape scale of each comparison (E + R as committed)

| Comparison (figure) | Scale of side A | Scale of side B | Mixed? |
| --- | --- | --- | --- |
| RL-fit, RL-genotype vs random vs MDP (base-case density panel, `R/figs.R:114-122`; S1 panels, `R/figs.R:137-197`) | normalised | normalised | no |
| **RL-fit / RL-genotype / random / MDP vs two-drug cycles (same density panel, `R/figs.R:119-123`)** | **normalised** | **raw** | **yes: STOP** |
| RL-fit vs random under noise (`R/figs.R:91-96`) | normalised | normalised | no, but the reported values are > 1 (item 9) |
| MDP discount sweep (`R/figs.R:852-871`) | raw only | — | — |

- **Conflicting reading.** If the figures came from evo_dm before 9865458, every condition in the density panel was raw and the panel is not mixed. The pinned commits do not establish that.
- **Status:** settled for E + R; **STOP S1** for the two-drug comparison.

## 2. Landscapes

### 2a. Which form the harness uses

- **Reading.** The list form. On the measured-landscape flag, `evol_deepmind` calls the landscape-definition function with no argument (`exp.py:129-131`), and the dictionary option defaults off (`evol_game.py:497`). `policy_sweep` and the discount sweep use the list form too (`exp.py:403`, `exp.py:277`).
- **Where the dictionary form is used.** Figure code only:
  - the "opportunity landscape" helper (`exp.py:494-498`), used by `R/prep_figs.R:789-799`;
  - the landscape summary panel (`R/figs.R:374`).
- **Status:** settled.

### 2b. ZOX, second genotype (index 1)

- **Reading.** The list value runs: code default 0.805 (`evol_game.py:527`). The dictionary holds 0.5 (`evol_game.py:510`).
- **History.**
  - The list has held 0.805 since E commit 7f42785 (2021-11-01).
  - The dictionary was changed from 0.805 to 0.5 in E commit 2f5dfb4 (2022-12-12). That commit added the opportunity-landscape helper and touched no other landscape value.
- **Which value matches the 2015 source's Table 4:** not found. That table is an image and was not read in this look. It belongs to E1, the landscape audit.
- **Status:** settled for "which runs"; not found for "which is correct".

### 2c. Other dictionary/list disagreements

- **Reading.**
  - A text comparison of all 240 entries, `evol_game.py:506-520` against `:523-537`, finds exactly one disagreement: the ZOX one in item 2b.
  - Drug order is the same in both forms: AMP, AM, CEC, CTX, ZOX, CXM, CRO, AMC, CAZ, CTT, SAM, CPR, CPD, TZP, FEP.
  - The parse ran on the file as text, with a script in the session scratchpad. Nothing was imported.
- **Open for E1.** The code treats list position i as the genotype whose bit pattern is i, with neighbours by bit flip (`landscapes.py:163`). Whether the 2015 table orders its 16 genotypes the same way was not checked. If the table orders them by number of mutations, the mapping differs.
- **Status:** settled (one disagreement). The genotype-order correspondence to Table 4 is not found.

## 3. The DQN as the R harness invokes it

Defaults are layered in three places: the hyperparameter class (`learner.py:60-110`), then the `evol_deepmind` signature (`exp.py:10-28`) applied at `exp.py:86-122`, then the harness arguments. The effective values at E are:

| Hyperparameter | Effective value | Set at | Overridden layer |
| --- | --- | --- | --- |
| Training episodes | 500 | harness (`R/test_harness.R:143`, `R/test_harness_fit.R:112`, `R/test_harness_noise.R:129`) | signature 50, class 500 |
| Steps per episode | 20 (+1 forced, item 1b) | harness | — |
| Discount | 0.99 | signature `exp.py:17` → `exp.py:104` | class 0.99 |
| Learning rate (Adam) | 1e-4 | signature `exp.py:18` → `learner.py:200` | class 1e-4 |
| Minibatch | 60 | signature `exp.py:18` → `exp.py:107` | class 100 |
| Target update period | 310 env steps (counter rule below) | signature `exp.py:24` → `exp.py:108` | class 310 (its comment says 500) |
| Replay capacity | 10 000 | class `learner.py:62` | not overridden |
| Warm-up | 1 000 transitions | class `learner.py:64` | not overridden |
| Epsilon start | 1 | class `learner.py:72` | — |
| Epsilon floor | 0.005 | harness, equal to the signature value | class 0.001 |
| Epsilon decay per episode | 0.005^(1/500) | `exp.py:122` | class 0.95 |
| Train frequency | 100 decision steps | harness `R/test_harness.R:145`; signature default `exp.py:25` for the other two harnesses | — |
| Loss | MSE | `learner.py:200` | — |
| Observation noise | on, modifier 1 | harness | class off |
| Normalised drugs | yes | signature `exp.py:14` | — |
| Win threshold / reward / cutoffs | 50 / 0 / −1 and 2 | harness; signature `exp.py:15` | class 1000 / 0 / 0.001 and 0.999 |

- **Corroboration.** S Table S1 (supplement line 395), reported by the authors: discount 0.99, learning rate 0.0001, minibatch 60, target update 310. It states no train frequency, replay size or epsilon schedule.
- **Adam's other parameters** are the Keras defaults of the installed TensorFlow. E pins `tensorflow~=2.11.0` (`setup.py:10`). The TensorFlow source is not in the clones, so the actual values are not found here.
- **Conflicting reading.** Before E commit 476922b (2023-05-18) there was no train frequency: a gradient step ran on every decision step after warm-up (see E 61ee56d, `learner.py` practice loop). The two harnesses that do not pass a train frequency predate that commit (H2).

### 3b. Train frequency semantics and update count

- **Reading.**
  - A single counter, set once before the episode loop, increments after each decision step and never resets (`learner.py:467`, `:535`).
  - When it is a multiple of 100, `train()` is called (`learner.py:525-527`). `train()` returns without an update while replay holds fewer than 1 000 transitions (`learner.py:219-220`).
  - Each decision step appends exactly one transition (`learner.py:205-206`).
  - A real update draws 60 transitions uniformly without replacement (`learner.py:223`) and makes one `fit` call over them, with batch size 60 and the default single epoch, so one optimiser step (`learner.py:236-237`).
  - **Derived count (arithmetic on code defaults, not a measurement):** 10 000 decision steps, with calls at multiples of 100 and the first real one at step 1 000, give 91 gradient updates per replicate.
- **Conflicting reading.** Under the figure-era code (pre-476922b), every decision step from the warm-up on is an update.
- **Status:** settled for E. **Stop S4** decides which count the reproduction uses.

### 3c. Epsilon schedule, target update, networks

- **Epsilon.**
  - Episode 1 acts with epsilon 1. After each episode epsilon is multiplied by the decay and floored at 0.005 (`learner.py:558-562`).
  - The decay is chosen so the floor is reached at episode 500 (`exp.py:122`).
  - The random action is uniform over drugs 1..15 (`learner.py:513`).
  - Settled.
- **Target update.**
  - The counter is incremented on every environment step, including the forced step (`evol_game.py:163`). It is not reset between episodes.
  - It is checked only inside `train()`, after a real update. When it exceeds 310 the target network copies the online weights and the counter resets (`learner.py:239-242`).
  - The two networks start identical (`learner.py:165-166`). The target is a plain max over the target network, not double DQN (`learner.py:253-254`).
  - **Derived count (arithmetic on code defaults):** 21 environment steps per episode and updates every 105 environment steps give a sync on the 1st, 4th, 7th, … real update, which is 31 syncs per replicate.
  - Settled.
- **RL-genotype network.**
  - Input: a one-hot genotype vector of shape (16, 1) (`evol_game.py:117-118`).
  - Layers: Conv1D with 64 filters, kernel 3, ReLU; a second identical Conv1D; max-pool of size 2; flatten; dropout 0.2; dense 28 with ReLU; a linear output of 15 (`learner.py:178-183`, `:197-199`).
  - Settled.
- **RL-fit network.**
  - Input of length 16: the one-hot previous action (15) plus one fitness reading (`evol_game.py:119-120`).
  - Layers: dense 64 with ReLU; dropout 0.2; dense 28 with ReLU; a linear output of 15 (`learner.py:184-187`, `:197-199`). Dropout is active only inside `fit`.
  - Settled.
- **What RL-fit observes (settled, E).**
  - At decision t the agent sees the one-hot of a_{t−1} and the noisy fitness f_{a_{t−1}}(g_{t−1}): the fitness recorded during the previous step, of the genotype before that step's transition (`learner.py:362-365`, `evol_game.py:228-231`, `:261-268`).
  - In the stored transition, the current-state fitness is the **true** previous fitness (`evol_game.py:248`, `:254-259`). The next-state fitness is the **noisy** one (`evol_game.py:228`).
  - So training inputs and acting inputs differ in noise.
  - Conflicting reading: S lines 93-94 say "population fitness at time t and one-hot encoded action at time t−1".
- **RL-genotype harness invocation.**
  - At R every active measured-landscape call uses the fitness input (`R/test_harness.R:143-145`, `R/test_harness_fit.R:112-114`, `R/test_harness_noise.R:129-131`).
  - The only genotype-input call is commented out and uses 1 000 episodes (`R/test_harness.R:146-147`).
  - The file-name parser expects genotype-input result files (`R/prep_figs.R:11-13`, `:150-152`), so such runs existed.
  - Reading adopted: the fitness harness with the input switched to the state vector and everything else unchanged.
  - **Status: not found** as committed code.

### 3d. Replay buffer

- **Reading.**
  - A FIFO deque with capacity 10 000 (`learner.py:168`).
  - Uniform sampling without replacement within a minibatch (`learner.py:223`). No prioritisation, no n-step returns.
  - Training writes exactly 10 000 transitions, so the buffer never evicts during a replicate.
  - The logging memory is a separate list and is diagnostic only (`learner.py:207-213`).
- **Status:** settled.

### 3e. Evaluation

- **Reading.**
  - After training, the harness calls `evol_deepmind` again with the trained agent in pre-trained mode (`R/test_harness.R:65`, `R/test_harness_fit.R:45`, `R/test_harness_noise.R:48`).
  - That branch clears the logging memory and reruns `practice` with **the agent's own** hyperparameters: 500 episodes, epsilon set to 0 every episode, no training (`exp.py:124-127`, `learner.py:472-473`, `:525`). The fresh hyperparameter object built from the signature defaults is discarded.
  - The final weights are used; there is no checkpoint selection. Noise stays on.
  - The environment's episode counter continues, so evaluation episodes are numbered 501..1000. The figure code keeps RL rows only with episode number above 500 (`R/prep_figs.R:142-143`, `:226-227`).
  - The random and MDP conditions are the 500-episode runs made inside the training call (`exp.py:134-143`).
- **Conflicting reading.**
  - The base-case density panel is drawn from replicates 1..30 only (`R/figs.R:72`, called with `replicates = 30` at `R/figs.R:877-878`).
  - P and S line 120 report 100 replicates.
- **Status:** settled.

## 4. Noise

### 4a. Definition

- **Reading (E).**
  - Each step draws one scalar from a normal distribution with mean 0 and **standard deviation** 0.05 × m, where m is the noise modifier (`evol_game.py:319-327`).
  - The draw is added to the fitness reading the agent sees (`evol_game.py:176-178`), never to the reward or the logged metric.
  - The reset also adds a draw to the stored initial fitness (`evol_game.py:384-385`). The forced first step then overwrites it (`evol_game.py:188`), but it does reach the first acting input of each episode through the forced step's sensor.
  - The definition has been the same since at least E 42e0925 (2021-10-06).
- **Conflicting reading.**
  - P (gatherer note): "f' = f + N(0, 0.05 × σ²)", with σ from 0 to 100.
  - S eq. 2 (line 100): s_t = s_t + w, w ~ N(μ, 0.05 × σ²), μ = 0, with σ² called the "noise parameter" and varied from 0 to 100.
  - If the second argument of N is a variance, the standard deviation would be √(0.05 σ²), not 0.05 m. The code and the text agree only if that argument is a scale and σ² is m.
  - Chosen: the code, because the reported numbers come from it. The paper's notation is recorded as ambiguous.
- **Status:** settled (E), with the textual ambiguity recorded.

### 4b. The "noise-free" base case

- **Reading.**
  - Yes: the base case as the harness runs it has sensor noise on, with modifier 1, so standard deviation 0.05 on the normalised scale at E (`R/test_harness.R:53`, `R/test_harness_fit.R:36`).
  - It affects only RL-fit's observation. RL-genotype sees the one-hot genotype, and random, MDP and two-drug do not read the sensor.
  - The `sigma = 0.5` harness argument is the random-landscape epistasis parameter. It has no effect on supplied landscapes (`evol_game.py:138-141`, `landscapes.py:71-78`).
- **Conflicting reading.** S line 100 says "μ was set to 0 such that σ² = 0 would introduce no noise". The base case does not use 0.
- **Status:** settled.

## 5. ZOX

- **Reading.** The list form (0.805 at genotype index 1) runs on every training, random, MDP and two-drug path (items 2a and 2b).
- **Status:** settled.

## 6. Two-drug cycles

- **Reading.**
  - Unordered pairs of 0-based drug indices i < j, which gives 105 (`exp.py:406-409`).
  - Each pair becomes a state-independent schedule alternating i, j, i, j … over 20 steps (`exp.py:423-436`). It is played through the MDP-policy path, which adds 1 to convert to the 1-based drugs (`learner.py:415`). The 0-based indices are therefore not a bug.
  - Only the order that starts with the lower-index drug is run. The 105 reverse orders are never run.
  - Each pair runs for every starting genotype 0..15. The loop bound is N², which equals 2^N only at N = 4 (`exp.py:412-419`).
  - It runs 100 episodes per pair and starting genotype (`R/prep_figs.R:663`), on the raw scale, after the forced AMP step.
  - Each value is estimated by simulation, not computed exactly.
  - The figure uses starting genotype 0: one mean per pair over its 100 episodes (`R/prep_figs.R:691-703`), drawn as a density beside the per-episode RL densities (`R/figs.R:121-122`).
- **Breakage.**
  - `R/prep_figs.R:668` calls the memory cleaner with one argument, but the version sourced by the figure script takes two (`R/clean_evol_out.R:74`; `R/figs.R:27-28` sources it last).
  - The one-argument cleaner lives in `R/prep_policy_sweeps.R:11-26`, which the figure script does not source.
  - The figure reads a cached `twodrugsweep.Rda` (`R/prep_figs.R:649-653`), which was not released.
- **Comparison rule behind "outperformed all two-drug cycles"** (P; S lines 139-141, 156-157): not found in code. Only the visual density overlay exists.
- **Status:** settled for enumeration, episodes and estimation. The comparison rule is not found.

## 7. Random cycling

- **Reading.**
  - Each replicate has its own 500-episode run inside the training call (`exp.py:141-143`).
  - Every decision is an independent uniform draw over drugs 1..15, on both epsilon branches (`learner.py:483-493`, `:512-513`), after the forced AMP step.
  - The value is estimated by simulation, not computed exactly.
  - Scale: normalised at E.
- **Status:** settled.

## 8. MDP baseline

- **Reading.**
  - Backward induction through the external pymdptoolbox `FiniteHorizon` (`dpsolve.py:135-159`). The horizon is RESET_EVERY, 20 (`learner.py:452-453`), the discount is the agent's, 0.99 (`exp.py:137-138`), and there is no terminal reward.
  - It maximises the discounted sum over 20 steps of 1 − f_{a_t}(s_t), with the reward on the pre-transition genotype (`dpsolve.py:94-113`). The scale is the environment's drugs, normalised at E (`learner.py:385-387`).
  - The policy is indexed by state and step; the step index is the decision count after the forced AMP step (`learner.py:479`, `:495`, `:415`).
  - It is evaluated over 500 simulated episodes per replicate.
- **Unconfirmed.** Whether column t of the `FiniteHorizon` policy is the decision at step t from the start depends on pymdptoolbox, which is not in the clones. Not found here. Our solver should assert its own indexing.
- **Fraction of steps using CTX.** Not computed. It would be computed:
  - for the main-figure MDP strip (P Fig. 3A), in `R/prep_figs.R:623-647`. That code tabulates drug against step from the MDP rollouts of one replicate file, the middle-ranked one (`R/prep_figs.R:301-304`), averages per-step selection shares, and is plotted at `R/figs.R:249-252` and `:272-282`. Scale: normalised at E.
  - for the discount sweep (S Fig. S9), in `R/prep_policy_sweeps.R:48-50`. Scale: raw.
  - For E2 we compute it exactly from our own MDP, on the scale fixed by Stop S3. Normalisation changes the MDP's objective, so the share can differ between scales.
- **Status:** settled, except the pymdptoolbox column order (not found).

## 9. Tracing E3 (1.88 random against 1.41 RL-fit at noise modifier 40)

- **Chain as committed.**
  1. `R/test_harness_noise.R` calls `evol_deepmind` with the fitness input, noise modifier m and 10 replicates (`:35-43`, `:127-131`).
  2. It saves one result file per replicate under `data/results`, with M and m in the name (`:55-73`).
  3. The heatmap-preparation function (`R/prep_figs.R:202`) with the noise flag either loads a cached `heatmap_raw_noise.Rda` (`R/prep_figs.R:203-205`) or rebuilds it from the files whose names contain M and a digit (`:65-66`).
  4. The replicate is wrapped to 1..10 (`:234-237`).
  5. Per replicate × episode × modifier, it takes the mean fitness per condition and the benefit = random − RL-fit (`:254-264`).
  6. The noise panel draws boxplots of that benefit by modifier (`R/figs.R:91-96`), saved as the base-case performance figure (`R/figs.R:877-882`).
- **Where it breaks.**
  1. The committed harness runs only modifier 100 (`R/test_harness_noise.R:127`). No committed code produces a modifier-40 file.
  2. The panel shows the **difference**, not the two means. No committed code prints 1.88 or 1.41. A search of R for those values, and for any summary at modifier 40, finds nothing.
  3. At E the harness path is on the normalised scale (item 1e), so no condition can average above 1. The value 1.88 cannot come from E + R as pinned. It fits only raw-scale runs, for example evo_dm before 9865458, which also trained on every step (item 3b).
  4. The harness does not complete against E (H2).
  5. The result files and the cache were not released.
  6. Panel letters disagree: S calls it Fig. 2D (line 145) and the gatherer note's published text calls it Fig. 2C, while the code's tag is B.
- **Episode counts behind the two numbers.**
  - Per the committed code: random comes from 500 training-time episodes per replicate, RL-fit from 500 greedy evaluation episodes per replicate, and there are 10 replicates.
  - That is 5 000 episodes per condition at one modifier. S's Fig. 2 caption, reported by the authors, says the same: "10 fully trained RL-fit replicates … 500 episodes per replicate (5000 total episodes)".
  - How 1.88 and 1.41 were aggregated (the mean of episode means, or the mean over replicates): not found.
- **Status: not traced.** Consequence per the design outcome §5: E3 drops and L3 becomes reported-only (Stop S2).

## 10. SI Fig. S1A "measurable decrease", and the edge weights behind 0.96 and 0.82

### 10a. "Measurable decrease"

- **What the figure code does.**
  - S1A is drawn by `R/figs.R:149-158`. The panel matches S's Fig. S1 caption, A to D (supplement), and `R/figs.R:884-888` saves it.
  - Each replicate gets a boxplot of 500 values with a dotted line at zero.
  - Each value is random − RL-fit for episode k, the mean fitness of the random run's episode k minus the RL-fit evaluation's episode k (`R/prep_figs.R:272-281`).
  - The pairing is by episode index across independent runs.
- **No test, threshold or summary statistic is computed** anywhere in R. A search for t-tests, rank tests and p-values finds none.
  - The only per-replicate summary is the mean benefit (`R/prep_figs.R:284-288`). It is used for ordering and colour (`R/figs.R:243-244`, `R/prep_figs.R:383-386`).
  - "Mean benefit > 0" is therefore a candidate operationalisation, but nothing in the code calls it that.
- **Status: not found.** Per the outcome, L2 uses the Welch fallback.

### 10b. Edge weights and correlations

- **Construction (R).**
  - `R/figs.R:591-612` draws panels A–C with a Pearson `stat_cor` (the `ggpubr` default). Its layout matches S's Fig. S2 caption: A to C are pairwise comparisons of "state transition frequency" with R = 0.96, 0.82 and 0.78, and D is a bar chart of state frequencies with a value heatmap. That is what `R/figs.R:614-655` draws.
  - The inputs are the 16 × 16 state-transition tables from `compute_joint_probability` in state mode (`R/prep_figs.R:249`, `:452-525`):
    - for each condition, each (episode, replicate) group's recorded **post-transition** states are joined in row order;
    - for each predecessor state j, successors are counted by a regular expression that matches j only after a separator, so the **first recorded state of every episode is never counted as a predecessor**;
    - each row is normalised to a conditional frequency, and an empty row gives 0/0;
    - counts are pooled over all replicates and episodes of the condition;
    - the random and MDP rows come from the fitness-input files only (`R/prep_figs.R:241-243`).
  - The correlation runs over the 256 (from, to) cells. Cells that are not finite in either column are dropped by the plotting layer, not by explicit code.
  - The column is called `joint_prob`, but it holds a row-conditional frequency.
- **Unambiguous? No**, for these reasons:
  - P describes "probability of observed state transitions" and specifies none of the first-transition exclusion, the conditional normalisation, the pooling, the zero-cell inclusion or the NaN handling;
  - NaN handling is implicit in the plotting library;
  - the data came from a code version other than E (H2).
- **Consequence.** Per the outcome, the correlations do not gate and are reported only.
- **Status:** construction settled as R code; ambiguity verdict "not unambiguous".

## 11. Seeds

- **Reading.** None are set on any measured-landscape path.
  - E contains no seeding call for numpy, Python's random module or TensorFlow (Grep over `evodm/`).
  - R's only seed call is `set.seed` in `R/generate_landscapes.R:6`. It seeds R's generator for the random-landscape files and does not reach the Python generators.
  - The random and MDP copies run before RL training and draw from the same global generators (`exp.py:134-146`).
- **Status:** settled. Only distribution-level agreement is possible.

---

## Unresolved

1. Which value matches the 2015 Table 4 for ZOX at genotype index 1 (item 2b), and whether the table's genotype order matches the code's bit-pattern order (item 2c). Both belong to E1.
2. The exact RL-genotype harness call. No active call exists at R (item 3c). Reading adopted by substitution.
3. Adam parameters other than the learning rate: Keras defaults of an unpinned TensorFlow 2.11.x, not in the clones (item 3a).
4. pymdptoolbox `FiniteHorizon` policy column order (item 8).
5. The comparison rule behind "outperformed all two-drug cycles" (item 6).
6. The definition of "measurable decrease" (item 10a).
7. How 1.88 and 1.41 were aggregated, and the code path that printed them (item 9).
8. The evo_dm commit that produced the published figures (H1, H2).
9. The published text and the SI, not opened in this look. Every P-tier reading above rests on the gatherer note's extraction-tool quotes and on S, which is a pre-publication manuscript.

## Stops

- **S1. Mixed scales in a two-drug comparison.** At E + R, the base-case density panel overlays RL-fit, RL-genotype, random and MDP on the **normalised** scale with two-drug cycles on the **raw** scale (`R/figs.R:119-123`; items 1e, 1f). Per the reproduction outcome this is a stop. L1's two-drug arm needs an owner decision on the scale before the preregistration.
- **S2. E3 not traced** (item 9). E3 drops and L3 becomes reported-only. Gate 0/1 becomes E2, L1 and L2 at their exact cut-offs, plus the discrimination controls and E3′ (the exact random expectation against our Monte Carlo within 3 SE), per design outcome §5. Record this before any run.
- **S3. The base-case scale is not identifiable from the pinned commits.**
  - E runs the harness path normalised.
  - The reported noise values (> 1) and the density axis (0 to 3.3) point to raw-scale figures.
  - E history shows the measured-landscape branch forced raw before 2023-02-23.
  - Normalisation changes the RL reward and the MDP objective, so it bears on L1, L2 and E2.
  - Owned by the preregistration: owner decision.
- **S4. The published DQN's update budget is not identifiable.**
  - E with the harness gives 91 gradient updates per replicate (item 3b).
  - The figure-era code (before 2023-05-18, H2) updated on every decision step after warm-up.
  - This decides what "the unchanged published DQN" means at rungs 1 and 3, and what the starved-budget control's "10% of the gradient updates" is 10% of.
  - Owner decision before the preregistration.
- **S5. The committed harness does not run against the pinned evo_dm (H2), and the batch scripts install it unpinned (H1).** The reproduction outcome's route ("the runs use the values on the code path the authors' harness executes") cannot be met literally at 16335ce + d68cf07. This SPEC therefore specifies E's behaviour where E and R disagree. Owner to confirm that reading, or name another commit.
