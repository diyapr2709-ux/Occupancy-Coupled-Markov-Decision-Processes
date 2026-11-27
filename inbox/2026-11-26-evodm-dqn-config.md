# evo_dm DQN configuration for drug cycling (Weaver et al. 2024)

Access date for every item: 2026-11-26. Gatherer: baseline-hunter.
Question: does a citable tuned DQN configuration exist?

## Paper

- https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439
  Weaver, King, Maltas, Scott, "Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance", PNAS 121(16) e2303165121, 2024, DOI 10.1073/pnas.2303165121.
  Reported by the authors (main text, Methods): training for 500 episodes of 20 evolutionary time steps. The RL-genotype network is input, two 1D convolutional layers, max pooling, dense 28, linear output. The RL-fit network is input, dense 64, dense 28, linear output. MSE loss, and a target network "only updated rarely". The text says 500 episodes was chosen "after hyperparameter tuning showed decreased or equal effectiveness with additional training".
  The main text does not report learning rate, discount, batch size, replay size, target-update interval, optimiser or epsilon schedule. No sweep table or figure was found. The SI was not opened.
  Code statement: code at https://github.com/DavisWeaver/evo_dm, figure code at https://github.com/DavisWeaver/rl_cycling. No Zenodo DOI. The data come from a previously published PLOS ONE dataset (journal.pone.0122283).
  Tag: BASELINE.

## Reference implementation

- https://github.com/DavisWeaver/evo_dm. Licence: GPL-3.0 (from the repository page). Latest commit on main: d975114b8cd01dad8036a21dba2b086cee149157, 23 Aug 2024, "update_target_counter back".
  Defaults in https://raw.githubusercontent.com/DavisWeaver/evo_dm/main/evodm/learner.py, class `hyperparameters`, at main as of the access date (the raw file was read at main, not pinned to the hash above):
  REPLAY_MEMORY_SIZE = 10000; MIN_REPLAY_MEMORY_SIZE = 1000; MINIBATCH_SIZE = 100; UPDATE_TARGET_EVERY = 310 (the inline comment says "every 500 steps"); DISCOUNT = 0.99; epsilon = 1; EPSILON_DECAY = 0.95; MIN_EPSILON = 0.001; LEARNING_RATE = 0.0001; EPISODES = 500. The state-vector network is Conv1D(64,3) x2, MaxPooling1D(2), Flatten, Dropout(0.2), Dense(28), linear output. Adam, MSE loss.
  These are code defaults. They are not a documented tuned configuration, and the paper does not say they are the values behind its figures.
  Tag: BASELINE.
- https://github.com/DavisWeaver/rl_cycling. No licence shown, no LICENSE file in the listing, 3 commits. The README gives no hyperparameters or run arguments. The batch/ scripts were not opened, so whether the figure runs override the defaults is not determined.
  Tag: BASELINE.

## Verdict

A citable tuned configuration was not found. The paper reports only the episode count as tuned. The rest is code defaults at an unpinned main, with one inconsistency between a comment and its value (UPDATE_TARGET_EVERY). Licence conflict for reuse: GPL-3.0 (evo_dm); none stated for rl_cycling.

## Unsourced

- The SI hyperparameter table, if any (SI not opened).
- The arguments actually used for the figure runs (rl_cycling batch/ not opened).
