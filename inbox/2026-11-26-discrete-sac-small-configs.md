# Comparator: discrete SAC (Christodoulou 2019), licences and small-task configurations

Tag: BASELINE
Access date: 2026-11-26
Status: gathered, not promoted. Supplements inbox/2026-11-16-discrete-sac.md (not
repeated here). Pages read through an extraction tool; commit hashes from the GitHub
REST API. Nothing cloned, downloaded or run. Numbers are "reported by" or are
repository defaults, never results of this project.

## 1. Original author's code: licence and a small discrete configuration

- Repository: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
  Licence: MIT (repository sidebar; licence text not opened).
- SAC-Discrete file: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
  last commit touching it c2da672abe6073ff197e8330d585bb70a15081f6 (2020-09-19).
  As extracted: target entropy `-np.log((1.0 / self.action_size)) * 0.98`; critic target
  is the exact expectation over actions of min(target Q1, target Q2) - alpha*log pi;
  log_alpha initialised at 0, Adam with the actor learning rate; Adam eps 1e-4 on all
  optimisers; 1e-8 added before log.
- Alpha loss (inherited from SAC.py, same repository, opened):
  `alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()`.
  Whether the discrete subclass passes the expected log-probability or a sampled one
  into this loss: NOT CHECKED.
- Configuration file: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/results/Cart_Pole.py
  last commit touching it 046941770898555b7bea5fc662e7f9707a41056d (2019-09-16).
  Exact claim: the "Actor_Critic_Agents" dictionary, run on CartPole-v0 (2 actions,
  vector observation), 450 episodes, 1 run per agent, agents including SAC_Discrete.
  Values as extracted: actor lr 3e-4, critic lr 3e-4, actor and critic hidden [64, 64],
  tau 0.005, buffer 1e6, batch 256, discount 0.99, min_steps_before_learning 400,
  update_every_n_steps 1, learning_updates_per_learning_session 1 (update ratio 1:1),
  automatic entropy tuning True, gradient clipping norm 5, Xavier init. These are
  configuration-file values, not stated anywhere as tuned; one run per agent.

## 2. Tianshou discrete SAC unit-test configuration (small discrete task)

- File: https://github.com/thu-ml/tianshou/blob/master/test/discrete/test_discrete_sac.py
  last commit touching it 770334d93b356c06cb925697e9f8df394411c1f4 (2025-10-29).
  Licence: MIT (GitHub API, https://api.github.com/repos/thu-ml/tianshou).
- Exact claim: a unit test that must reach reward threshold 170 on CartPole-v1. Defaults
  as extracted: buffer 20000, actor lr 1e-4, critic lr 1e-3, alpha lr 3e-4, gamma 0.95,
  tau 0.005, fixed alpha 0.05 (auto-alpha off by default; when on, target
  0.98*log(action_dim)), 5 epochs x 10000 steps, 10 steps per collect, update-per-step
  0.1, batch 64, hidden [64, 64], 10 training envs, n-step 3.
- Status: a passing-threshold test configuration, not a published tuned configuration.

## 3. CleanRL licence

- https://github.com/vwxyzjn/cleanrl/blob/master/LICENSE : MIT, "Copyright (c) 2019
  CleanRL developers" (the API reports NOASSERTION; the file text is MIT).

## 4. Conclusion for this project

A published, tuned discrete-SAC configuration for a small discrete task with 4 to 20
actions and 12 000 to 48 000 decisions: NOT FOUND. The nearest are untuned CartPole
configurations (sections 1 and 2) and the TES-SAC LunarLander runs (see
inbox/2026-11-26-discrete-sac-target-entropy.md), which used one Atari-style
hyperparameter table.
