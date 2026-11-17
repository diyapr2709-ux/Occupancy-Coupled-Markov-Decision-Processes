# Comparators: PPO with entropy bonus, PPO-Lagrangian

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. Lower-priority pass: implementations and default
configurations were opened; the benchmark papers' result tables were not. Pages were
read through an extraction tool; hashes are from the GitHub REST API. No code was
cloned or run.

## PPO with entropy bonus

Algorithm paper: arXiv 1707.06347, "Proximal Policy Optimization Algorithms"
(abstract page opened: https://arxiv.org/abs/1707.06347). No code link on the page.
Reported numbers: not extracted (abstract only).

Reference implementation: CleanRL cleanrl/ppo.py
- https://github.com/vwxyzjn/cleanrl, commit e421c2e50b81febf639fced51a69e2602593d50d
  (2023-11-28, last commit touching the file). File opened at this commit.
- Defaults: learning_rate 2.5e-4 with linear annealing, num_envs 4, num_steps 128,
  gamma 0.99, gae_lambda 0.95, num_minibatches 4, update_epochs 4, advantage
  normalisation on, clip_coef 0.2, clipped value loss on, ent_coef 0.01, vf_coef 0.5,
  max_grad_norm 0.5, target_kl None. Separate actor and critic, two tanh layers of 64,
  orthogonal init (policy head std 0.01, value head std 1.0), Adam eps 1e-5.

Published tuned configurations (small discrete-action problems):
RL Baselines3 Zoo hyperparams/ppo.yml
- https://github.com/DLR-RM/rl-baselines3-zoo, commit
  eec15dc22e1ec7167064aae0c1218eefccaf0ff4 (2026-04-01, last commit touching the file).
  File opened at this commit.
- CartPole-v1: n_envs 8, 1e5 steps, n_steps 32, batch 256, n_epochs 20, gamma 0.98,
  gae_lambda 0.8, ent_coef 0.0, learning rate linear from 0.001, clip linear from 0.2.
- LunarLander-v3: n_envs 16, 1e6 steps, n_steps 1024, batch 64, n_epochs 4,
  gamma 0.999, gae_lambda 0.98, ent_coef 0.01.
- MiniGrid family: n_envs 8, n_steps 128, batch 64, n_epochs 10, gamma 0.99,
  gae_lambda 0.95, ent_coef 0.0, learning rate 2.5e-4, clip 0.2, normalisation on.

A published tuned configuration for a discrete, delayed-credit problem of about 40
steps per episode: NOT FOUND. The configurations above are the nearest published
ones and none has that structure. What follows from the structure (analysis, not
sourced): the rollout length should be a whole number of episodes so that GAE never
bootstraps mid-episode from a value that cannot see the delayed consequence; gamma and
gae_lambda near 1 are the settings that matter for a consequence arriving `delay`
quarters later; ent_coef has to be in the tuning grid since the published values
range from 0 to 0.01. These are tuning-grid suggestions, not published settings.

## PPO-Lagrangian

Reference implementation 1 (original): openai/safety-starter-agents
- https://github.com/openai/safety-starter-agents, commit
  4151a283967520ee000f03b3a79bf35262ff3509 (2019-11-21, HEAD). README and
  safe_rl/pg/run_agent.py opened at this commit. Archived; "code is provided as-is, no
  updates expected". TensorFlow 1 era. Licence: not stated on the README.
- Companion to "Benchmarking Safe Exploration in Deep Reinforcement Learning" (2019).
  That paper was NOT opened; its reported numbers are not recorded here.
- run_polopt_agent defaults: steps_per_epoch 4000, epochs 50, max_ep_len 1000,
  gamma 0.99, lam 0.97, cost_gamma 0.99, cost_lam 0.97, ent_reg 0, cost_lim 25
  (the script's command line default is 10), penalty_init 1.0, penalty_lr 5e-2,
  target_kl 0.01, vf_lr 1e-3, vf_iters 80, hidden 64x2.
- Multiplier: softplus of a free parameter, initialised so the multiplier equals
  penalty_init; loss `penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)`;
  optional rescaling of the penalised objective by 1/(1 + penalty).
- The README itself says its PPO is "not the strongest PPO implementation": no
  observation or reward normalisation, no clipped value loss, early stopping on KL.
  Seeds used in the paper per the README: 0, 10, 20.

Reference implementation 2 (maintained): OmniSafe
- https://github.com/PKU-Alignment/omnisafe, commit
  15603dd7a654a991d0a4648216b69d60b81a6366 (2025-03-17, HEAD).
  omnisafe/configs/on-policy/PPOLag.yaml opened at this commit.
- Defaults: steps_per_epoch 20000, update_iters 40, batch 64, target_kl 0.02 with KL
  early stop, entropy_coef 0.0, gamma 0.99, cost_gamma 0.99, lam 0.95, lam_c 0.95,
  clip 0.2, max_grad_norm 40, observation normalisation on, reward and cost
  normalisation off, actor and critic 64x64 tanh at lr 3e-4 with linear decay.
  Lagrange block: cost_limit 25.0, lagrangian_multiplier_init 0.001, lambda_lr 0.035,
  Adam.
- All per-environment overrides in that file are for continuous control. A tuned
  PPO-Lagrangian configuration for a discrete-action problem: NOT FOUND.

Notes for a strongest-honest build (analysis, not sourced):
- The two references disagree by three orders of magnitude on the multiplier's initial
  value (1.0 versus 0.001) and both use a cost limit of 25 for 1000-step episodes;
  neither transfers. The cost limit must be stated in this problem's own units and the
  multiplier's initial value and learning rate tuned at parity.
- Plain gradient ascent on the multiplier is known to oscillate; a PID-controlled
  multiplier is the usual stronger variant. No source for that was opened in this
  pass, so it is listed as a suggestion without citation.
- The entropy bonus defaults to zero in both references; it has to be switched on and
  tuned for the comparison with entropy-regularised agents to be fair.

## Release status

- CleanRL, RL Baselines3 Zoo, safety-starter-agents, OmniSafe: code released.
- Numbers reported by the original authors of either algorithm: not extracted in this
  pass (abstract only for PPO; benchmark paper not opened for PPO-Lagrangian).
