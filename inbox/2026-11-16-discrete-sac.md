# Comparator: discrete-action soft actor-critic

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. All numbers are "reported by" the cited authors and
are not results of this project. Pages were read through an extraction tool; hashes
are from the GitHub REST API. No code was cloned or run.

## 1. Reference implementations

| Source | URL | Commit | Licence | Notes |
|---|---|---|---|---|
| Original author's code (named in the paper's conclusion) | https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch | 4835bac8557fdacff1735eca004e35ea5a4b7443 (2023-08-23, HEAD of default branch) | not checked | The SAC-Discrete file itself was not opened; only the commit list was. |
| CleanRL single-file discrete SAC, cleanrl/sac_atari.py | https://github.com/vwxyzjn/cleanrl | 004f8a086a892a2a180f4dd332b90d83a968aa7a (2025-07-08, last commit touching the file) | not checked | File opened at this commit. Convolutional encoder for Atari; the update rule is what transfers. |
| Stable discrete SAC ("SD-SAC") | https://github.com/coldsummerday/SD-SAC | 006809db08513fbeea2bfaecea1c03b59c3bb0d5 (2024-11-23, HEAD) | not checked | Repository source not opened; only the commit list was. |

Recommended starting point for a strongest-honest baseline: the CleanRL update rule
(exact expectation over actions in both the critic target and the actor loss, twin
critics, learned temperature) with the stabilisers from section 4.

## 2. Published configurations

Original paper, arXiv 1910.07207, "Soft Actor-Critic for Discrete Action Settings"
(opened via https://ar5iv.labs.arxiv.org/html/1910.07207), Table 2, Appendix B:
batch 64, replay 1e6, discount 0.99, learning rate 3e-4, Adam, 4 environment steps
per update, hard target update every 8000 steps, 20,000 initial random steps, reward
clipping, entropy target "0.98 * (-log (1 / |A|))". The paper states no tuning was
done. Target smoothing coefficient, temperature learning rate and initial temperature
are not given in the table.

CleanRL sac_atari.py defaults at the pinned commit: total_timesteps 5e6, buffer 1e6,
gamma 0.99, tau 1.0, batch 64, learning_starts 2e4, policy_lr 3e-4, q_lr 3e-4,
update_frequency 4, target_network_frequency 8000, alpha 0.2 initial, autotune True,
target_entropy_scale 0.89, Adam eps 1e-4 for all three optimisers. Target entropy
line as extracted:
`target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))`.
CleanRL documentation (https://docs.cleanrl.dev/rl-algorithms/sac/, opened): 0.89 is
used instead of the paper's 0.98 for "improved stability when training for more than
100k steps", and "Tuning this parameter to the environment at hand is advised and can
lead to significant performance gains."

SD-SAC paper, arXiv 2209.10081v4, "Revisiting Discrete Soft Actor-Critic" (opened at
https://arxiv.org/html/2209.10081v4; the final 1% of the page was not read), Table 3
and Appendix B.2 as extracted: entropy-penalty coefficient beta 0.5, Q-clip range
c 0.5, fixed temperature alpha 0.05, learning rate 1e-5 (Adam), batch 64, buffer 1e5,
gamma 0.99, tau 0.005, n-step 3, two hidden layers of 512.

A published tuned configuration for a SMALL discrete problem (tens of actions, short
episodes, vector observations): NOT FOUND. Every configuration located is for Atari.
The Atari settings that will not transfer unchanged: 20,000 warm-up steps, hard target
updates every 8000 steps, 1e6 replay, one update per 4 steps. These have to be tuned
at parity on dev seeds; that tuning is this project's, not a published configuration.

## 3. Target-entropy convention

- Original: 0.98 * log|A| (paper Table 2). The SD-SAC paper and CleanRL both repeat
  this as the default convention.
- CleanRL: 0.89 * log|A|, with the advice to tune it per environment.
- Annealed target: arXiv 2112.02852, "Target Entropy Annealing for Discrete Soft
  Actor-Critic" (abstract page opened: https://arxiv.org/abs/2112.02852). NeurIPS 2021
  Deep RL workshop. Proposes scheduling the target entropy downward during training
  and compares with constant-target SAC on Atari. Code: none linked. Numbers: not
  extracted (abstract only).
- Consequence for this project: a target near log|A| forces a near-uniform policy, so
  the fraction c is the single most consequential setting for an entropy-space
  comparison. The published values (0.98, 0.89) come from an 18-action pixel domain
  with untuned or lightly tuned settings; neither is evidence for the right c on a
  small problem. The per-agent scaling c*log(K) versus c*log(M) follows the same
  convention applied to each agent's own entropy space.

## 4. Known failure modes a strong baseline must handle

Reported by the SD-SAC authors (arXiv 2209.10081v4):
- "Unstable coupling training": deceptive early rewards sharpen Q, policy entropy
  collapses fast, and the learning target becomes unstable. Their remedy: an entropy
  penalty in the policy objective, beta * 0.5 * (H_old - H)^2, with H_old the entropy
  of the data-collecting policy.
- "Pessimistic exploration": clipped double Q (min of two critics) underestimates in
  discrete settings. Their remedy: average the two target critics instead of taking
  the minimum, plus clipping the critic update within c of the target critic.

From the CleanRL implementation notes (opened):
- A target scale of 0.98 is unstable beyond 100k steps; lowered to 0.89.
- Adam epsilon raised to 1e-4 for stability.
- Updating only every n-th environment step avoids degradation in long runs.

From the temperature rule itself (analysis, not sourced):
- With automatic temperature and a target above what the reward-optimal policy needs,
  alpha grows without bound until the policy is forced to the target entropy; with a
  target that an action mask makes unreachable (entropy of the feasible set below the
  target), alpha diverges. A masked problem needs the target computed against the
  feasible set, and alpha clamped or the target capped at a fraction of
  log(number of feasible actions).
- Entropy and expectations must be taken over the masked distribution, with 0*log 0
  handled, otherwise the alpha loss produces NaN.
- A fixed, tuned alpha (as in the SD-SAC configuration) is a legitimate strong variant
  and should be in the tuning grid next to the automatic rule.

## 5. Numbers reported by the original authors

Reported by the author of arXiv 1910.07207, Table 1 and Figure 1 (20 Atari games,
5 seeds, 100,000 steps, score at end of training, Rainbow numbers taken from prior
work): SAC-Discrete beat Rainbow in 10 of 20 games; median relative performance -1%,
maximum +4330%, minimum -99%. Examples: MsPacman 690.9 vs 364.3; Crazy Climber 3668.7
vs 12558.3.

Reported by the authors of arXiv 2209.10081v4, Table 1 (20 Atari games, 3 seeds, 10M
steps, evaluation every 50k steps over 10 episodes, normalised scores): mean SD-SAC
220.0% vs discrete SAC 151.4%, Rainbow 187.4%; median 114.1% vs 90.8% vs 79.2%.

Reported by the CleanRL maintainers (docs page, training return, seed count not
stated): Pong 100k -20.21 +/- 0.62; BeamRider 100k 396.15 +/- 155.81; at 5M steps
Pong 19.24 +/- 1.81, Breakout 343.66 +/- 93.34, BeamRider 8658.97 +/- 1554.66.

None of these is on a domain comparable to a 40-quarter stewardship episode.

## 6. Release status

- arXiv 1910.07207: code released (repository above). Data: not applicable (Atari).
- arXiv 2209.10081: code released (repository above); not inspected beyond the commit
  list.
- arXiv 2112.02852: code not found.
- CleanRL: code released; benchmark logs are referenced by its docs but were not
  opened.
