# Comparator setting: target entropy for discrete SAC, including the failure of 0.98 log|A|

Tag: BASELINE
Access date: 2026-11-26
Status: gathered, not promoted. Supplements section 3 of inbox/2026-11-16-discrete-sac.md.
Numbers are "reported by" the cited authors, never results of this project.

## 1. TES-SAC (target entropy annealing)

- https://arxiv.org/html/2112.02852 (v1, 2021-12-06): "Target Entropy Annealing for
  Discrete Soft Actor-Critic", Xu, Hu, Liang, McAleer, Abbeel, Fox. Venue on the arXiv
  page: none; the PDF header (search snippet, not read) says NeurIPS 2021 workshop.
- Exact claims as extracted: Christodoulou "suggests setting the target entropy to 98%
  of the maximum possible policy entropy" and "in most scenarios it greatly
  underperforms" (Section 1). Constant targets C*log|A| with C in {0.98, 0.5, 0.01}
  (Section 4). Fig. 3: with C = 0.01 on MsPacman alpha decays exponentially because
  policy entropy plateaus near 0.1, above the target.
- Schedule (Section 3.2, Algorithm 1): exponential moving mean and standard deviation of
  policy entropy; when the mean is within 0.01 of the target and the standard
  deviation below 0.05 for T counted steps, target is multiplied by 0.9; window
  discount 0.999. T, initial alpha and alpha lr: not given.
- Table 1 (Appendix A): lr 3e-4, Adam, batch 256, gamma 0.99, buffer 1e5, 2 x 512 ReLU,
  tau 0.005, target update interval 1, 1 gradient step.
- Reported by the authors, Table 2 (Appendix E), 3 runs, 1M interactions, LunarLander
  (4 actions, vector observation; the only small discrete task found with a target
  sweep): C=0.98 -217.63 (8.21); C=0.5 187.17 (57.81); C=0.01 -251.57 (623.25);
  TES-SAC 277.43 (5.46) (standard deviations in parentheses). Text: TES-SAC beats
  constant targets in 8 of 24 environments. Whether LunarLander used a separate network:
  not stated.
- Code: not found (no repository linked on the arXiv HTML page).

## 2. CleanRL 0.89 (already filed 2026-11-16; not repeated)

## 3. Decoupling actor and critic entropy

- https://arxiv.org/abs/2509.09838 : "Dissecting Discrete Soft Actor-Critic: Limitations
  and Principled Alternatives", Asad, Babanezhad, Vaswani; v2 2026-05-11; no venue
  listed. Exact claim (abstract, paraphrased by the extraction tool): the coupling
  between actor and critic entropy is "the primary reason behind the poor performance
  of DSAC"; setting the critic entropy coefficient to zero while keeping the actor
  entropy "significantly improves DSAC" on Atari. The paper notes "Prior work
  attributes this weakness to the use of a fixed target entropy" (citing Xu et al.
  2021). Experiments use stable-baselines3, 5 seeds, Atari and 20 stable-retro games.
  Hyperparameters (Appendix E.1): NOT READ (page truncated in extraction). Code: not
  found on the abstract page.
- Relevance: a decoupled-critic variant is a candidate strongest-honest discrete SAC
  and a reviewer may ask for it; adding it is a scope decision.

## 4. Located but not opened

- IEEE TSMC 55(2), 2025, DOI 10.1109/TSMC.2024.3505613, "On the Effectiveness of
  Regularization Methods for Soft Actor-Critic in Discrete-Action Domains": page did
  not load; content unsourced.
- Ray forum thread on whether the discrete-SAC default target is too high:
  https://discuss.ray.io/t/target-entropy-in-discrete-sac-implementation/12182 (not
  opened; forum post, not evidence).
- DSAC-C, https://arxiv.org/abs/2310.17173 (IJCNN'25, Neo and Chen): abstract opened;
  constrained maximum-entropy discrete SAC on Atari; no code link. Not a target-entropy
  source.

## 5. Bearing on the protocol (analysis, not sourced)

The CLAUDE.md learned-temperature candidates c in {0.77, 0.89} lie between the TES-SAC
constants that were reported to fail (0.98) and to work on LunarLander (0.5). No source
found measures c in that interval on a small task.
