# Prior art: Claim B (entropy on the pushforward of the policy onto action classes)

Access date for every entry: 2026-11-16.

Claim under test: maximum-entropy RL (SAC) where the entropy bonus is taken on the pushforward of the policy through a map from actions to equivalence classes (drugs to resistance mechanisms via a cross-resistance matrix) instead of on the action distribution. This includes a cluster-free, similarity-sensitive (Leinster-Cobbold-style) entropy. Hypothesis: class entropy beats action entropy when classes have unequal sizes.

Search status: cut short at the turn limit. See the end of this file.

## PRE-EMPTS, and the BASELINE it supplies

### Action Redundancy in Reinforcement Learning (UAI 2021)
- URLs: https://arxiv.org/abs/2102.11329, read in full at https://ar5iv.labs.arxiv.org/html/2102.11329; proceedings: https://proceedings.mlr.press/v161/baram21a.html
- Exact claim (from the text):
  - Action entropy is wasted on redundant actions that lead to the same next state.
  - The paper maximises transition entropy instead: g_pi(s, a) = H(s'|s, a) + KL(P(.|s, a) || P(.|s; pi)), where the KL term is the action redundancy.
  - In deterministic MDPs, g_pi(s, a) = -log(pi(a|s) + eta^pi(s, a)), where eta is the probability mass on other actions with the same next state. The paper describes this as maximising action entropy of a modified action space in which all redundant actions are unified.
  - Figure 1 shows the unequal-class case: two actions lead to s1 and one to s2, so action entropy puts 2/3 of the mass on s1.
  - Algorithms: MinRed SAC, with a redundancy-corrected soft reward and a learned inverse model q(a|s, s'); MinRed Q-learning.
  - Experiments: grid world with n redundant actions (SAC fails as n grows, MinRed SAC succeeds), Atari with macro actions, sparse MuJoCo.
- Tag: PRE-EMPTS, and BASELINE.
- Reason: in deterministic MDPs this is SAC with entropy on the pushforward of the policy onto action equivalence classes, motivated by the same unequal-class-size failure of action entropy.
- What would still be novel if it stands:
  - the classes come from a known prior map (a cross-resistance matrix) defined by delayed, long-run effects (shared resistance mechanisms), not by identical next states learned through an inverse model;
  - the cluster-free, similarity-sensitive (Leinster-Cobbold) entropy;
  - the interaction with occupancy-coupled dynamics and the stewardship efficacy outcome.
  The general claim "entropy over classes beats entropy over actions when classes are unequal" is already argued and shown on a synthetic task here. It should not be claimed as new.
- BASELINE note: MinRed SAC is a direct comparator. Run it with the learned inverse model, or with the known class map in the deterministic ARS form. No reference implementation is linked from the paper. Appendix B lists Stable Baselines3 and the hyperparameters, so those are the published configuration to start from.

## RELATED

- https://arxiv.org/abs/2512.04359 | Efficient RL with Semantic and Token Entropy for LLM Reasoning (SENT, Dec 2025). Semantic entropy over equivalence classes of responses orders training data (curriculum); KL regularisation is applied to low-entropy tokens. | RELATED | per the abstract, class-level entropy is used for curriculum, not as a policy entropy bonus. Not verified against the full text.

## Seen only in search-result snippets (not opened, not tagged)

- https://arxiv.org/html/2601.03064 Similarity-Sensitive Entropy under Representation Change and Inference (theory of Leinster-Cobbold entropy; no RL in the snippet)
- https://arxiv.org/html/2511.03849 Which Similarity-Sensitive Entropy (Sentropy)?
- https://arxiv.org/html/2509.02930 VendiRL: self-supervised RL of diverse skills using the Vendi score as an intrinsic reward. Similarity-kernel entropy is used over skills/trajectories, not over the action distribution. Open before ruling it out.
- Temporal Vendi Score (ICLR 2026 workshop): https://iclr.cc/virtual/2026/10012551
- https://arxiv.org/pdf/2603.11682 Entropy-Preserving Reinforcement Learning
- https://www.mdpi.com/1099-4300/28/7/811 Entropy regularization review (2026)

## Not searched before the turn limit

- arXiv API queries 6 and 7 from the standing query file (entropy regularisation AND action abstraction / equivalence classes; similarity-sensitive AND entropy AND cs.LG);
- follow-ups citing the Action Redundancy paper (anything later that fixes action-entropy redundancy);
- mutual-information / marginal-prior regularised soft Q-learning;
- "state-action entropy with duplicate actions", "entropy over action effects / outcomes", "action embedding entropy";
- semantic-entropy bonuses in LLM RL (other than SENT).
