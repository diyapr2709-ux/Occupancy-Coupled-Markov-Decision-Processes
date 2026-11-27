# Claim B1 prior-art search: action entropy versus class entropy under exact interchangeability

Access date for every item: 2026-11-26. Gatherer: lit-scout.

Claim searched: with exactly interchangeable actions (identical embeddings, tied networks), the performance gap between action-entropy SAC and class-entropy (pushforward) SAC at fixed temperatures equals the gap between the two soft-optimal policies of a mean-field model.

No PRE-EMPTS hit among the items opened. Searches for similarity-sensitive (Leinster-Cobbold) entropy used as a policy regulariser in RL found no such paper. This bears on the B2 gate as well.

## Items opened

- https://arxiv.org/abs/2102.11329 (Action Redundancy in Reinforcement Learning, UAI 2021, PMLR 161; ar5iv full text https://ar5iv.labs.arxiv.org/html/2102.11329)
  Claim: action entropy differs from transition entropy when actions are redundant. Proposition 2 writes action entropy as transition entropy plus a redundancy term. Corollary 1 says the two coincide when no two actions share a next state. MinRed SAC and MinRed Q-learning are proposed.
  Tag: BASELINE. Reason: this is the class-versus-action entropy decomposition that B1 checks. The paper gives no formula or test for the performance gap between the two soft-optimal policies; a gridworld with n duplicated actions is shown only in a figure. No official code is linked in the paper. A tuned configuration exists for MuJoCo only (Tables 6 and 7: per-environment alpha and lambda, rule of thumb lambda = 0.01 alpha). There is none for discrete drug choice.
- https://arxiv.org/abs/2312.06258 (No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning, 2023)
  Claim: masks redundant actions using a similarity factor estimated from distances between state distributions, with no prior knowledge needed.
  Tag: BASELINE. Reason: the comparator list includes it if code exists, and it does: https://github.com/zhongdy15/npm. Compatibility with SAC or max-entropy training is not stated in the abstract.
- https://arxiv.org/abs/2601.03064 (Similarity-Sensitive Entropy under Representation Change and Inference, 2026)
  Claim: similarity-kernel entropy, its behaviour under maps, channels and conditioning, data-processing inequalities, and counterexamples to concavity.
  Tag: RELATED. Reason: pure information theory with no RL. Its concavity counterexamples may matter for an H_Z regulariser.
