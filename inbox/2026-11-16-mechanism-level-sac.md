# Required comparator: SAC with entropy over a mechanism-level action space

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. Contains one prior-work item that should be read in
full and assessed for PRE-EMPTS before anything else is built (section 3). Numbers are
"reported by" the cited authors. Pages were read through an extraction tool. No code
was cloned or run.

The comparator: the agent chooses a mechanism class m with a learned categorical
p(m|s); the drug is then drawn uniformly within the class. Entropy bonus and
temperature act on p(m|s).

## 1. Is there a standard formulation with reference code?

NOT FOUND as a named, standard algorithm. What exists is adjacent:

| Work | URL (opened) | What it gives | Code |
|---|---|---|---|
| "Discrete and Continuous Action Representation for Practical RL in Video Games" (Hybrid SAC), arXiv 1912.11077, AAAI-20 RL in Games workshop | https://arxiv.org/abs/1912.11077 and https://ar5iv.labs.arxiv.org/html/1912.11077 | Factored policy pi(a\|s) = pi(a_d\|s) * pi(a_c\|s, a_d); joint entropy written as H(pi(a_d\|s)) + sum over a_d of pi(a_d\|s) * H(pi(a_c\|s, a_d)); SEPARATE temperatures alpha_d and alpha_c, each auto-tuned to its own target. The second level is continuous there, but the decomposition is the chain rule and applies unchanged to a discrete second level. Target values: authors report only "a cursory search over reasonable-looking values". | None released (paper names only a video and reuses third-party evaluation code). |
| "SOAC: The Soft Option Actor-Critic Architecture", arXiv 2006.14363 | https://ar5iv.labs.arxiv.org/html/2006.14363 | Maximum-entropy objective with an entropy term for the option-selection policy and one for the intra-option policy, plus mutual-information and robustness terms; a single alpha in the derived losses. Discrete option space, Gaussian intra-option policies, MuJoCo only. | None released. |
| "Soft Options Critic", arXiv 1905.11222 | https://arxiv.org/abs/1905.11222 | Entropy on both inter-option and intra-option policies. WITHDRAWN by an author; the arXiv comment cites "an error in the definition of the value function". Do not use as a reference. | None. |

Closest usable formulation: the Hybrid SAC two-level factorisation with two
temperatures. The mechanism-level comparator is its special case with the lower level
fixed to uniform (no lower-level parameters, alpha on the upper level only).

Published tuned configuration for this comparator: NOT FOUND. Reference code:
NOT FOUND. It has to be built from the discrete-SAC baseline (see the discrete-SAC
inbox file) by replacing the K-way head with an M-way head and a fixed uniform
within-class sampler.

## 2. Does it coincide with entropy over the pushed-forward policy? (analysis, not sourced)

This section is a derivation, not a citation, and not a measurement. It must be
checked against the code and by a run before it is relied on.

Notation: classes C_1..C_M partition the K drugs, n_m = |C_m|, psi maps a drug to its
class, q = psi#pi is the class marginal of a drug-level policy pi.

Chain rule: H(pi) = H(q) + sum_m q(m) * H(pi(. | m)).

(a) Mechanism-level SAC as a drug policy. With the within-class draw uniform,
pi(a|s) = p(psi(a)|s) / n_psi(a), so H(pi) = H(p) + sum_m p(m) log n_m. Its entropy
bonus H(p) is exactly the pushforward entropy of the drug policy it induces.
Equivalently it is drug-level SAC restricted to within-class-uniform policies, with a
reward correction of -alpha * log n_m; with equal class sizes the correction is a
constant and the two targets differ by log(K/M).

(b) Against a drug-level policy trained with bonus H(psi#pi). The objective functional
is the same. The policy class differs: the drug-level agent may put its within-class
mass anywhere; the mechanism-level agent must spread it uniformly.

(c) Therefore:
- If drugs within a class are exactly interchangeable in reward, cost, feasibility and
  effect on the kernel, every drug-level policy has a mechanism-level twin with the
  same class marginal, the same return and the same bonus. The two problems have the
  same optimal value and the same optimal class marginals. Any measured difference is
  then optimisation (head size M versus K, critic over K versus M actions, gradient
  variance), i.e. a parametrisation effect, not an objective effect. In that regime
  the mechanism-level comparator is expected to be at least as easy to train.
- They separate only when within-class drugs differ: unequal cost or efficacy, partial
  cross-resistance (classes not exact), or per-patient feasibility masks that remove
  some members of a class. Then uniform-within-class is a handicap, and the fair
  version of the comparator is a two-level policy with a LEARNED within-class choice
  at zero within-class temperature (Hybrid-SAC form with alpha_within = 0). That
  variant has the same objective AND the same policy class as a drug-level agent with
  pushforward entropy; if it ties, the contribution is a parametrisation.
- A similarity-sensitive, cluster-free entropy over a soft drug embedding has no
  exact mechanism-level twin when cross-resistance is partial; that is the one setting
  where the objective, not the parametrisation, could carry the difference.

(d) Masking detail for a fair build: with a feasibility mask, "uniform within class"
must mean uniform over feasible members, classes with no feasible member must be
masked at the upper level, and the target must be c * log(number of feasible classes).

Three comparators follow, and a reviewer will ask for all three:
1. mechanism-level SAC, uniform within class (the required one);
2. two-level SAC, learned within-class choice, temperature only on the class level;
3. drug-level SAC with the reward correction of (a), i.e. action entropy minus the
   expected log class size, which is drug-level action entropy made class-size-neutral.

## 3. Prior work on the same idea: entropy over what actions do rather than over actions

"Action Redundancy in Reinforcement Learning", arXiv 2102.11329, UAI 2021.
Opened: https://arxiv.org/abs/2102.11329 and https://ar5iv.labs.arxiv.org/html/2102.11329

Exact claims as extracted:
- "action entropy does not necessarily coincide with state entropy"
- "we propose to maximize the transition entropy, i.e., the entropy of next states"
- transition entropy decomposes into "model-dependent transition entropy and action
  redundancy"
- the approach is described as "a MaxEnt framework in a modified action space that
  softly combines actions according to their induced transitions"
- "action redundancy is a fundamental problem in reinforcement learning"

Content: equivalent actions are those leading to the same next state; in the
deterministic case the bonus is -log(pi(a|s) + probability of the other actions
equivalent to a), which is the log-probability of the action's equivalence class.
Algorithms: MinRed SAC (a modified SAC whose critic target carries a redundancy term
estimated through a learned action posterior q(a|s,s')) and MinRed Q-learning (replay
augmentation with equivalent actions). Reported by its authors: on a four-room grid
with one action duplicated n times, SAC fails to reach the goal as n grows while
MinRed SAC reaches it (Fig. 3); gains on Atari with enlarged action sets (Fig. 4) and
on sparse MuJoCo variants (Fig. 5). No numeric scores are stated in the text. Code:
none found.

Why it matters here: for exactly interchangeable drugs, entropy of the policy pushed
through the class map is the deterministic-case objective of that paper with the
equivalence classes given instead of learned. Differences as far as the extraction
shows: there the kernel is Markov in (s, a) and equivalence is defined by the next
state; it does not treat a kernel that depends on the agent's own past action
distribution, a delay, or a known mechanism structure, and it learns the equivalence
rather than taking it from a cross-resistance matrix.

Assessment: not tagged PRE-EMPTS by this pass, because only an extraction of the
ar5iv text was read and the tag decision is not a gatherer's. It should be read in
full before any claim about "entropy over mechanisms" as a new objective is worded,
and MinRed SAC should be considered as a comparator (no reference code exists, so it
would have to be implemented from the paper).

## 4. Release status summary

- Hybrid SAC (1912.11077): code not found.
- SOAC (2006.14363): code not found.
- Soft Options Critic (1905.11222): withdrawn; code not found.
- Action Redundancy / MinRed (2102.11329): code not found.
