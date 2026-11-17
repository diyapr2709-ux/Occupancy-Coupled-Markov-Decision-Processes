# Comparators: sliding-window UCB and discounted UCB

Tag: BASELINE
Access date: 2026-11-16
Status: gathered, not promoted. Numbers and settings are "reported by" the cited
authors. Pages were read through an extraction tool; the hash is from the GitHub REST
API. No code was cloned or run.

## Original formulation

"On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems",
arXiv 0805.3415 (submitted 2008; published at ALT 2011 per the reference
implementation's docstring).
Opened: https://arxiv.org/abs/0805.3415 and https://ar5iv.labs.arxiv.org/html/0805.3415
(the PDF itself could not be parsed).

Rewards bounded by B; Upsilon_T = number of breakpoints up to horizon T.

- Discounted UCB (credited in the paper to earlier work from 2006): index = discounted
  empirical mean, weights gamma^(t-s), plus padding
  c_t(gamma, i) = 2B * sqrt(xi * log n_t(gamma) / N_t(gamma, i)),
  with N_t the discounted pull count of arm i and n_t the sum over arms.
- Sliding-window UCB (introduced in this paper): index = empirical mean over the last
  tau plays plus padding c_t(tau, i) = B * sqrt(xi * log(min(t, tau)) / N_t(tau, i)).

Recommended settings, reported by the authors:
- gamma = 1 - (4B)^-1 * sqrt(Upsilon_T / T); regret O(sqrt(T * Upsilon_T) * log T).
- tau = 2B * sqrt(T * log T / Upsilon_T); regret O(sqrt(Upsilon_T * T * log T)).
- If breakpoints have density r (Upsilon_T <= r*T): gamma = 1 - sqrt(r)/(4B),
  tau = 2B * sqrt(-log r / r).
- Theory requires xi > 1/2; the simulations use xi = 1/2.
- Lower bound of order sqrt(T) for any policy under abrupt changes; both policies match
  it up to a logarithmic factor. SW-UCB's bound is slightly better.

Simulations reported by the authors (Figure 1): (left) K = 3 Bernoulli arms, T = 1e4,
one arm jumping from 0.4 to 0.9 on 3000 <= t < 5000, with D-UCB at
gamma = 1 - 1/(4*sqrt(T)) and SW-UCB at tau = 4*sqrt(T log T); (right) K = 2 with a
sinusoidal mean. Reported outcome: D-UCB and SW-UCB perform similarly and both beat
UCB-1 and EXP3.S. Results are given as curves; no numeric table.

## Reference implementation

SMPyBandits
- https://github.com/SMPyBandits/SMPyBandits, MIT, commit
  012fc13a9d82e361930fdf701dfff9a91439d60c (2026-06-19, HEAD of master).
- SMPyBandits/Policies/SlidingWindowUCB.py (opened at this commit): classes SWUCB
  (tau default 1000, alpha default 1.0), SWUCBPlus (tau = int(4*sqrt(T*log(T))) from
  the horizon), SWklUCB, SWklUCBPlus. Index: windowed mean +
  sqrt(alpha * log(min(t, tau)) / N). Unpulled-in-window arm gets +inf.
- SMPyBandits/Policies/DiscountedUCB.py (opened at this commit): classes DiscountedUCB
  (alpha 1, gamma 0.99), DiscountedUCBPlus (gamma = 1 - sqrt(Upsilon/T)/4),
  DiscountedklUCB, DiscountedklUCBPlus. Index: discounted mean +
  sqrt(alpha * log(n_gamma) / (2 * N_gamma)). Prints a warning when gamma is close
  to 1.
- Note: the constants in the code (alpha, the factor 2 under the root) are not
  identical to the paper's (2B and xi). A reimplementation should follow the paper and
  state B.

## What does not transfer (analysis, not sourced)

- The recommended tau and gamma assume abrupt, exogenous changes with a known count.
  Here the change is endogenous: resistance moves because of the policy's own
  prescribing shares, with a lag. There is no Upsilon_T to plug in. The window or
  discount therefore has to be tuned on dev seeds at parity; a natural grid is centred
  on the environment's effective memory and delay, in decisions rather than quarters.
- The bandit sees no state. If the observation carries resistance and pending
  occupancy, a context-free bandit is a deliberately weak-information comparator and
  should be labelled so; it answers "does state feedback matter", not "is the entropy
  space right".
- With per-patient feasibility masks the index must be maximised over feasible arms
  only.
- Rewards here are per-decision treatment outcomes in [0, 1], so B = 1.

## Release status

- Paper: no code or data released by the authors (not found).
- SMPyBandits: code released, MIT.
- A published tuned window or discount for an endogenous, lagged non-stationarity:
  NOT FOUND.
