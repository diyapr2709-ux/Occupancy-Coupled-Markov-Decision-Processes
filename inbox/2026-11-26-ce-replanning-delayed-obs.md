# Certainty-equivalent replanning with delayed state observations: references and known bounds

Access date for every item: 2026-11-26. Gatherer: baseline-hunter.
Purpose: sourcing for Claim A component (iii), the suboptimality of certainty-equivalent (CE) replanning, and the CE replanner listed as a non-learning reference in arena/2026-11-23-scope-outcome.md item 3.
Setting it must fit: 2 to 20 discrete actions, quarterly decisions, reporting delay 0 to 2 quarters, exponentially weighted action memory, Gaussian process noise on resistance.

## Standard references (method definition)

- https://portal.mardi4nfdi.de/wiki/Publication:2511993
  Bertsekas, "Dynamic programming and suboptimal control: a survey from ADP to MPC", European Journal of Control 11 (2005) 310-334, DOI 10.3166/EJC.11.310-334. The page gives the DOI and zbMATH keywords and no abstract. The claim below comes from search-result summaries, because the full text was not opened.
  Claim (from search-result summaries): common MPC schemes can be viewed as rollout algorithms with a sequentially improving base policy, related to policy iteration.
  Tag: BASELINE (the method definition for the CE replanner). The rollout view gives the cost-improvement property against the base policy. It does not bound the gap to the optimal closed-loop policy.
  The textbook definition of CEC used in the CE-MPC literature: at each stage, apply the control that would be optimal if the uncertain quantities were fixed at their expected values (Bertsekas, Dynamic Programming and Optimal Control, Ch. 6). That wording comes from a search summary. The textbook itself was not opened; section number unverified.

- Delayed observations by state augmentation: Katsikopoulos and Engelbrecht, "Markov decision processes with delays and asynchronous cost collection", IEEE Trans. Automatic Control 48(4) 568-574, 2003, DOI 10.1109/TAC.2003.809799.
  The repository page https://eprints.soton.ac.uk/415448/ returned HTTP 403, so the paper was not opened. Its content is known only from search-result summaries: an MDP with constant observation, action or cost delay reduces to a delay-free MDP on an augmented state (the last observed state plus the actions taken since).
  Tag: RELATED. Status: citation not verified from an opened page. This is the standard justification for putting the pending occupancy (last `delay` quarters of prescribing) in the planner's state.

## Known bounds on the CE versus optimal closed-loop gap

- https://arxiv.org/abs/2403.04559
  Messerer, Baumgärtner, Lucia, Diehl, "Fourth-order suboptimality of nominal model predictive control in the presence of uncertainty", arXiv 2403.04559 v2 (13 May 2024); the related DOI 10.1109/LCSYS.2024.3396611 indicates IEEE Control Systems Letters.
  Exact claim: for a nonlinear discrete-time stochastic system in a smooth, unconstrained setting, nominal (certainty-equivalent) MPC has suboptimality, against the stochastic optimal closed-loop controller, that grows at fourth order in the uncertainty level. The optimal and nominal inputs differ at second order. The relationship may cease to hold with state constraints. No code linked.
  Tag: RELATED. Applicability: the result needs smoothness in the inputs. A discrete drug choice does not satisfy this, so the bound does not transfer to this setting. At most it motivates a small-noise prediction to be tested.

- https://arxiv.org/abs/2602.02814
  Bozkurt, Mahajan, Nayyar, Ouyang, "Sub-optimality bounds for certainty equivalent policies in partially observed systems", arXiv 2602.02814 v1 (2 Feb 2026).
  Exact claim (abstract): for general non-linear partially observed stochastic systems, CE policies using any state estimate are not optimal in general. Upper bounds on their sub-optimality are derived when the cost and dynamics are smooth "in an appropriate sense". The form of the bound was not read, because only the abstract page was opened. No code linked.
  Tag: RELATED. This is the closest match to delayed or noisy observation. Whether its smoothness condition admits finite action sets: not determined.

- https://arxiv.org/abs/2412.10625
  Liu, Shi, De Schutter, "Certainty-Equivalence Model Predictive Control: Stability, Performance, and Beyond", arXiv 2412.10625 v3 (3 Feb 2026), to appear in IEEE TAC (DOI 10.1109/TAC.2026.3661789).
  Exact claim: suboptimality of CE-MPC under parametric (multiplicative) model mismatch, relative to the infinite-horizon optimal controller with perfect model knowledge, via relaxed dynamic programming. Includes a competitive-ratio bound for uncertain linear systems. No code linked.
  Tag: RELATED. This is model mismatch, not process noise, so it does not bound component (iii) here.

- A computable, instance-specific bound: the policy-gap inequality from information relaxation, 0 <= cost(CE policy) - optimal <= cost(CE policy) - dual bound. See inbox/2026-11-26-information-relaxation-dual-bounds.md. This is the only bound found that applies to discrete actions without smoothness assumptions.

## Reference implementation

No reference implementation of CE replanning for discrete actions with delayed observations was found. The three bound papers link no code. General MPC toolboxes were not searched or opened.

## Unsourced

- A closed-form or a priori bound on the CE gap for finite action sets under Gaussian process noise: not found.
- Verified full text for Bertsekas Ch. 6 (CEC) and Katsikopoulos and Engelbrecht 2003: not opened.
