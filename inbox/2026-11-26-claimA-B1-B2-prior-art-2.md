# Second-pass prior-art search: Claim A, Claim B1 and the B2 gate

Access date for every item: 2026-11-26. Gatherer: lit-scout. This pass extends
inbox/2026-11-26-claimA-feedback-boundary-search.md and
inbox/2026-11-26-claimB1-redundant-action-entropy-search.md and does not refile
their items.

Claim A searched: in MDPs whose kernel depends on an exponentially weighted occupancy of
the agent's own past actions, with a reporting delay and process noise, the value of
state feedback over the best committed schedule (planned-once schedule minus myopic
greedy) has a sign-change boundary in (delay, memory, noise), derived before measurement
and tested at held-out cells against a theory-free null.

Claim B1 searched: with exactly interchangeable actions and tied networks, the gap
between action-entropy SAC and class-entropy SAC at fixed temperatures equals the gap
between the mean-field soft-optimal policies.

B2 gate searched: similarity-sensitive, Vendi or kernel entropy used as a policy
regulariser.

## PRE-EMPTS

- https://www.sciltp.com/journals/aieng/articles/2610005408 (Equivalent Optima, Different Policies: Certified Distribution-Preserving Reformulations for AI Decision Systems, AI Engineering 2(2) art. 20, 21 Oct 2026, DOI 10.53941/aieng.2026.100020; abstract page only, full text not opened). Found as a citer of Baram et al. 2021 (MinRed).
  Claim: for finite log-linear policies under a relative-entropy objective with deterministic decoding, a reformulation preserves the decoded law if and only if the weighted residual-feature spectra match across decoder fibres. When the spectrum has several atoms, the reformulation adds a nonnegative conditional-divergence penalty. Penalties add across stages, and an exact correction recovers the physical objective. The only test is a six-period scheduling study (median physical-policy total variation of 0.245 with untied auxiliary bits). There are no RL agents.
  Tag: PRE-EMPTS (partial; B1 theory component only). Reason: decoder fibres are action classes, so this already derives, in a static log-linear setting, that redundant representations shift the entropy-regularised optimum by a conditional-divergence term, which is the mean-field side of B1. What remains novel if it stands: that the measured performance gap between two learned SAC agents (sequential, function approximation, tied networks, fixed temperatures) equals that predicted gap, and the empirical test of that equality. The derivation itself should be cited, not claimed.

## BASELINE

- https://api.openalex.org/works/doi:10.1109%2FLRA.2025.3572822 (Maximum Next-State Entropy for Efficient Reinforcement Learning, IEEE RA-L 10(7):6896-6903, 2025, DOI 10.1109/lra.2025.3572822; abstract reconstructed from the OpenAlex record, full text not open access)
  Claim: policy-entropy regularisation does not directly promote next-state diversity. The abstract says it "theoretically reveal[s]" the gap between maximum next-state entropy and policy-entropy regularisation. MNSE adds an action-mapping layer after the inner policy and maximises next-state entropy by optimising inner-policy entropy. Gains are reported on continuous control.
  Tag: BASELINE. Reason: it is a direct alternative to class-entropy SAC and a MinRed follow-up. The abstract quantifies no return gap. The record mentions no code and gives no tuned configuration. Its stated "gap" theory must be read in full before B1's theory is claimed as new.

## RELATED

Claim A, priority 1:
- https://arxiv.org/abs/2306.01243 (Chen, Meng, Bai, Ye, Poor, Wang, Efficient RL with Impaired Observability: Delayed and Missing State Observations, 2023; full text read via https://ar5iv.labs.arxiv.org/html/2306.01243, first 100k characters)
  Claim: Prop. 4.6 constructs a constant-delay MDP whose optimal executable policy has zero gap to the no-delay optimum at delay d and a gap of at least 1/2 at delay d+1. Regret is of order (H∧D)^{5/2}. Planning is NP-hard. Thm 5.2 gives a missing-rate condition, with λ0 = 1-1/A conjectured to be critical.
  Tag: RELATED. Reason: the sharp threshold is a single constructed instance, measured against full observability. It is not a derived boundary over a family, not a comparison with committed schedules, and it has no occupancy kernel, memory axis or noise axis. It does show that a delay threshold can be arbitrarily sharp, which is relevant to cell spacing in a held-out test.
- https://ideas.repec.org/a/pal/jorsoc/v50y1999i6d10.1057_palgrave.jors.2600745.html (Bander and White, MDPs with noise-corrupted and delayed state observations, JORS 50(6):660-668, 1999; abstract only)
  Claim: better observation quality or shorter delay never decreases the optimal expected discounted reward. The paper defines the marginal value of reducing the delay from k to k-1, and as k goes to infinity the problem becomes the completely unobserved case.
  Tag: RELATED. Reason: it gives no threshold and no committed-schedule comparator. Its monotonicity result means optimal feedback minus optimal open-loop is never negative, so Claim A's sign change can only exist because it compares two suboptimal policies (planned-once schedule against myopic greedy). The claim's wording must make that explicit.

Claim A, priority 2 (stewardship, 2023-2026; abstracts via Europe PMC REST records):
- https://doi.org/10.64898/2026.01.18.700135 (Shuaibu et al., graph-theoretic pathogen-specific antibiotic cycling, bioRxiv 2026)
  Claim: pre-optimised collateral-sensitivity "trap loops" are compared in stochastic simulations, with clinical error and biological noise, against reactive antibiogram-driven care. The fixed loops suppress resistance.
  Tag: RELATED. Reason: it is the closest stewardship analogue of a committed schedule against greedy feedback, but it has no reporting delay, no occupancy memory and no derived boundary.
- https://doi.org/10.1016/j.idm.2025.04.001 (Anderson et al., Invariant set theory for predicting potential failure of antibiotic cycling, Infect. Dis. Model. 10(3), 2025, PMID 40297503)
  Claim: switched-systems and set-control theory define therapeutic windows for sequential therapy in P. aeruginosa.
  Tag: RELATED. Reason: it gives failure conditions for cycling, but no feedback-versus-schedule comparison, no delay and no noise.
- https://doi.org/10.1038/s44259-025-00160-w (Anderson et al., Computational framework for sequential antibiotic therapy, npj AMR 2025, PMID 41184519)
  Claim: a collateral-sensitivity framework and tool that delineates conditions under which sequential therapy fails.
  Tag: RELATED. Reason: it has no feedback arm, no delay and no memory.
- https://doi.org/10.3390/antibiotics14121266 (Diebner et al., Antibiotics 14:1266, 2025, PMID 41463768; one abstract line only)
  Claim: a differential-equation model of the resistant share driven by time-varying consumption, with time-delayed cross-correlations, in a German ICU.
  Tag: RELATED. Reason: it shows empirical lag between consumption and resistance, but has no policy comparison.

Claim A, priority 3 (recovering and recharging bandits; abstracts via the arXiv API):
- http://arxiv.org/abs/2610.19179 (Palma Foster and Tardos, Strategic Queuing with Geometric Recharge, 2026)
  Claim: the setting is equivalent to recharging bandits with geometric recharge. Any no-regret learner is at least 0.77-competitive (tight).
  Tag: RELATED. Reason: geometric recharge is the discrete form of exponential recovery, but there is no delay, no noise boundary and no committed-schedule comparison.
- http://arxiv.org/abs/2403.11520 (Tanimoto and Fukumizu, State-Separated SARSA with recovering rewards, 2024)
  Claim: SARSA over round-indexed states for rewards that depend on time since the last pull, with asymptotic convergence.
  Tag: RELATED. Reason: its memory is time since the last pull, not exponential occupancy.
- http://arxiv.org/abs/1910.14354 (Recovering Bandits, NeurIPS 2019)
  Claim: Gaussian-process model of reward as an unknown function of time since the last play.
  Tag: RELATED. Reason: a different memory form, with no feedback-value analysis.
- http://arxiv.org/abs/0711.3861 (Guha, Munagala, Shi, Approximation Algorithms for Restless Bandit Problems)
  Claim: Feedback MAB admits no greedy index-optimal policy. A 2+ε greedy approximation is given.
  Tag: RELATED. Reason: it bounds greedy's suboptimality, not a boundary in delay, memory or noise.
- http://arxiv.org/abs/0811.0637, http://arxiv.org/abs/0810.4658, http://arxiv.org/abs/0802.1379, http://arxiv.org/abs/1305.6993, http://arxiv.org/abs/1309.3692 (myopic-optimality line for restless bandits and opportunistic access)
  Claim: closed-form sufficient conditions (stochastically identical, positively correlated Gilbert-Elliott channels; multi-state extensions; imperfect sensing) under which the myopic policy is optimal.
  Tag: RELATED. Reason: these are derived regions where greedy is optimal. Claim A's boundary must be shown not to be a corollary of them.
- http://arxiv.org/abs/2105.10625 (Combinatorial Blocking Bandits with Stochastic Delays), http://arxiv.org/abs/2110.11819 (Last Switch Dependent satiation and seasonality)
  Claim: greedy approximation analysis under blocking; last-switch-dependent payoffs.
  Tag: RELATED. Reason: different memory forms.

Claim B1, priority 4:
- http://arxiv.org/abs/2501.14543 (Reducing Action Space for Deep RL via Causal Effect Estimation, 2025)
  Claim: an inverse-dynamics causal-effect estimate suppresses redundant actions during exploration.
  Tag: RELATED. Reason: it has no entropy regulariser. It is in the same family as NPM (filed as BASELINE in the first pass). Code not checked.
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2102.11329/citations (the 17 citers of MinRed, titles only)
  Claim: no citer other than the two above is about entropy over action classes or soft-optimal policies under redundancy.
  Tag: RELATED. Reason: a search record of the MinRed follow-ups.

B2 gate, priority 5:
- http://arxiv.org/abs/2509.02930 (VendiRL, NeurIPS 2025)
  Claim: Vendi-score diversity over a set of skills, under user-chosen similarity, as a self-supervised objective.
  Tag: RELATED. Reason: it is the closest hit, but it measures diversity across skills, not similarity-sensitive entropy of one policy's action distribution as a SAC-style regulariser.
- http://arxiv.org/abs/2610.36425 (RePPO, 2026) and http://arxiv.org/abs/2609.14896 (MoDA, 2026)
  Claim: population-level and mode-level diversity; E-Vendi used as an evaluation metric.
  Tag: RELATED. Reason: neither regularises a single policy.

## Not opened or blocked

- OpenReview forum and PDF pages return a browser-verification screen to the fetcher. https://openreview.net/notes/edits/attachment?id=oJFycAAbuu&name=pdf (file metadata names "policy_entropy_driven") could not be read and is not assessed.
- The full texts of the Bander and White paper and of MNSE were not available. The Temporal Vendi Score paper (ICLR 2026) was seen only as a search snippet.
