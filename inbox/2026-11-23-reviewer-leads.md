# Reviewer leads, filed 2026-11-23

Access date for every entry: 2026-11-23. "Opened" means the abstract page (and, where stated, full text or code) was fetched. Entries seen only in a search snippet or an arXiv API feed are marked as such.

Note: the earlier passes (inbox/2026-11-16-prior-art-*.md) could not be located from this session (no directory listing available; guessed names did not exist), so overlap with them was not checked.

## 1. abx_amr_simulator (Lee and Blumberg, arXiv 2603.11369, March 2026; SoftwareX 35, 102872)

- URL: https://arxiv.org/abs/2603.11369 ; full text https://arxiv.org/html/2603.11369v1
- Code: https://github.com/jl56923/abx_amr_simulator (archive DOI 10.5281/zenodo.18969700). LICENSE file: BSD 3-Clause, copyright 2026 (https://raw.githubusercontent.com/jl56923/abx_amr_simulator/main/LICENSE). The README licence section still holds an unfilled template line, so the README and the LICENSE file disagree.
- Exact claim: a Python, Gymnasium-compatible simulation package for antibiotic prescribing under antimicrobial resistance, with user-defined patient populations, antibiotic-specific resistance response curves, reward functions trading immediate clinical benefit against long-term resistance, a "leaky balloon" resistance model, and partial observability (noisy, biased or delayed observations); presented as "a configurable benchmark environment" for sequential decision-making.
- Resistance update, read from source (https://raw.githubusercontent.com/jl56923/abx_amr_simulator/main/src/abx_amr_simulator/core/leaky_balloon.py and .../core/abx_amr_env.py):
  - per antibiotic, a scalar latent pressure: `pressure += doses; pressure = max(0, pressure - leak)`. The leak is a constant subtraction, not proportional decay, so pressure is a decayed accumulator of the agent's own prescribing but not an exponentially weighted occupancy;
  - observable resistance = residual + range * (2*sigmoid(pressure/flatness) - 1);
  - cross-resistance: configurable matrix; effective doses for a target drug = sum over prescribed drugs of count * crossresistance_matrix[prescriber][target];
  - delay: none in the dynamics. The balloon is stepped with the current step's doses immediately; there is no prescribing queue. Delay exists only in observation (visible resistance refreshed every n steps, default 1);
  - observation: patient features plus visible resistance per drug (optionally steps since update); past prescribing is not in the observation;
  - action: MultiDiscrete, one antibiotic or no treatment per patient, default 10 patients per step.
- RL results: none reported in the paper or README (PPO, hierarchical PPO and Optuna tuning are supported but no outcomes are given).
- Tag: PRE-EMPTS
- Reason: it pre-empts any claim to be the first (Gymnasium) benchmark for antibiotic prescribing under resistance with own-prescribing-driven dynamics and configurable cross-resistance; it does not pre-empt the delayed, exponentially weighted occupancy kernel.
- What it does: a released, licensed Gymnasium environment where the agent's own prescribing, coupled through a cross-resistance matrix, raises a leaky accumulator that drives resistance.
- What would remain novel if it stands: (a) formulation: a kernel driven by an exponentially weighted occupancy with an explicit prescribing delay, observation carrying the pending occupancy, and theory for this class (suboptimality of deterministic stationary policies); entropy over mechanism classes; (b) benchmark: only "an instance family with a planner, heuristics and frozen seed tiers, with delay and exponential memory", not "the first benchmark". It has no RL numbers, so it is not a reproduction target for rung 1.

## 2. Rebounding Bandits for Modeling Satiation Effects (Leqi, Kilinc-Karzan, Lipton, Montgomery; NeurIPS 2021; arXiv 2011.06741)

- URL: https://arxiv.org/abs/2011.06741 ; proceedings https://proceedings.neurips.cc/paper/2021/hash/2109737282d2c2de4fc5534be26c9bb6-Abstract.html (NeurIPS 2021 venue confirmed there)
- Exact claim: a multi-armed bandit where satiation follows time-invariant linear dynamics; each arm's expected reward declines with consecutive pulls and rebounds toward its initial value when not pulled; greedy (cyclic) play is optimal for identical deterministic dynamics (the abstract page links this case to Max K-Cut per search snippet); Explore-Estimate-Plan for unknown stochastic dynamics.
- Reward form b_k - lambda_k * s_{k,t} with retention factor gamma_k in [0,1): seen in a search snippet only; the exact satiation recursion was not opened (PDF could not be parsed).
- Code: no link on the proceedings page.
- Tag: PRE-EMPTS (formulation component only)
- Reason: an environment whose state is a geometrically decayed record of the agent's own past actions, driving its payoffs, with cyclic schedules beating repeated play, already exists in published form.
- What it does: per-arm, uncoupled, undelayed decayed-exposure state driving rewards; bandit, no further state.
- What would remain novel: a delay between action and effect; coupling of the decayed occupancy across actions through a cross-resistance matrix onto mechanism classes; a persistent resistance state whose kernel the occupancy drives (rather than reward being a direct function of occupancy); deep RL with entropy over mechanism classes. Nothing about the entropy claim is touched.

## 3. Linear Bandits with Memory: from Rotting to Rising (Clerici, Laforgue, Cesa-Bianchi; arXiv 2302.08345)

- URL: https://arxiv.org/abs/2302.08345 ; full text https://arxiv.org/html/2302.08345
- Exact claim: a nonstationary linear bandit in which rewards depend on the learner's past actions within a window of size m; memory matrix A_{t-1} = (A_0 + sum_{s=1}^m a_{t-s} a_{t-s}^T)^gamma and expected reward <a_t, A_{t-1} theta*>; gamma < 0 rotting, gamma > 0 rising; an OFUL variant competing with cyclic policies with regret ~ sqrt(d) (m+1)^{1/2+max(gamma,0)} T^{3/4}; model selection for unknown m, gamma. The paper notes the optimal policy may not be cyclic and that repeatedly playing theta* may not be optimal in the rotting case.
- Code: none mentioned in the text read.
- Tag: PRE-EMPTS (formulation component only)
- Reason: own-action memory coupled across correlated actions through action features (outer products; their example: a pop-rock song raises rock satiation less than pure rock) is the bandit analogue of occupancy pushed through a drug-to-mechanism embedding.
- What it does: a fixed window (equal weights, no exponential decay, no delay), reward directly a function of memory, bandit.
- What would remain novel: exponential memory with delay; an MDP with a resistance state; the entropy-over-mechanisms exploration objective and its comparison with action entropy.

## 4. Generalized Linear Bandits with Memory (Ann, Choi, Hwang, Shin, Cheong, Oh; arXiv 2608.15848; ICML 2026 per abstract page)

- URL: https://arxiv.org/abs/2608.15848
- Exact claim: generalised linear bandits where rewards depend on past actions through a finite memory matrix; sharper analysis gives O~(sqrt T) for the linear case, versus the earlier O~(T^{3/4}); block-wise algorithm with shrunken confidence bounds for GLMs.
- Tag: RELATED
- Reason: same problem class as entry 3 with a nonlinear link; adds nothing toward delay, a resistance state or entropy.

## 5. Decoupling regularization from the action space (Mohammadpour, Frejinger, Bacon; ICLR 2024; arXiv 2406.05953)

- URL: https://arxiv.org/abs/2406.05953 ; full text https://arxiv.org/html/2406.05953v1
- Code: https://github.com/SobhanMP/decoupled-soft-RL (link from the full text; licence not checked)
- Exact claim: entropy-regularised RL is over-regularised as the number of actions grows, and inconsistently regularised when action sets are state-dependent. Static rule: tau' = tau / log|A(s)|. Dynamic rule: temperature tuned by SAC's dual loss with target H(pi(.|s)) >= alpha H(U) + (1 - alpha) H(V), U uniform, V minimum-entropy policy; recommended alpha about 0.77 for actions in [-1, 1]. Evaluated on a hypergrid toy, DeepMind Control and a drug-design MDP.
- Not addressed: equivalent or redundant actions, clustering or metrics over actions.
- Tag: BASELINE
- Reason: for discrete actions H(V) = 0, so the dynamic rule gives target alpha*log|A(s)|, which is the project's c*log(K) for SAC; it is published, has code, and is the strongest honest form of the action-entropy comparator, including the per-state log|A(s)| version under action masks.
- Note: per-agent scaling of the target with log of the support size is prior work and cannot be claimed; the mechanism-class pushforward is not in this paper.

## 6. Maximum Entropy RL (Provably) Solves Some Robust RL Problems (Eysenbach, Levine; ICLR 2022; arXiv 2103.06257)

- URL: https://arxiv.org/abs/2103.06257
- Exact claim: MaxEnt RL optimises a lower bound on a robust RL objective, so it is robust to some perturbations of the dynamics and reward; the authors say it is the first rigorous proof of this.
- Tag: RELATED
- Reason: no action-history dynamics or entropy over action classes; relevant as an alternative explanation for any robustness gain from entropy under resistance noise.

## Corrections (validator check of the sources, 2026-11-24)

- 2011.06741: the link between rebounding bandits and Max K-Cut is in the paper's body, not its abstract.
- 2302.08345: the full text's Appendix C appears to link an anonymous code repository. This was seen through a fetch summary and not opened. The d versus sqrt(d) factor in its regret bound was not checked.
