# Prior art: Claim A (occupancy-coupled transition kernel)

Access date for every entry: 2026-11-16.

Claim under test: an MDP whose transition kernel depends on the agent's own action history through an occupancy measure, P(s' | s, a; mu_t), where mu_t = (1 - lam) * sum_k lam^k d_{t-k-delay} is an exponentially weighted, delayed average of the agent's past empirical action distributions. The coupling enters the kernel, not the objective.

Search status: cut short at the turn limit. See the end of this file.

## PRE-EMPTS

### Reinforcement Learning with History-Dependent Dynamic Contexts (ICML 2023)
- URL: https://arxiv.org/abs/2302.02061 (definitions read from https://ar5iv.labs.arxiv.org/html/2302.02061)
- Exact claim (from the text): in a Dynamic Contextual MDP the context x_h is drawn from P(x_h | tau_h), where tau_h is the full history of the agent's states, actions and contexts. The context then sets both the reward r(s_h, a_h, x_h) and the state transition P_h(s' | s_h, a_h, x_h). In the logistic DCMDP the context distribution is a softmax of sum_{t<h} alpha^{h-t-1} f_t(s_t, a_t, x_t), which is an exponentially discounted sum of features of the agent's own past state-action-context triples.
- Tag: PRE-EMPTS
- Reason: the state kernel depends, within an episode, on an exponentially weighted functional of the agent's own past actions, which is the structure of Claim A. Choosing f as a one-hot action feature makes the aggregate an exponentially discounted action count.
- What would still be novel if it stands:
  - an explicit delay (d_{t-k-delay}) with the pending occupancy in the observation;
  - a deterministic, normalised occupancy (a convex combination of empirical action distributions) entering the kernel directly, instead of through a sampled latent context;
  - results specific to that structure: delay effects and the interaction with entropy over resistance classes.
  The formulation by itself is a special case of the general DCMDP, or an MDP with an augmented state. A review will say so.

### Performative Reinforcement Learning in Gradually Shifting Environments (UAI 2024; arXiv v2 31 May 2024)
- URL: https://arxiv.org/abs/2402.09838 (model read from https://arxiv.org/html/2402.09838)
- Exact claim (from the text): the environment in round t is P_t = calP(pi_t, P_{t-1}, r_{t-1}), r_t = calR(pi_t, P_{t-1}, r_{t-1}). The dependence on the policy runs through its occupancy measure d. The paper calls this an environment that gradually adjusts to the deployed policy, and proposes the DRR and MDRR algorithms.
- Tag: PRE-EMPTS
- Reason: this is the 2024 performative-RL follow-up with gradual, history-dependent response. Unrolling the recursion makes the kernel a history-weighted functional of the occupancies of all previously deployed policies.
- What would still be novel if it stands:
  - coupling at the time-step level inside one trajectory, through realised empirical action distributions, rather than across deployment and retraining rounds;
  - an explicit delay parameter (the paper has none);
  - optimal control of the coupled process, where the agent plans over its own future influence, rather than performative stability;
  - the stewardship application.
- Code: https://github.com/rank-and-files/performative-rl-gradually-shifting-envs (link stated in the paper; not opened, not cloned).

## RELATED

- https://arxiv.org/abs/2609.06467 | Local and Global Stability in Performative RL (Sep 2026). Environment is a map of the current deployed policy, pi -> (P_pi, r_pi); stability under mixtures and Hedge. | RELATED | dependence is on the current policy only; no history or delay is stated in the abstract.
- https://arxiv.org/abs/2512.20576 | Performative Policy Gradient (Dec 2025). Proves performative versions of the performance difference lemma and the policy gradient theorem; PePG converges to performatively optimal policies. | RELATED | policy-induced shift, no history or delay is stated in the abstract.
- https://arxiv.org/abs/2510.04430 | Achieve Performatively Optimal Policy for Performative RL (Oct 2025). Zeroth-order Frank-Wolfe method reaches the performatively optimal policy. | RELATED | dependence is on the current policy.
- https://arxiv.org/abs/2411.05234 | Performative RL with Linear MDP (AISTATS 2025). Repeated retraining converges to stable policies in linear MDPs. | RELATED | dependence is on the current policy.
- https://arxiv.org/abs/2504.20593 | Independent Learning in Performative Markov Potential Games (AISTATS 2025). | RELATED | multi-agent; depends on the current policies.
- https://arxiv.org/abs/2505.05609 | On Corruption-Robustness in Performative RL (May 2025). | RELATED | robust retraining; no history is stated.
- https://arxiv.org/abs/2607.20152 | Active Inference as a Convex MDP (Jul 2026). Expected free energy minimisation is cast as a convex MDP; joint world-model and policy learning is framed as performative RL. | RELATED | occupancy enters the objective; no kernel coupling to policy history is stated in the abstract.
- https://arxiv.org/abs/2410.23251 (model read from https://arxiv.org/html/2410.23251) | Performative Control for Linear Dynamical Systems (NeurIPS 2024). A_t = A + Delta_t, with Delta_t ~ D_t(M) depending on the current policy parameter M; repeated SGD across rounds. | RELATED | the text states no lagged or stateful performative response.
- https://arxiv.org/abs/2204.08281 | Decision-Dependent Risk Minimization in Geometrically Decaying Dynamic Environments (AAAI 2022). The decision-dependent distribution evolves by geometric decay. | RELATED | the closest analogue of the exponential-lag response, but it is risk minimisation, not an MDP kernel. Should be cited as an ancestor.
- https://arxiv.org/abs/2011.03885 | Performative Prediction in a Stateful World (AISTATS 2022). Population response depends on the deployed classifier and the current population state. | RELATED | stateful, history-dependent response in supervised prediction, not an MDP.
- https://arxiv.org/abs/2112.07042 | How to Learn when Data Gradually Reacts to Your Model (Dec 2021). Stateful PerfGD. | RELATED | gradual response in prediction, not an MDP.
- https://arxiv.org/abs/2106.00589 | Improving Long-Term Metrics in Recommendation Systems using Short-Horizon RL (2021). Mentions policy-induced drift in user behaviour across sessions. | RELATED | the abstract gives no model of the drift.

## Seen only in search-result listings (not opened, not tagged)

- 2407.10207 Learning to Steer Markovian Agents under Model Uncertainty
- 2609.29007 When Does Action Credit Need Updating?
- 2610.13871 Partially Informed Stochastic Optimization via Zeroth-Order Residual Tracking
- 2610.01181, 2608.15079 (mean-field game on occupation measures; the listing says its kernel is independent of the population term)
- Remaining entries of the arXiv API "history-dependent" query listing (2611.05562, 2610.31617, 2605.14758, 2605.12410, 2603.00945, 2510.15728, 2510.03207, 2508.09971, 2501.12633, 2410.14001, 2408.17286, 2402.08406, 2311.09018, 2010.08030)

## Not searched before the turn limit

Run these before the claim is considered checked:
- the named neighbours themselves: Mandal et al. 2023 and Perdomo et al. 2020 abstracts; convex MDP / general utility (Zahavy et al. 2021, Zhang et al. 2020); non-stationary RL with a variation budget (Cheung et al.; Wei and Luo); LOLA and successors; mean-field control and McKean-Vlasov MDPs;
- bandits whose rewards depend on decayed action history (rebounding / satiation, linear bandits with memory, habituation-recovery bandits);
- the reworded queries "endogenous non-stationarity", "self-induced distribution shift" and "auto-induced distributional shift".
