# Claim A prior-art search: value of feedback in occupancy-coupled MDPs

Access date for every item: 2026-11-26. Gatherer: lit-scout.

Claim searched: in MDPs whose transition kernel depends on an exponentially weighted occupancy of the agent's own past actions, with a reporting delay and process noise, the value of state feedback over the best committed schedule (planned-once schedule minus myopic greedy) has a sign-change boundary in (delay, memory, noise), derived before measurement and tested at held-out cells against a theory-free null.

No PRE-EMPTS hit among the items opened. None of them derives or tests a boundary in (delay, memory, noise) for the value of feedback over a committed schedule.

## Items opened

- https://pmc.ncbi.nlm.nih.gov/articles/PMC11032439/ (Weaver, King, Maltas, Scott, PNAS 2024, 121(16) e2303165121)
  Claim: RL drug-cycling agents on empirical E. coli fitness landscapes (16 genotypes, 15 beta-lactams) beat random selection and two-drug fixed cycles, and approach a backward-induction MDP optimum. Tested with noisy fitness observations, and with observation delay for the genotype agent only.
  Tag: BASELINE. Reason: it is the closest RL cycling result with noise and delay arms, but it reports per-condition performance and no value-of-feedback curve or boundary, and its resistance has no occupancy memory. Code: https://github.com/DavisWeaver/evo_dm and https://github.com/DavisWeaver/rl_cycling. It may already be filed under prior-art-antibiotic-rl.
- https://arxiv.org/abs/2603.11369 (abx_amr_simulator, SoftwareX 35, 2026)
  Claim: a Gymnasium antibiotic-prescribing simulator in which resistance is a leaky accumulator of prescribing (a "leaky balloon" with a per-drug leak rate), with configurable noisy, biased and delayed observations.
  Tag: RELATED. Reason: it is the nearest environment to the occupancy-plus-delay setup, but the paper reports no experiments and no feedback-versus-schedule analysis. Code: https://github.com/jl56923/abx_amr_simulator.
- https://arxiv.org/abs/2604.01434 (Leveraging the Value of Information in POMDP Planning, 2026)
  Claim: VOI-gated POMDP planning (VOIMCP) with near-optimality bounds.
  Tag: RELATED. Reason: VOI is used as a planning device, with no delay, memory or noise boundary.
- https://arxiv.org/abs/2608.18387 (Adaptive therapy under parametric, structural, and measurement uncertainty, 2026)
  Claim: biomarker-feedback adaptive therapy (AT50) robustly improves time to progression under noisy measurements, and may be disadvantageous under a metastasis-risk metric.
  Tag: RELATED. Reason: it compares feedback with continuous therapy, but no boundary in delay, memory or noise is derived.
- https://arxiv.org/abs/2405.18100 (A Pontryagin Perspective on Reinforcement Learning, 2024/2025)
  Claim: open-loop RL that learns a fixed action sequence via Pontryagin's principle.
  Tag: RELATED. Reason: it offers a way to learn committed schedules but does not compare their value with feedback across noise or delay. No code linked.
- https://arxiv.org/abs/2302.02061 (RL with History-Dependent Dynamic Contexts, ICML 2023)
  Claim: dynamic contextual MDPs whose context transitions depend on aggregated features of the history, with regret bounds.
  Tag: RELATED. Reason: it is a history-aggregated kernel, but the abstract does not say the aggregation is exponential, and the paper has no open-loop versus feedback analysis.
- https://arxiv.org/abs/2411.16362 (Optimal switching strategies in multi-drug therapies for chronic diseases, 2024)
  Claim: switches are modelled as stochastic resets and optimal switching intervals are derived.
  Tag: RELATED. Reason: it has no feedback-versus-schedule comparison and no memory or delay.
- https://arxiv.org/abs/2611.16617 (Optimal One Health control of antimicrobial resistance, 2026)
  Claim: open-loop optimal control of a household mean-field AMR model, where "community feedback" is an epidemiological coupling.
  Tag: RELATED. Reason: it does not study state feedback, delay or memory.
- https://arxiv.org/abs/2609.06467 (Local and Global Stability in Performative Reinforcement Learning, 2026)
  Claim: stability of performative RL, where the deployed policy shapes later environments.
  Tag: RELATED. Reason: the policy dependence runs across deployment rounds, not within the episode through action occupancy.
- http://arxiv.org/abs/2011.06741 (Rebounding Bandits for Modeling Satiation Effects, NeurIPS 2021), abstract via the arXiv API
  Claim: arm rewards follow linear satiation dynamics driven by past pulls with stochastic noise; greedy is optimal under identical deterministic dynamics; Explore-Estimate-Plan.
  Tag: BASELINE. Reason: it is a memory-bandit planner, and its greedy-optimality condition is directly relevant to where planned-once minus greedy is zero. Code not checked.
- http://arxiv.org/abs/2302.08345 (Linear Bandits with Memory: from Rotting to Rising, TMLR 2024 / ICLR 2025 poster), abstract via the arXiv API
  Claim: rewards depend on a fixed window of past actions; regret is measured against cyclic policies.
  Tag: BASELINE. Reason: memory-bandit planner (window memory, not exponential). It is probably already filed under action-memory-bandits-one-hop. Code not checked.
- http://arxiv.org/abs/2306.00338 (Last Switch Dependent Bandits with Monotone Payoff Functions, 2023), abstract via the arXiv API
  Claim: constant-factor planning approximation when payoffs depend on time since the last switch.
  Tag: RELATED. Reason: a different memory form, and no feedback-value analysis.
