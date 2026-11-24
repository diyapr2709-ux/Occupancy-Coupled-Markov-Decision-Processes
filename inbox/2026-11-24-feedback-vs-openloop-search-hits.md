# Search hits: feedback vs open-loop for drug scheduling or action-history-dependent MDPs, 2025-2026

Access date for all entries: 2026-11-24. Only abstracts were read unless stated otherwise.

Queries run:
- arXiv API: antibiotic AND (cycling OR stewardship) AND (RL OR optimal control OR feedback)
- arXiv API: (open-loop OR certainty equivalent) AND (closed-loop OR feedback) AND (delay OR action history OR hysteresis) AND (regime OR phase diagram)
- arXiv API: (antimicrobial, drug or antibiotic resistance) AND (RL OR MPC OR Markov decision)
- arXiv API: (performative OR action history OR history-dependent transition OR self-induced) AND (open-loop OR value of feedback OR closed-loop). This query returned 4,620 results; only the first page was screened.
- Web: when feedback beats open-loop drug cycling, adaptive therapy, phase diagram, 2025-2026
- Web: value of feedback, closed-loop vs open-loop, MDP transitions depending on past actions, 2025-2026

---

## abx_amr_simulator

- URL: https://arxiv.org/abs/2603.11369 (HTML full text read: https://arxiv.org/html/2603.11369)
- Bibliographic: Lee, Blumberg. arXiv 2603.11369, submitted 2026-03-11, cs.LG.
- Exact claims (verbatim fragments):
  - "a Python-based simulation package designed to model antibiotic prescribing and AMR dynamics"
  - "antibiotic resistance dynamics modeled via a leaky-balloon abstraction"
  - "Resistance pressure increases with prescribing frequency and decays over time in the absence of selection pressure."
  - "tools to explore partial observability through noise, bias, and delay in observations"
- Tag: RELATED
- Reason: a Gymnasium RL environment in which resistance is a decaying accumulator of past prescribing, which is close to occupancy coupling. The paper reports no baseline policies, no results and no feedback-vs-open-loop characterisation. Its delay applies to observations, not to the resistance dynamics, and no process noise in the resistance dynamics is described. Code is listed at https://github.com/jl56923/abx_amr_simulator but was not cloned or reviewed.

## Adaptive chemotherapy control under tumor heterogeneity via RL

- URL: https://arxiv.org/abs/2609.12264
- Bibliographic: Kidane, Motayed, Wang. arXiv 2609.12264, submitted 2026-09-10.
- Exact claims (verbatim fragments):
  - "The DRL policies are benchmarked against a Pontryagin's Maximum Principle (PMP)-derived open-loop benchmark."
  - "simulation-trained DRL can learn state-dependent feedback dosing policies"
- Tag: RELATED
- Reason: closed-loop RL is compared against an open-loop optimal benchmark, but for dosing, not drug choice. There is no regime map over delay, memory or noise, and full observation is assumed.

## Conditional success of adaptive therapy

- URL: https://arxiv.org/abs/2502.09392
- Bibliographic: Sun et al. arXiv 2502.09392, v1 2025-02-13, v3 2025-07-12, q-bio.
- Exact claims (verbatim fragments):
  - "adaptive therapy underperforms regardless of threshold"
  - "efficacy requires threshold optimization"
  - "adaptive therapy consistently outperforms alternatives"
  - success "depends critically on the threshold for pausing and resuming treatment."
- Tag: RELATED
- Reason: a regime classification of when threshold-feedback (adaptive) therapy beats MTD and intermittent therapy in a Lotka-Volterra tumour model. Its axes are threshold, competition, burden and resistant fraction, not delay, occupancy memory or process noise.

## RL with multi-step transition predictions: a Bellman theory

- URL: https://arxiv.org/abs/2610.32887
- Bibliographic: Lu, Chen, Li, Wu, Wierman. arXiv 2610.32887, submitted 2026-10-23, cs.LG.
- Exact claim (verbatim fragment): "predictive information creates value by recursively changing the local order of maximization and expectation"
- Tag: RELATED
- Reason: it quantifies the value of within-block feedback given K-step predictions in standard finite MDPs. Transitions do not depend on the agent's own past actions.

## MDP modeling for multi-stage stochastic programs

- URL: https://arxiv.org/abs/2509.22981
- Bibliographic: Morton, Dowson, Pagnoncelli. arXiv 2509.22981, v1 2025-09-26, v2 2026-04-07.
- Exact claim (verbatim fragment): "We extend policy graphs to include decision-dependent uncertainty for one-step transition probabilities"
- Tag: RELATED
- Reason: decision-dependent one-step transitions, with no delayed or occupancy-memory dependence and no feedback-vs-open-loop regime analysis.
