# Open-loop vs certainty-equivalent vs optimal control under action-dependent observations

- URL: https://arxiv.org/abs/2610.33412 (Atom record: http://export.arxiv.org/api/query?id_list=2610.33412)
- Access date: 2026-11-24
- Source read: title, authors, categories and abstract only. Full text not read.
- Bibliographic: "Performance Gaps Between Open-Loop, Certainty-Equivalent, and Optimal Control Under Action-Dependent Observations", arXiv 2610.33412v1, submitted 2026-10-24, cs.LG and eess.SY.

## Exact claims (verbatim abstract fragments, as extracted by the fetch tool)

- Setting: "We study this question in a linear-quadratic system with bilinear observations"
- Regime parameter: "We identify an instance-dependent information-strength parameter $q$"
- Result: "establishing an $\Omega(T)$ gap between CE and optimal control."

Abstract content that the fetch tool paraphrased rather than quoted: actions influence both the state and the information available later. Three regimes in q: for small q, no policy beats open-loop by more than O(1) cumulative cost. For large q, CE has an Omega(T) advantage over open-loop. For intermediate q, a policy that occasionally probes for information has an Omega(T) advantage over open-loop while CE gets O(1). The analysis uses a regret decomposition that weighs the immediate cost of gathering information against its future benefit.

## Tag: RELATED

Reason: it decomposes open-loop, CE and optimal control with a regime map, but in linear-quadratic dynamics with bilinear (action-dependent) observations. There the gap comes from information acquisition (dual control or probing). It does not cover a transition kernel that drifts because of a delayed, exponentially weighted occupancy of the agent's own past actions, and it does not cover process noise under full observation.

What differs from Claim A, precisely:
- Mechanism: in this paper, actions change what is observed. In Claim A, actions change the future kernel through delayed occupancy, and the state is fully observed.
- Source of the feedback advantage: here, information value under partial observation. In Claim A, process noise with full observation, so feedback can only win through reacting to realised noise, not through learning.
- Axes of the regime map: here, a single information-strength parameter q. In Claim A, delay, occupancy memory and process-noise scale.
- Dynamics: here, linear dynamics with quadratic cost. In Claim A, a nonlinear selection kernel over resistance with action-history dependence.

Why it is still the closest methodological analogue: its open-loop / CE / optimal decomposition with a derived regime boundary is the same analytical template Claim A uses. Claim A cannot present the decomposition-plus-regime-map framing as new. Its novelty has to rest on the mechanism: self-induced, delayed kernel drift under process noise with full observation. Caveat: only the abstract was read. The full text was not checked for an extension to action-dependent dynamics.
