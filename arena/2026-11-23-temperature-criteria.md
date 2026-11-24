# Arena: the temperature protocol for comparing agents whose entropies live in different spaces

Fixed before any proposal is written. The judge scores against these and nothing else.

## The decision

What replaces the protocol's entropy-target rule (CLAUDE.md, Protocol: H_target_OC = c log M, H_target_SAC = c log K, one c fixed before test runs) for every comparison between OC-SAC (entropy of the class marginal) and SAC (entropy over drugs), and for any further entropy-regularised comparator. State the rule concretely enough to implement in ocmdp/agents/ (SACConfig, losses.entropy_target, the temperature loss), how its constants are chosen, on which seed tier, and with what budget per agent.

## What it affects downstream

Every agent comparison and its preregistration. The current rule makes SAC's temperature inert by construction, so every logged comparison so far sets a regulated temperature against a collapsing one.

## Facts the proposals must take as given

- With exactly interchangeable drugs and tied networks, H(pi) = H(q) + sum_j q_j log n_j, so SAC's entropy cannot fall below about log n_min; SAC's temperature can only rise if c > log n_min / log K (verified numerically; decisions/2026-11-23.md). On the default instance (two classes of two drugs) that is never for c <= 0.5.
- Logged (runs/train/train_20261123_190717_20498.json, train_20261123_190750_20538.json): SAC's temperature sits at the clamp floor at every evaluation state; OC-SAC's median evaluation temperature is at the floor too, while its training-window mean stays high on some seeds.
- Published rules to weigh, among others you find: the decoupled rule of Mohammadpour, Frejinger and Bacon (ICLR 2024, arXiv 2406.05953; inbox/2026-11-23-reviewer-leads.md); the discrete-SAC target 0.98 log|A| (Christodoulou 2019) and CleanRL's 0.89 log|A|; fixed temperatures (SD-SAC, arXiv 2209.10081); per-agent tuning at parity.
- The optimal policies of this model are near-deterministic (the planner's best schedules are rotations), so entropy is at most a learning aid here unless a robustness argument says otherwise.

## Criteria, in priority order

1. **Parity.** Neither agent's temperature mechanism is inert or saturated by construction, on equal and on unequal classes, and each agent gets the same tuning budget on the same seeds and episodes.
2. **Isolation.** The comparison differs only in the space the entropy is measured in; anything else that differs (reachable range, effective bonus) is named.
3. **Fixed in advance.** The rule and its constants are chosen by a procedure stated now, on dev or val seeds, never after seeing a test-tier outcome.
4. **Precedent.** Published support for the rule, or a stated reason it has none.
5. **Generality.** Works for unequal classes and for partially overlapping cross-resistance, where classes are not crisp.
6. **Implementable** in ocmdp/agents/ and experiments/ without touching ocmdp/eval/, ocmdp/envs/ or ocmdp/core/.

## Hard constraints

- CLAUDE.md is the owner's to edit; the proposal states the replacement text for its Protocol bullet, and the code change.
- No rule chosen because it makes one agent win.

## What would settle parts of it empirically instead

Whether a proposed rule keeps both temperatures off the clamp can be checked with one dev launch (both agents, five seeds, temperatures logged). Name that check if the proposal depends on it.

## Material

CLAUDE.md; ocmdp/agents/oc_sac.py, ocmdp/agents/losses.py, ocmdp/objectives/pushforward_entropy.py; experiments/train_agent.py; decisions/2026-11-23.md; inbox/2026-11-16-discrete-sac.md, inbox/2026-11-16-mechanism-level-sac.md, inbox/2026-11-23-reviewer-leads.md; the two runs/train reports named above.
