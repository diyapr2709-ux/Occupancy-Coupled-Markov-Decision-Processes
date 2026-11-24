# Proposal 2

## Proposal: set each agent's temperature target as a fraction of the entropy range its own policy class can reach, use one shared fraction chosen on val by minimax regret, and make the clamp reflect instead of absorb

The decision is partly empirical. Whether the temperatures actually stay off the clamp during training can only be settled by the dev launch in section 4. The normalisation itself is a design choice, and I propose it below.

### 1. The rule

**Targets.** Each agent i gets the target H_target_i = H_min_i + c · (H_max_i − H_min_i).
- H_min_i and H_max_i are the least and greatest entropy that agent's actor can express, measured in its own entropy space.
- They are computed once, when the agent is built, from the actions the actor cannot tell apart. The scorer is a function of (state, embedding), so drugs with identical embedding rows always get equal probability.
- Reachable ranges this gives:
  - OC-SAC on crisp classes: [0, log M].
  - SAC with tied networks: [log n_min, log K].
  - SAC with no tied drugs: [0, log K], which is exactly the published c·log|A| rule.
  - OC-SAC under partial cross-resistance (assignment = row-normalised C): [min over drugs of the entropy of that drug's assignment row, max over p of H(pS)]. Its H_min is above 0 here, so the current c·log M rule can make OC-SAC's temperature inert in the same way it makes SAC's inert now.

**Choosing c.**
- Grid, fixed now: c ∈ {0.05, 0.15, 0.3, 0.5, 0.77, 0.89}. The grid spans the old flat value, Mohammadpour et al.'s recommended 0.77 and CleanRL's 0.89. 0.98 is left out because CleanRL's docs report it unstable beyond 100k steps.
- A reference arm at c = 0 is run as well. Its temperature is inert by design, so it can never be selected.
- Seed tier: val seeds 100–109, the same 10 for both agents at every grid point.
- Evaluation: the same 20 val evaluation episodes everywhere.
- Fixed and shared, not tuned: 12,000 training quarters, alpha_init, the log-alpha clamp range and the temperature learning rate.
- Backend: one pinned backend, with both agents launched in the same command, as the script already enforces.
- Cells: each cell the preregistration covers. At least one equal-class instance (2,2) and one unequal instance (1,3) must be among them.

**Selection rule (symmetric; it never looks at the gap between agents).**
- Score: m_i(c) is the interquartile mean (IQM) over the val seeds of the preregistration's primary metric. That metric is owned by the preregistration and must be stated in it before the grid is run.
- Regret: r_i(c) = m_i(c) − best over the grid of m_i, averaged over the preregistered cells.
- Choice: c* = argmin over c of max(r_OC(c), r_SAC(c)). Ties go to the smaller c.
- Per-agent bests c_i* = argmin m_i are also recorded.

**Test tier.**
- Primary contrast: both agents at c*.
- Secondary contrast: each agent at its own c_i*. If c_i* = c* for both, it is the same run.
- The claim holds only if both contrasts pass at the preregistered level.

**Budget per agent:** 6 grid points × 10 val seeds × the preregistered cells, plus the c = 0 reference. Each grid point is one launch carrying both agents, inside the daily cap of 40 runs.

**Replacement text for the CLAUDE.md Protocol bullet:**

> - Entropy targets are a fraction of the entropy range each agent's own policy class can reach in its own entropy space: H_target = H_min + c(H_max − H_min). H_min and H_max are computed when the agent is built, from its entropy estimator and the actions it cannot tell apart (drugs with identical embeddings). For OC-SAC on crisp classes the range is [0, log M]; for SAC with tied networks it is [log n_min, log K]. One c, shared by both agents, is chosen on val seeds from the grid {0.05, 0.15, 0.3, 0.5, 0.77, 0.89}. The choice minimises the larger of the two agents' regrets against their own best val score, ties going to the smaller c. Each agent at its own best c is run beside it at test. A target written as c·log M or c·log K, a target outside the reachable range, or one flat target shared across agents is a bug.

**Code change (only ocmdp/agents/, experiments/ and tests/):**

1. `ocmdp/agents/oc_sac.py`, SACConfig:
   - Replace `entropy_low` / `entropy_high` with `target_fraction_low: float` and `target_fraction_high: float | None = None`.
   - The low fraction is keyword-only and required, so no run inherits a constant silently. If the high fraction is None it equals the low one (a flat target).
   - Validation: 0 ≤ low ≤ high < 1.
   - The rename keeps old reports unambiguous.
2. `Agent.__init__`: replace `max_entropy` with `self.entropy_range = reachable_entropy_range(self.entropy, tied_classes(embeddings), n_drugs)`.
3. New `ocmdp/agents/entropy_range.py`:
   - `tied_classes(embeddings)`: groups rows of the embedding tensor that are exactly equal. This is computed inside agents, without editing core.
   - `reachable_entropy_range(entropy, classes, n_actions)`:
     - Both estimators in use (categorical entropy, and entropy of a linear pushforward) are concave in the policy.
     - So H_min is the minimum over the class vertices, i.e. uniform probability on one tied class.
     - H_max comes from Adam ascent in float64 on softmax class masses expanded uniformly within each class.
     - It raises an error unless H_max > H_min.
     - The docstring states the concavity assumption. The similarity-sensitive estimator is not covered until its concavity is shown.
4. `_update`: `lo, hi = agent.entropy_range; target = entropy_target(obs[:, :n_mech], lo + f_low*(hi-lo), lo + f_high*(hi-lo), cfg.fixation_threshold)`. `losses.entropy_target` itself is unchanged.
5. `losses.temperature_loss` gets the raw (unclamped) head output and the clamp range. In `_update` the clamp is still applied, but only where alpha is used: in the actor loss and the critic target.
   ```python
   gap = (entropy - target).detach()
   low, high = log_alpha_range
   inside = (raw_log_alpha >= low) & (raw_log_alpha <= high)
   inward = ((raw_log_alpha < low) & (gap < 0)) | ((raw_log_alpha > high) & (gap > 0))
   return (raw_log_alpha * gap * (inside | inward)).mean()
   ```
   - Why: line 247 of oc_sac.py clamps log_alpha before the temperature loss, and `clamp` has zero gradient outside its range. The head reads `h.detach()`, so the temperature loss is its only gradient.
   - Consequence: a state whose raw log-alpha has fallen below −12 gets no restoring gradient from itself. It can only come back indirectly, through shared head weights or drift in h. The floor is therefore close to absorbing by construction.
   - I found this by reading the code; no run has checked it. It is one possible explanation for OC-SAC's median evaluation temperature sitting at the floor while its training-window mean stays high on some seeds.
6. `experiments/train_agent.py`:
   - Record `entropy_range_nats` per variant and derive `entropy_target_nats` from it (line 633 currently uses `max_entropy`).
   - Add `--set-variant VARIANT KEY=VALUE` for the secondary contrast.
   - Per run, add the share of alpha_path windows at each clamp edge, the share of evaluation states at each edge, and the mean of (H − target) at evaluation.
7. New `experiments/select_target_fraction.py`: reads val reports and prints c* and c_i* by the rule above. It launches nothing.
8. Tests:
   - Ranges: (2,2) gives OC [0, log 2] and SAC [log 2, log 4]. (1,3) gives OC [0, log 2] and SAC [0, log 4]. A partial-overlap matrix gives OC H_min equal to the smallest row entropy.
   - Equal classes: SAC target − OC target = log n at every c.
   - Reflection: zero gradient outward, nonzero gradient inward.
   - Update `tests/test_oc_sac.py:57` and `tests/test_system.py:381,414`.

### 2. Criterion by criterion

1. **Parity.**
   - For 0 < c < 1 the target lies strictly inside [H_min, H_max] for every agent and every instance. Some policy the actor can express sits below the target (temperature can rise) and some sits above it (temperature can fall), on equal and unequal classes alike.
   - The fact that SAC's temperature can only rise when c > log n_min / log K disappears, because the target is now above log n_min for any c > 0.
   - The reflecting clamp removes the near-absorbing floor.
   - Same grid, same val seeds, same episodes, same quarters, same backend for both agents.
   - Limit: "not inert by construction" does not mean "not slack in practice". On (1,3), SAC putting all its mass on the size-3 class reaches log 3 and already meets any target with c < 0.79. That is the treatment, not an artefact; see criterion 2.
2. **Isolation.**
   - On equal classes, SAC's entropy is H(q) + log n and its target is c log M + log n, so both agents' constraint becomes the same inequality, H(q) ≥ c log M.
   - Under tied networks they still differ by a reward shift alpha(s)·log n. It does not depend on the action at the current state, but it does depend on the state, so it reaches Q through the soft value of later states. The equal-class control (rung 2) is expected to tie up to that shift.
   - On unequal classes the remaining difference is alpha(s)·Σ q_j log n_j, which is the within-class entropy term the claim is about.
   - Other named differences: the reachable ranges differ, by design matched only in fraction. The critic's soft target carries alpha·H in each agent's own space, as in standard SAC.
   - The secondary contrast adds one more named difference, c_OC* ≠ c_SAC*.
3. **Fixed in advance.** The grid, seed tier, seed count, episode count, selection rule, tie-break and the both-must-pass condition are all stated now. The metric must be fixed by the preregistration before the val grid runs. No test-tier output is used.
4. **Precedent.**
   - The rule is the dynamic rule of Mohammadpour, Frejinger and Bacon (ICLR 2024): target = αH(U) + (1−α)H(V), with V "the minimum reasonable entropy policy". Here V is taken to be the least-entropy policy the actor can express. That instantiation is my own and is not in their paper; their discrete analysis assumes a minimum entropy of zero.
   - The failure it fixes is documented by Xu et al. (arXiv 2112.02852): with similar actions, "some environments have a minimum entropy level not close to zero", and a target below it makes "α drop exponentially fast".
   - The fractions come from Christodoulou (0.98), CleanRL (0.89) and Mohammadpour et al. (0.77).
5. **Generality.**
   - Unequal classes are covered, as above.
   - Partial overlap: SAC's range becomes [0, log K]. OC-SAC's range is computed from the soft assignment. This matters because the current rule already breaks there for OC-SAC.
   - Per-state feasibility masks would need per-state ranges (Mohammadpour et al.'s motivation). The quarterly actor has no mask today.
   - Limit: the computation needs a concave entropy estimator.
6. **Implementable.** Everything is in ocmdp/agents/, experiments/ and tests/. It calls `ocmdp.objectives` read-only and does not touch eval, envs or core.

**Tension between criteria 1 and 2 (not a contradiction).** If "same tuning budget" is read as "each agent tuned for itself", the agents end up with two different constants, which breaks isolation. If it is read as "one c chosen on both agents' data", isolation holds but one agent may be under-tuned. I do not pick a reading. I run both contrasts and require both to pass.

### 3. The strongest objection to this proposal

**Objection:** "Your SAC is not SAC. Its target depends on your shared-embedding architecture, and on equal classes it constrains H(q) exactly as OC-SAC does. You have made the baseline into the method, and on unequal classes it is not the published rule."

**Not fatal, because:**
- On this architecture the published c·log K rule is inert. That is both an identity and what the logged runs show. Published-rule SAC is therefore the weaker baseline, not the honest one.
- This rule reduces exactly to c·log|A| whenever no actions are tied. It is the published rule wherever the published rule is well posed; on (1,3) the two coincide.
- The grid contains the published fractions. Together with the per-agent secondary, a reviewer can see how SAC does at its own best published-range c.

**Remaining risk:** near-ties. Rows that are almost but not exactly identical give H_min = 0 in principle, but the network may not separate them in practice. The practical floor would then be close to log n and the old problem would come back unseen. If an untied-head SAC is wanted to answer the objection directly, that is a new comparator, and adding one is the owner's decision.

### 4. What would show the proposal wrong

1. **One dev launch.**
   - Setup: both agents, dev seeds 0–4, instances (2,2) and (1,3), c = 0.5, sigma 0, temperatures logged.
   - Failure condition: either agent's median evaluation alpha sits at a clamp edge on 3 or more of 5 seeds while its entropy is on the side of the target that should push alpha back.
   - Also check: across windows, the sign of (H − target) should agree with the change in log alpha.
2. **The analytic H_min against the achieved one.** At c = 0 on dev, compare the lowest policy entropy reached at evaluation states with the computed H_min. A gap above 10% of the range means the analytic range is wrong, which is the near-tie risk.
3. **The isolation check on equal classes.** On (2,2), the paired difference at matched c should be no larger than the seed-chaos floor measured by the zero-target run. If it is larger, something other than the entropy space differs, most likely the alpha(s)·log n shift.
4. **The cost of sharing c.** On val, if max_i r_i(c*) is large compared with the between-seed spread, the shared c handicaps one agent. Then the per-agent contrast should become primary, through a new dated preregistration.

### 5. Prior work

- Mohammadpour, Frejinger, Bacon, "Decoupling regularization from the action space", ICLR 2024, arXiv 2406.05953: static rule τ/log|A(s)|, dynamic target αH(U) + (1−α)H(V), α ≈ 0.77. Fetched 2026-11-23.
- Xu, Hu, Liang, McAleer, Abbeel, Fox, "Target Entropy Annealing for Discrete Soft Actor-Critic", arXiv 2112.02852: diagnoses the non-zero minimum entropy caused by similar actions. Their fix anneals the target down from log|A|. I did not adopt it because the final target depends on each agent's own training run, which makes parity opaque. Fetched 2026-11-23.
- Christodoulou 2019, arXiv 1910.07207 (0.98·log|A|); CleanRL sac_atari (0.89); SD-SAC, arXiv 2209.10081 (fixed α = 0.05). All from `inbox/2026-11-16-discrete-sac.md`.
- I considered fixed α and declined it as primary. It isolates the entropy space cleanly, but its scale against the reward would need its own tuning, and the strongest published discrete SAC baselines tune α automatically.
- Hybrid SAC, arXiv 1912.11077 (a separate temperature per factor), and MinRed / Action Redundancy, arXiv 2102.11329: both from `inbox/2026-11-16-mechanism-level-sac.md`.
- I did not find any published rule that normalises the target by the minimum entropy reachable under tied or duplicate actions. That comes from one standard-mode search, not an exhaustive one.

### Files

- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/arena/2026-11-23-temperature-criteria.md
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/ocmdp/agents/oc_sac.py (lines 58–59, 113, 247, 252)
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/ocmdp/agents/losses.py (lines 49–52)
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/experiments/train_agent.py (lines 6, 568, 625, 633)
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/tests/test_oc_sac.py:57
- /Users/diyap/Occupancy-Coupled-Markov-Decision-Processes/tests/test_system.py:381,414

Nothing was written to the repository.

Sources:
- [arXiv 2406.05953 (full text)](https://arxiv.org/html/2406.05953v1)
- [arXiv 2112.02852 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2112.02852)
- [arXiv 2112.02852 (PDF)](https://arxiv.org/pdf/2112.02852)
- [SAC Algorithms and Applications, arXiv 1812.05905](https://ar5iv.labs.arxiv.org/html/1812.05905)
- [OpenAI Spinning Up, SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
