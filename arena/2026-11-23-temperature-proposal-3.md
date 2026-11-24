# Proposal 3

**Proposal: the temperature protocol for comparing OC-SAC and SAC (arena/2026-11-23-temperature-criteria.md)**

I wrote nothing to the repository. Everything below comes from reading the files the arena lists. The only figures I quote are from runs/train/train_20261123_190717_20498.json.

## 0. What the code shows, beyond the given facts

- **Why the temperature cannot be tuned "per space" with tied drugs.** In that run, every sigma=0 seed of both agents spends about 0.6931 nats of policy entropy inside the classes (`policy_entropy.within_class`), which is log 2. The pairwise scorer gives identical embeddings identical logits, so SAC's entropy is always at least log 2. No fraction-of-log-K target can make SAC's controller act on the class marginal.
- **A second cause of the collapse.** `oc_sac._update` clamps `log_alpha` before `temperature_loss`, and the comment says the gradient stops outside the range. The alpha head reads detached features (`policy_head.py`), so its only training signal is the temperature loss. Once a state's raw log-alpha drops below -12, nothing pulls it back up even when entropy falls below target. The only exit is the trunk's features drifting.
  - This is a code-reading inference, not a measurement.
  - The logs are consistent with it. `alpha_at_evaluation` q10/q50/q90 is 6.144e-06 for all ten agent-seeds at sigma 0. OC-SAC seed 2's training-window means wander between about 6.7e-06 and 0.176, while the other seeds stay pinned.
  - So OC-SAC's temperature also collapses, even though its target (0.05·log 2) is reachable.

## 1. The recommendation

**The temperature becomes a per-agent hyperparameter. Every entropy-regularised agent picks its value from the same fixed candidate set, at parity, on val seeds.** Fixed temperatures are the core of the set. Two learned-temperature candidates, using the published targets corrected to the range each agent can actually reach, are added so that SAC's published configuration is covered.

**Candidate set (identical for every agent).**
- Fixed α ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3}.
  - The unit is scaled reward per nat, at `reward_scale` 10, which all agents share.
  - A nat is a nat in either space, so an absolute α is the same reward-per-nat exchange rate for both agents. Dividing by log M or log K is what created the asymmetry in the current rule.
- Learned scalar α with target H_floor + c·(H_max − H_floor), for c ∈ {0.89, 0.98}.
  - H_floor and H_max are the least and greatest entropy the agent's own policy class can reach, measured in its own entropy space.
  - With tied drugs, SAC's range is (log n_min, log K) and OC-SAC's is (0, log M). This is the decoupled dynamic rule of arXiv 2406.05953, with their minimum-entropy policy V taken inside the agent's own policy class.
  - Whenever a point mass is reachable, the floor is 0 and this reduces exactly to the published c·log|A| targets (0.98 and 0.89).
  - On equal tied classes it gives both agents the same class-level constraint, H(q) ≥ c·log M.

**Rule for widening the grid (fixed now; uses entropy only, never failure rates).**
- Run on dev first.
- If, at the largest fixed α, either agent's mean class entropy at evaluation is below 0.9·log M, append half-decade points (10, 30, …) until both reach it.
- If, at α = 0.01, either agent's class entropy is above 0.1·log M, prepend 0.003, 0.001, … until both are below.
- The grid stays identical for all agents.

**Selection.**
- Each agent takes the candidate with the lowest val mean of the preregistration's primary metric. The mean is over 5 val seeds and over every cell (instance × sigma) the preregistration will test, cells weighted equally.
- One value per agent per preregistration. Ties go to the earlier candidate in the order: fixed ascending, then auto ascending in c.
- The full table and both choices are logged, and frozen in a new dated preregistration before any test-tier run.

**Budget per agent.** 9 candidates × 5 val seeds × the preregistration's cells, all 12,000-quarter runs.
- Every val launch trains both agents at the same candidate on the same seeds and evaluation episodes (train_agent.py already pairs them), so equal budget is structural.
- For PairedDrugs at two sigmas that is 180 training jobs. It must fit budget.yaml or be spread over days.

**Dev check before val (the check the arena names, adapted).** Candidates {fixed 0, 0.1, 3; auto 0.89}, both agents, dev seeds 0-4, sigma 0, on class sizes (2,2) and (8,5,4,2,1). That is 80 jobs, used for:
- the grid-widening rule;
- confirming the learned α is off the clamp in every training window and at evaluation;
- the equal-class identity check in section 4.

**Proposed CLAUDE.md Protocol bullet (replaces the H_target_OC / H_target_SAC bullet):**

> - Temperature is a per-agent hyperparameter chosen at parity, never one constant scaled per agent. Every entropy-regularised agent is tuned over the same candidate set, on the same val seeds, cells and evaluation episodes, with the same number of runs: fixed temperatures {0, 0.01, 0.03, 0.1, 0.3, 1, 3} in scaled reward per nat, and learned scalar temperatures with target H_floor + c(H_max − H_floor) for c in {0.89, 0.98}, where H_floor and H_max are the least and greatest entropy the agent's own policy class can reach in its own entropy space. The fixed set may only be widened, identically for every agent, by the entropy-only spanning rule run on dev. Each agent takes the candidate with the lowest val mean of the preregistration's primary metric over its cells; the full table and both choices go into the preregistration before any test run. The matched-temperature curve is reported beside the tuned comparison. A learned temperature whose target lies outside (H_floor, H_max) for some agent on some instance, or whose gradient stops at a clamp, is a bug.

**Code change** (only ocmdp/agents/, ocmdp/components/, experiments/ and tests/; nothing in eval, envs, core, baselines or theory):

- **`SACConfig`**
  - Remove `entropy_low`, `entropy_high` and `fixation_threshold`.
  - Add `temperature: str = "fixed"` ("fixed" or "auto"), `alpha: float = 0.0` (must be ≥ 0), `target_fraction: float = 0.89` (strictly between 0 and 1) and `alpha_lr: float = 3e-4`.
  - Keep `alpha_init` and `log_alpha_range` for "auto".
- **New `ocmdp/agents/entropy_range.py`**
  - `tied_groups(embeddings)` groups identical embedding rows, which the pairwise scorer cannot separate.
  - `reachable_entropy_range(entropy_fn, groups, n_actions)`:
    - floor = the minimum over groups of `entropy_fn(uniform on the group)`. Entropy is concave, so its minimum over the polytope of group-uniform policies sits at a vertex.
    - max = exponentiated-gradient ascent over group weights. This is exact for hard classes and numerical for the soft assignment of partial cross-resistance, where `mechanism_assignment()` is row-normalised and the floor is min_g H(S_g).
- **`losses.py`**
  - `entropy_target(floor, maximum, fraction)` returns `floor + fraction * (maximum - floor)`.
  - `temperature_loss` takes a scalar raw `log_alpha` and the batch-mean entropy.
- **`oc_sac.Agent`**
  - Add an `nn.Parameter` `log_alpha` when the temperature is "auto", with its own Adam optimiser.
  - Add `temperature()`, which returns `cfg.alpha` for "fixed", or `log_alpha.clamp(range).exp()` for "auto". The clamp affects the value only: the loss is taken on the raw parameter, so the clamp cannot absorb it.
  - Store `entropy_floor`, `entropy_max` and `entropy_target`.
- **`_update`**: use `agent.temperature()` in the soft critic target and the actor loss. Step the α optimiser only for "auto".
- **`MixingPolicyHead`**: remove `log_alpha_head` and return probabilities only. Leaving it would be dead code. This drops the state-dependent α(s) and the rising target, which validation had already set flat. **This is a design change for you to approve.**
- **`experiments/train_agent.py`**
  - Add a `--temperature {fixed,auto}` flag.
  - Add `--variant-set VARIANT:KEY=VALUE`, for test runs whose selected values differ between agents.
  - Make `_continuation_return` and `alpha_at_evaluation` use `agent.temperature()`.
  - Replace `entropy_target_nats` with per-variant `{floor, max, target, alpha}`.
  - Add per-run `bonus_share`: mean α·H at evaluation states divided by mean |scaled reward|.
- **New `experiments/select_temperature.py`**
  - Reads the val reports.
  - Checks they share tier, seeds, episodes, cells and code hash, and that every candidate covers both agents on the same seeds.
  - Applies the selection rule and writes `runs/selection/…json`.
- **Tests**
  - At fixed α = 0, OC-SAC and SAC produce bitwise-identical parameters after N updates.
  - Ranges: (2,2) gives SAC (log 2, log 4) and OC (0, log 2); (8,5,4,2,1) gives (0, log 20) and (0, log 5); a soft-assignment case gives floor min_g H(S_g).
  - The temperature gradient is nonzero when `log_alpha` is below the clamp range.
  - With equal classes, the actor gradients are identical when the two critics differ by a function of state only.
  - Update tests/test_agent_objective.py, tests/test_oc_sac.py and the target assertions in tests/test_system.py.

## 2. Criterion by criterion

1. **Parity.**
   - Not inert or saturated by construction: a fixed α has no target, so nothing can saturate, and α > 0 always enters the actor loss. The learned targets lie strictly inside each agent's reachable range for 0 < c < 1, on equal and unequal classes alike, and the unabsorbing clamp removes the second way the temperature was stuck.
   - Same budget: identical candidates, val seeds, cells and evaluation episodes for each agent, and each launch trains both agents.
   - Whether the bonus *matters* at the selected values is empirical. `bonus_share` and the class-entropy-versus-α table show it.
2. **Isolation.** At matched α, with tied drugs, SAC is exactly OC-SAC plus a reward of α·log n_class(a) (decisions/2026-11-23.md, theory-checker entry). So the agents differ only in the entropy space, and that term is all the space contributes. What else differs, named:
   - (a) the tuned α values may differ between agents, so the matched-α val curve is reported beside the tuned comparison;
   - (b) the reachable ranges are (0, log M) against (log n_min, log K);
   - (c) with unequal classes, the extra term is a bonus towards larger classes, and under the learned targets SAC meets its constraint more cheaply when it favours them;
   - (d) with partial overlap, the embeddings are no longer tied, so SAC can also earn entropy by splitting between near-identical drugs;
   - (e) state-dependent α(s) is removed for both agents.
3. **Fixed in advance.** The candidate set, the widening rule (entropy only), the selection metric and the tie-break are fixed here. The dev and val steps come before any test run. A new dated preregistration freezes the chosen values.
4. **Precedent.**
   - Fixed temperature: SAC v1 used a fixed temperature (as a reward scale) per task; SD-SAC used a fixed α of 0.05; arXiv 2406.05953 uses a static rule and fixed temperatures on DeepMind Control.
   - The closest prior on equivalent actions, MinRed SAC (arXiv 2102.11329), used a fixed α per environment, taken from SAC's auto-tuned value (Appendix B.2, Table 7).
   - The learned candidates are the published 0.98 and 0.89 targets, with the floor from arXiv 2406.05953. Moving the floor into the agent's policy class has no published support. The reason: under tied networks the point mass their V assumes cannot be reached.
5. **Generality.** Fixed candidates need no range at all. The learned candidates compute floor and maximum for any class sizes and for soft (partial-overlap) assignments. Not covered: per-state feasibility masks, where arXiv 2406.05953 shows a fixed α over-regularises states with more feasible actions. The training environments have no contraindications at the quarter decision.
6. **Implementable.** Only ocmdp/agents/, ocmdp/components/policy_head.py, experiments/ and tests/ change.

## 3. Strongest objection, and whether it is fatal

**Objection.** Fixed temperatures are brittle across training and across instances; this is why automatic tuning was introduced (arXiv 1812.05905). Choosing one α per agent from a half-decade grid on 5 val seeds is noisy, and it lets the test-tier gap include "which α each agent drew" as well as the space. A single α across a preregistration's cells may also favour whichever agent's best α varies less across instances.

**Not fatal:**
- Selection noise is symmetric: equal grid, equal seeds. It is resolved on test seeds the selection never saw, so it adds variance but no bias towards either agent. The preregistration's power calculation has to absorb that variance.
- The matched-α curve shows whether the conclusion depends on α.
- The learned candidates are the instance-adaptive option inside the same budget.

**Second objection, which goes to you.** Removing α(s) drops a stated OC-SAC design element. The logs show that element is already pinned at the floor in practice.

**On the arena's instructions about contradictory or empirical criteria.** The criteria do not contradict each other. Parity (per-agent tuning) and isolation (only the space differs) pull in different directions; that is handled by naming the α difference and reporting the matched curve. Several parts are empirical, not design: whether the grid spans the useful range, and which candidate each agent selects. The fact that the optimal policies are near-deterministic makes α = 0 for both agents a live outcome. That would be a tie, which is a null result to report, not something to tune away.

## 4. What would show I am wrong

- **Grid is not useful.** On the dev check, class entropy at evaluation does not change across the nonzero fixed α for either agent, or val failure rates jump by more than the between-seed spread between neighbouring grid points.
- **Learned rule still stuck.** A learned α sits at the clamp on any seed: the range calculation or the gradient fix is wrong.
- **Isolation premise fails.** On (2,2), the paired OC-minus-SAC difference at matched α falls outside the seed-chaos floor; for reference, the agreed-noise run is `runs/train/train_20261123_190750_20538.json`. Something besides the space would then differ.
- **Selection is noise.** Rerunning selection on each half of the val seeds moves either agent's choice by more than one grid step: more seeds are needed.
- **SAC baseline too weak.** Any SAC configuration found outside the set (by you or a reviewer) beats the selected SAC on val: the candidate set is not the strongest honest baseline.
- **Entropy inert for both.** `bonus_share` at both selected values is negligible: the comparison then cannot test entropy at all, and the finding is that action entropy suffices.

## 5. Prior work

- arXiv 2406.05953 (ICLR 2024). Static rule τ/log|A(s)|; dynamic target αH(U)+(1−α)H(V) with V the minimum-entropy policy; α≈0.77; temperature fixed at 0.25 on DeepMind Control. It notes τ "will grow to infinity if the target entropy is infeasible".
- arXiv 2102.11329 (UAI 2021, MinRed SAC). Fixed α per environment, set from SAC's auto-tuned average. This is the closest prior comparison of entropy over equivalence classes against action entropy. The text does not say whether SAC and MinRed SAC shared α.
- arXiv 1910.07207: target 0.98·log|A|, untuned. CleanRL: 0.89·log|A|, with the advice to tune it (inbox/2026-11-16-discrete-sac.md).
- arXiv 2209.10081 (SD-SAC): fixed α of 0.05.
- arXiv 1812.05905: automatic tuning, introduced because a fixed temperature is brittle. This is the counter-precedent.
- arXiv 2007.01932 (Meta-SAC) and arXiv 2112.02852 (TES-SAC): the target entropy is itself a hyperparameter that needs tuning per task. Both support treating either form as tuned at parity.
- arXiv 1912.11077 (Hybrid SAC): separate temperatures per factor, each auto-tuned to its own target after "a cursory search".
- arXiv 2103.06257: the robustness reading of maximum-entropy RL. Robustness appears only at relatively large coefficients. Whether its bound assumes a fixed coefficient was not checked.

Relevant files:
- ocmdp/agents/oc_sac.py
- ocmdp/agents/losses.py
- ocmdp/components/policy_head.py
- experiments/train_agent.py
- runs/train/train_20261123_190717_20498.json
- runs/train/train_20261123_190750_20538.json

Sources:
- [arXiv 2406.05953 (html)](https://arxiv.org/html/2406.05953v1)
- [arXiv 2102.11329 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2102.11329)
- [arXiv 2103.06257](https://www.arxiv.org/abs/2103.06257)
- [arXiv 2103.06257 (ar5iv)](https://ar5iv.org/html/2103.06257)
- [BAIR blog on MaxEnt robustness](https://bair.berkeley.edu/blog/2021/03/09/maxent-robust-rl)
- [arXiv 2112.02852 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2112.02852)
- [arXiv 2209.10081v4](https://arxiv.org/html/2209.10081v4)
- [arXiv 2007.01932](https://arxiv.org/pdf/2007.01932)
- Also cited, through the repository's inbox files rather than fetched this session: arXiv 1910.07207, 1912.11077 and the CleanRL docs; arXiv 1812.05905 from general knowledge.
