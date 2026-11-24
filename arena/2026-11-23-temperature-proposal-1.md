# Proposal 1

## Proposal: set each agent's entropy target as a fraction of the range its own policy class can reach

The current rule is the discrete case of a published rule, but it takes a zero entropy floor for granted. Under tied networks, SAC's floor is not zero. The fix is to measure c from each agent's real floor, not from zero.

### 1. The rule

**Rule.** For each agent X (OC-SAC with H_X = H(psi#pi); SAC with H_X = H(pi); any later entropy-regularised comparator with its own H_X):

- H_floor,X = the minimum of H_X over the policies X's network can express.
- H_ceil,X = the maximum of H_X over the same set.
- H_target,X(s) = H_floor,X + c_X(s) * (H_ceil,X - H_floor,X).
- c_X(s) = c_low + (c_high - c_low) * closeness(s), with 0 < c_low <= c_high < 1, both strictly inside. This keeps the existing fixation sigmoid. For this decision the target stays flat (c_low = c_high = c_X).

**What "can express" means.** The scorer gives drugs with identical embeddings identical logits, so the expressible policies are the ones that are uniform within each group of identical embeddings. Both entropies the Agent uses (action entropy, and pushforward through a hard or soft assignment) are concave in pi. So:
- The floor is reached at a vertex: all mass spread uniformly over one group. It can be computed exactly by enumerating the groups.
- The ceiling is a concave maximisation over the group masses. Exponentiated-gradient ascent in float64 finds it, and tests check it against closed forms.

| Instance | SAC range | OC-SAC range |
|---|---|---|
| PairedDrugs (2,2) | [log 2, log 4] | [0, log 2] |
| Classes (3,1) | [0, log 4] | [0, log 2] |
| Partial overlap (rows of C distinct) | [0, log K] | [min_a H(S_a), max_pi H(pi S)], found numerically |

**How c is chosen, on which tier, with what budget.**
- **Grid, fixed now:** G = {0.05, 0.15, 0.3, 0.5, 0.89}. 0.05 is the current default, read in the new units. 0.89 is CleanRL's published discrete default. Both agents use G.
- **Step A, dev check (filters on temperature only).**
  - One launch: both agents, dev seeds 0-4, sigma 0, every c in G, on PairedDrugs and on `--class-sizes 3 1`.
  - A value of c passes for an agent on an instance if, on at least 4 of 5 seeds, both of these hold:
    - (i) fewer than half of the evaluation states have raw log alpha at either clamp bound;
    - (ii) over the last training window, mean |H - target| / (H_ceil - H_floor) is below 0.1.
  - Any c that fails for either agent on either instance is dropped for both. Failure rates are not consulted.
  - If the upper clamp binds, the clamp is widened for both agents identically, and the change is logged in decisions/.
- **Step B, val selection.**
  - Each agent trains at every surviving c on the same five val seeds and the same val evaluation episodes, with the same 12 000 quarters per run.
  - Each agent's c is the one that minimises its own val mean of the preregistration's primary metric. Ties go to the smaller c. Each choice gets a line in decisions/.
  - Budget per agent is identical: |G_surviving| x 5 seeds x 12 000 quarters per instance.
- **Step C, the preregistration fixes:**
  - c*_OC and c*_SAC;
  - the primary comparison: each agent at its own tuned c, which gives the baseline its strongest honest form;
  - the matched-c comparison: both agents at each c in G. It costs nothing extra because it is already in the val data, and it is reported beside the primary one.
  - Nothing is chosen after a test-tier outcome.

**Replacement text for the CLAUDE.md Protocol bullet**

> - Entropy targets are a fraction of the range each agent's own policy class can reach in its own entropy space: H_target = H_floor + c (H_ceil - H_floor), where H_floor and H_ceil are the minimum and maximum of that agent's entropy over the policies its network can express (computed in ocmdp/agents/entropy_range.py and recorded in every report; with drugs scored by identical embeddings, SAC's floor is the log of the smallest tie-group size, not zero). c lies strictly inside (0, 1). A shared target in nats, or a fraction of log|space| without the floor, is a bug. Each agent's c is chosen from one grid, fixed in a dated decision before any val run, after a dev check that drops, for both agents, every grid value at which either agent's temperature sits at a clamp or fails to track its target. Each agent gets the same grid, val seeds, evaluation episodes and training budget, and is selected on the preregistration's primary metric. The matched-c comparison across the grid is reported beside the tuned one. Both selected values are fixed in the preregistration before any test run.

**Code change** (only ocmdp/agents/, experiments/ and tests/; nothing in eval/, envs/ or core/)

- **New `ocmdp/agents/entropy_range.py`:**
  ```python
  def reachable_entropy_range(entropy, embeddings, steps=20_000, step_size=0.5, tol=1e-12) -> tuple[float, float]:
      """Floor and ceiling of a concave entropy over the policies a scorer of `embeddings` can express:
      those uniform within each group of identical embeddings. Concavity puts the floor at a vertex
      (all mass uniform on one group); the ceiling is found by exponentiated-gradient ascent on group masses."""
      _, group = torch.unique(embeddings, dim=0, return_inverse=True)
      # spread: (G, A) float64, row g uniform over group g
      # floor = entropy(spread).min(); ceiling = entropy(w @ spread) at the EG fixed point
  ```
  `entropy` must be evaluated in float64; the caller passes `lambda p: pushforward_entropy(p, assignment.double())` or `categorical_entropy`. If H_ceil - H_floor < 1e-6, raise ValueError: there is no range to regulate.
- **`losses.entropy_target`:**
  ```python
  def entropy_target(resistance, h_floor, h_ceil, c_low, c_high, threshold=0.5, sharpness=10.0):
      closeness = torch.sigmoid(sharpness * (resistance.max(dim=-1).values - threshold))
      return h_floor + (h_ceil - h_floor) * (c_low + (c_high - c_low) * closeness)
  ```
  It raises unless 0 < c_low <= c_high < 1.
- **`SACConfig`:** `entropy_low` and `entropy_high` are redefined as "fractions of the reachable entropy range in the chosen space". `__post_init__` checks that both lie in (0, 1).
- **`Agent.__init__`:** `self.max_entropy` is replaced by `self.entropy_floor, self.entropy_ceiling` from `reachable_entropy_range`, using the agent's own entropy and embeddings.
- **`_update`:**
  - The target becomes `entropy_target(obs[:, :n_mech], agent.entropy_floor, agent.entropy_ceiling, cfg.entropy_low, cfg.entropy_high, cfg.fixation_threshold)`.
  - The raw log alpha is kept before clamping, and two quantities are returned and logged per window in `TrainResult`: the fraction of the batch at each clamp bound, and the mean of (entropy - target).
- **`experiments/train_agent.py`:**
  - `--set` accepts `oc-sac.KEY=VALUE` and `sac.KEY=VALUE`; an unprefixed key still applies to both agents.
  - Line 625's `max_entropy` is replaced by `entropy_range_nats` (floor and ceiling per variant, computed by the same function).
  - `entropy_target_nats` is computed per variant from that variant's config.
  - Each run records the clamp fractions (lower and upper, in training and at evaluation states) and the path of entropy minus target. The summary aggregates them.
- **Tests:**
  - PairedDrugs ranges are [log 2, log 4] and [0, log 2]; (3,1) ranges are [0, log 4] and [0, log 2].
  - With equal classes, SAC target - OC target = log n for every c.
  - On a partial-overlap matrix, the pushforward floor equals the smallest row entropy.
  - c = 0 and c = 1 are rejected.
  - For every c in (0,1), SAC's target is above its floor. This is the regression test for the inert temperature.

### 2. Against each criterion

1. **Parity.**
   - Because every target lies strictly inside its agent's reachable range, entropy can sit on either side of it, and the dual update can move alpha both ways. That holds on equal and unequal classes alike.
   - The old failure ("c > log n_min / log K") cannot happen: SAC's target is log n_min + c(...) > log n_min for any c > 0.
   - Not inert by construction does not guarantee not inert in practice. That is what Step A tests, and its filter is symmetric.
   - Same grid, seeds, episodes and quarters for both agents.
2. **Isolation.**
   - Under tied networks, H(pi) = H(q) + sum_j q_j log n_j.
   - **Equal classes:** SAC's target is log n + c log M, which holds H(q) = c log M. That is exactly OC-SAC's target, so at matched c the two agents optimise the same objective up to a constant. The rule makes the control rung a tie by construction, which the claims ladder says is the correct result.
   - **Unequal classes:** the only remaining difference is the within-class term sum_j q_j log n_j. That is the entropy space, the thing under test.
   - Named differences:
     - (a) The same c maps to different nats, because the range widths differ (log M versus log K - log n_min).
     - (b) SAC can meet its target with H(q) near 0 by concentrating on a large class. Its effective bonus is alpha * log n_class(a), which favours large classes.
     - (c) The alpha values each dual reaches differ, and are logged.
     - (d) In the primary comparison c*_OC and c*_SAC may differ. The matched-c comparison removes (d).
3. **Fixed in advance.** The grid, the dev pass rule (temperature fields only), the val selection rule, the tie-break and the seed counts are all stated here. Selection happens on val, and nothing reads a test outcome.
4. **Precedent.** The rule is the dynamic rule of Mohammadpour, Frejinger and Bacon (target alpha H(U) + (1 - alpha) H(V), V the minimum-entropy policy), with U and V taken within the agent's expressible policy class.
   - They set H(V) = 0 for discrete actions. That assumption is what fails here.
   - Their static rule likewise divides the temperature by the regulariser's range.
   - Xu et al. document the exact failure: when the minimum reachable entropy is above the target, alpha drops exponentially.
   - What has no precedent I found: computing the floor over a network-restricted policy class. Prior work assumes the full simplex.
5. **Generality.**
   - Unequal classes are handled by the per-group floor.
   - Partial overlap works without crisp classes. Distinct rows mean no ties, so SAC's floor is 0, and the pushforward range comes from the soft assignment.
   - Per-state feasibility masks would need the range computed per state over the feasible actions, as in the paper's log|A(s)|. The Agent passes no mask today, so I propose no code for it (no dead code).
   - The similarity-sensitive entropy is not wired into the Agent. It is not known to be concave, so the vertex floor would not be guaranteed for it.
6. **Implementable.** All the changes are in ocmdp/agents/ and experiments/. The reads of `coupling` in core are existing calls.

### 3. Strongest objection to my own proposal

**The objection.** The floor is a property of the architecture, computed from exact ties. With near-identical embeddings (partial overlap close to crisp), the computed floor is 0, but the scorer can separate near-tied drugs only with huge logit differences. In practice there is a soft floor near log n. SAC's temperature could then again be driven to a clamp, this time the ceiling, while the rule says it is fine. Relatedly, "fraction of the reachable range" is a normalisation convention, not a fact. With unequal classes, the same c means different exploration pressure in nats.

**Not fatal:**
- The default and (3,1) instances have exact ties, so the floor is exact there.
- The soft-floor case is caught by Step A's ceiling-clamp and tracking conditions, which are checked per agent and per instance.
- The convention problem is answered by tuning c per agent at an equal budget, with the matched-c curve reported beside it, so neither reading of "equal pressure" is hidden.

It becomes fatal only if no grid value passes Step A for both agents. Then the rule fails criterion 1 in practice, and that should be reported rather than patched.

**A rejected alternative that should be weighed:** one fixed alpha shared by both agents (the SD-SAC style). Under tied networks it isolates exactly: SAC equals OC-SAC plus alpha * sum_j q_j log n_j. But it has no temperature mechanism at all, and it drops the state-dependent alpha that OC-SAC is designed around.

**One tension in the criteria, not a contradiction:** criterion 1's equal-budget, strongest-baseline tuning permits different c per agent, while criterion 2 wants only the entropy space to differ. I handle it with two reported comparisons rather than by choosing one reading.

### 4. What would show I was wrong

- **Step A:** per agent, c and instance, the fraction of evaluation states at each clamp bound and the tracking error |H - target| / range. If no c passes for both agents, the rule fails parity.
- **Equal-class identity:** on PairedDrugs at matched c, the paired OC minus SAC difference should be within the seed-chaos floor measured by the zero-target run (`runs/train/train_20261123_190750_20538.json`). If it is not, there is an implementation asymmetry and the isolation claim is false.
- **Floor validity:** the minimum SAC entropy reached in training compared with the computed H_floor. If entropy stays well above the floor while alpha is at the upper clamp, the floor is soft.
- **Grid adequacy:** if either agent's selected c is at an edge of G (0.05 or 0.89), the grid was too narrow. A symmetric extension would need a new dated decision before any test run.
- **The old failure in the logs:** in `runs/train/train_20261123_190717_20498.json` the targets are 0.0347 nats (OC-SAC) and 0.0693 nats (SAC). Within-class policy entropy is 0.6931 and alpha_at_evaluation q10 = q50 = q90 = 6.144e-06 (the exp(-12) floor) in all ten runs I read. Under the new rule that pattern must not recur at any surviving c.
- **Cost:** not measured for the (3,1) instance or for the full grid.

### 5. Prior work

- **Mohammadpour, Frejinger, Bacon, "Decoupling regularization from the action space"**, ICLR 2024, arXiv 2406.05953.
  - Static rule: temperature divided by the regulariser's range L(Omega, A(s)).
  - Dynamic rule: target alpha H(U) + (1 - alpha) H(V), with V the minimum-entropy policy; H(V) = 0 for discrete actions; recommended alpha about 0.77.
  - Per the fetched text, the dynamic rule was tested only on continuous DeepMind Control, so its discrete use here has no direct empirical support. It does not address redundant actions. This is the closest prior rule.
- **Xu, Hu, Liang, McAleer, Abbeel, Fox, "Target Entropy Annealing for Discrete Soft Actor-Critic"**, arXiv 2112.02852 (NeurIPS 2021 Deep RL workshop page).
  - Section 4.2: when the minimum entropy is above the target, "the policy entropy will never reach the target entropy, which makes the temperature alpha drop exponentially fast."
  - Their fix starts the target at log|A| and multiplies it by 0.9 whenever entropy has stabilised.
  - Not adopted: the decay is triggered by each agent's own entropy dynamics, so the two agents' schedules would diverge, which breaks isolation.
- **Christodoulou, "Soft Actor-Critic for Discrete Action Settings"**, arXiv 1910.07207: target 0.98 log|A|.
- **CleanRL `sac_atari.py`** (docs.cleanrl.dev/rl-algorithms/sac): 0.89 log|A|, with the advice to tune per environment. Both this and the 0.98 rule are the zero-floor special case.
- **Delalleau et al., Hybrid SAC**, arXiv 1912.11077: a separate temperature and target per factor of a factored policy. Precedent for per-space targets.
- **SD-SAC**, arXiv 2209.10081: fixed alpha 0.05, the fixed-temperature alternative.
- **Baram et al., "Action Redundancy in Reinforcement Learning"**, UAI 2021, arXiv 2102.11329: redundant actions inflate action entropy. The same phenomenon is what puts SAC's floor at log n_min here.
- No source I found computes the entropy floor over a network-restricted policy class. The reason it is absent: prior work assumes the policy class is the full simplex.

**Files read:**
- arena/2026-11-23-temperature-criteria.md
- ocmdp/agents/losses.py
- ocmdp/agents/oc_sac.py
- ocmdp/objectives/pushforward_entropy.py
- ocmdp/components/policy_head.py
- experiments/train_agent.py
- decisions/2026-11-23.md
- inbox/2026-11-23-reviewer-leads.md
- inbox/2026-11-16-discrete-sac.md
- inbox/2026-11-16-mechanism-level-sac.md
- runs/train/train_20261123_190717_20498.json

Nothing was written.

Sources:
- [Decoupling regularization from the action space (arXiv 2406.05953)](https://arxiv.org/html/2406.05953v1)
- [Target Entropy Annealing for Discrete Soft Actor-Critic (arXiv 2112.02852)](https://ar5iv.labs.arxiv.org/html/2112.02852)
- [Target Entropy Annealing, NeurIPS 2021 workshop page](https://neurips.cc/virtual/2021/35700)
- [Revisiting Discrete Soft Actor-Critic (arXiv 2209.10081)](https://arxiv.org/html/2209.10081v4)
