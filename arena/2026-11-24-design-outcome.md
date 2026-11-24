# Outcome: research-design decisions left open by the 2026-11-24 plan revision

Judged blind against `2026-11-24-design-criteria.md`. The judge ranked proposal 3 first, proposal 2 second and proposal 1 third, with 2 and 1 close. Proposal 3 is adopted with the judge's nine amendments. Owner acts are at the end.

## Adopted

### 1. Rules every entropy preregistration carries

- **"Not tested"** (bias bound from proposal 2, settled on val only).
  - For each agent, B_i = α_i (H_max,i − H_floor,i) / reward_scale, in expected-failure-rate units.
  - If max_i B_i < m, the entropy-space comparison is "not tested". This is decided on val before any test launch, and no test seeds are spent on it.
  - If min_i B_i < m ≤ max_i B_i, the tuned comparison is reported as "entropy against effectively none". The entropy-space reading then comes only from the matched arm.
  - The realised share α (mean evaluation entropy − H_floor) / reward_scale is reported descriptively.
  - At m = 0.01 the rule needs α ≥ 0.032 for drug-entropy SAC at K = 22 and α ≥ 0.056 for the mechanism agent at M = 6 (derived).
- **Matched arm.**
  - Every test seed also runs each agent at the other's selected temperature.
  - A learned temperature is matched by a fixed α equal to that agent's median learned α over the last val window, frozen in the preregistration.
  - Verdict rule (from proposal 1), with Holm correction across the two matched contrasts:
    - "due to the entropy space" only if both matched contrasts share the tuned contrast's sign and exclude 0;
    - "equivalent in entropy space" only if both lie within ±m;
    - otherwise "entropy space and temperature level not separated".
- **Selection.** Each agent takes the candidate with the best arithmetic mean over its 10 val seeds and cells, every run counted, with the IQM and runs learned reported beside it. This needs an owner edit to `CLAUDE.md`.
- **Non-learning runs.**
  - They are never excluded.
  - "Learned" means more than half the reference's gain over uniform. The reference is the DP where it exists, otherwise the every-quarter replanner. The planner may serve as reference at K = 22 only after restart agreement passes there (proposal 2).
  - Per agent, a Clopper–Pearson interval on runs learned; an exact McNemar test as a secondary; the contrast among seeds where both learned is descriptive only.

### 2. Test-tier capacity

- **Learned agents.** The equivalence margin is m = 0.01 in expected failure rate, also reported as a share of the gap between uniform and the reference. Power is recomputed in each preregistration from the val spread of the selected configurations, and an inconclusive result is reported as such.
- **Claim A's evaluation episodes.** The test tier widens to seeds 1000–1299 (owner edit, from proposal 2). Episodes per cell are sized from the measured paired spreads, at most 300.
- There is no restriction to noise of 0.2 or less. That cut would be chosen with the dev-map crossings in view.

### 3. The noise-optimal reference

- **Structure (proposal 2).**
  - It lives in `experiments/dp_reference.py`, pinned by SHA-256 in the preregistration.
  - Claim A splits in two. The boundary test (`planned_once_minus_greedy` against the theory-free null) needs no DP. The decomposition test freezes only after every check below passes, with its predictions hashed first.
- **Checks, each fixed before any held-out cell runs:**
  - brute-force enumeration on a tiny cell;
  - at noise 0, a match with the planner within 1e-3;
  - a grid-refinement change below 1e-3;
  - the computed value agreeing with the simulated value within Monte Carlo error;
  - no loss to the CE replanner beyond one-sided Monte Carlo error;
  - lying between the CE replanner and the perfect-information bound;
  - the environment's reflection at 0 and 1 modelled;
  - from proposal 3: the penalised perfect-information dual bound, and exact enumeration to 1e-9.
- **Order.** The perfect-information gate runs first.
- **Placement** (`experiments/` or `ocmdp/baselines/`) is the owner's call. The scope outcome named `ocmdp/baselines/`.

### 4. Claim B2

- **Metrics.** Expected failure rate is primary. A threshold-free horizon efficacy is secondary (proposal 1), so no efficacy threshold needs setting.
- **Entropy.** The one entropy is the pushforward through row-normalised C, H(π C̃).
  - It is concave and has no free parameter.
  - A drug whose clipped row is all zero is assigned one-hot to its target group.
  - The cosine similarity entropy is an ablation only if a concavity computation on the 22-drug matrix finds no counterexample; sampling can refute concavity, never prove it. The Gaussian kernel is excluded.
- **C as input.** Intersection-union test (proposal 2): the H_C agent must beat drug-entropy SAC both with and without C as input, and equivalence needs TOST against both. The SAC-with-C arm reverses the scope item "no true C in the networks" for that comparator. That reversal is an owner act.
- **Comparators.**
  - Co-primary, with step-down correction: MinRed SAC with a learned inverse model, from its published configuration, and hard-class pushforward SAC on the target groups.
  - Also required: the α = 0 control.
  - Ablations: the misspecified-C ablation (proposal 2), no-C inputs, the cosine entropy (dev only), two-level SAC and the class-size-corrected variant.
  - PPO-entropy goes in the completeness batch.
- **The reserve critic path is deleted** (proposal 2): `ocmdp/agents/lagrangian.py`, the reserve critic and the ξ term. It is inert in every logged window, and removing it drops scope, so it is an owner act.

### 5. Rung 1

- **Tracing E3.** "Traced" is defined before any of our values exist: the one code path and arguments that print the panel holding 1.88. If it is not traced:
  - E3 drops and L3 becomes reported-only;
  - the gate is E2, L1 and L2 at their exact cut-offs, the discrimination controls, and a simulator check, E3′: the exact random expectation against our Monte Carlo within 3 SE.
- **Gate-0 additions.**
  - **Greedy.** Greedy's exact L1 status is computed first. If greedy passes, the sentence adds "which a myopic rule also passes". If it fails exactly, it becomes a third negative control that the empirical pipeline must also fail.
  - **Positive control.** The MDP policy must pass L1, and its exact value must be at least that of every cycle and random; failing either means the gate is broken, so stop.
  - **Starved budget.** A DQN with 10% of the gradient updates must fail L1 and L2; otherwise the sentence drops "learned".
- **Rung-3 estimand (proposal 2).**
  - θ is the mean over seeds of [(Y_OC − Y_SAC) at κ* minus the same at κ = 0], where Y is the published metric, mean population fitness.
  - Each agent's temperature is selected once on val and held across κ.
  - κ* is the smallest grid κ at which the coupled MDP optimum differs from the uncoupled one in at least 10% of reachable states, fixed without any agent.
  - n is sized from a dev A/A test.

### 6. Collateral sensitivity

- It is a stated limitation, not a core change.
- Before the B2 preregistration, report the floored share of absolute mass and which drugs carry it.
- If that share exceeds 20%, the instance is named "the cross-resistance part of the 22-drug matrix".
- Every B2 sentence carries "with collateral sensitivity floored at zero".

### 7. Sequencing

- **Claim A alongside rung 1.** Claim A's planner work runs alongside rung 1. Its test launch waits for rung 1's verdict, whichever way it goes. This needs an owner edit to `CLAUDE.md`.
- **The blind derivation restarts now, first in the queue,** once the held-out cell list is hashed.
  - The owner sets a spend allowance.
  - Each delivery stage is hashed on delivery.
  - A stage not delivered is "not predicted" and is never filled in after data exist (proposal 2).
  - Earlier partial notes are reused only after an audit.
- **What the derivation agent may read and compute.** It works in a sanitised export with no `.git`.
  - May read: the source of `ocmdp/core/`, `ocmdp/envs/` and `ocmdp/theory/`; the contrast definitions and the scorer's format; the held-out list and the noise grid; public literature.
  - May compute: closed forms and deterministic noise-0 quantities of its own reduced models (proposal 1's limit).
  - May not read: `runs/`, `figures/`, `decisions/`, `arena/`, the plan document, the README results, `provenance/`, launch records, the ledger, `tests/`, `inbox/`, memory notes or git history.
  - May not run any stochastic simulation, and may not import the environments at run time.

## Where the proposals disagreed

- **"Not tested" threshold.** A share-of-objective cut of 0.05 (proposal 3) or a bias bound against m (proposals 1 and 2). The bound is adopted.
- **Margin.** Fixed at 0.01 (proposals 1 and 3) or set by substance at 0.1 × the gap between uniform and the reference (proposal 2). 0.01 is adopted, with the gap reported beside it.
- **DP placement.** `ocmdp/baselines/` before the freeze (proposal 3) or `experiments/` pinned by hash (proposals 1 and 2). This is left to the owner.
- **All-zero rows of C.** Stop (proposals 1 and 2) or assign one-hot to the target group (proposal 3, adopted).

## Not satisfied by any proposal

- **Power.** None sizes B2 at K = 22 or the rung-3 interaction from measured spreads; all defer to dev pilots.
- **Comparator list.** None keeps the full required list from the scope outcome. Mechanism-level SAC, two-level SAC and the class-size-corrected variant are demoted to ablations, and none names that demotion as an owner act. Dropping a comparator is a stop item, so **the owner must confirm the demotions**.
- **Argmax evaluation.** Evaluation takes the most probable drug, while every not-tested threshold bounds the stochastic soft-optimal policy. None shows that an equivalence verdict on argmax evaluation cannot fail below the threshold.
- **Margin.** None ties the margin to clinical relevance and also caps it.

## Convergence

All three proposals chose the mean for selection, never excluded non-learning runs, chose expected failure rate as primary, chose the row-normalised pushforward, and treated collateral sensitivity as a limitation. Identical briefs produce correlated answers, so this agreement is weak evidence.

## Owner acts, with the exact change

- `CLAUDE.md`: selection by the mean, and the claims-ladder sentence on Claim A's timing (`claude-md.patch`).
- `seeds.yaml`: test tier `[1000, 1299]` (`seeds-test-tier.patch`).
- **Approve:**
  - the SAC-with-C comparator arm for B2;
  - deleting the reserve-critic path;
  - the comparator demotions listed above;
  - the DP's placement;
  - the held-out cell list;
  - the derivation's spend allowance.
