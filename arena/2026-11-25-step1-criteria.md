# Arena: readiness of operations step 1 (convergence, bitwise A/A, reproducibility)

Fixed before any proposal is written. The judge scores against these and nothing else. No step-1 run has been launched, and no outcome of any step-1 run exists.

## The decisions

Answer each separately and concretely enough to act on: exact rule text, thresholds, and the code change (file and function) where one is needed.

1. **Order against rung 1.**
   - CLAUDE.md, Claims ladder rung 1: rung 1 "comes before any new method work". Only the planner-only Claim A work is named as running alongside it.
   - Step 1 (arena/2026-11-24-operations-outcome.md §1) picks the training length L* that every later agent run uses.
   - `experiments/repro/` does not exist, and no rung-1 gate (E1–E3, L1–L3) has started.
   - Decide:
     - Is step 1 "new method work" under that sentence? It is dev diagnostics of the existing OC-SAC/SAC code.
     - Must it wait for rung 1's Gate 0 or for its verdict, or may it run now?
     - If it may run now, what may and may not be concluded from it before rung 1's verdict?
2. **Which configuration step 1 qualifies.**
   - The scope outcome (arena/2026-11-23-scope-outcome.md, item 4) says the main experiments put no true C in the networks: one-hot or learned drug embeddings, and the raw drug-indexed observation with the pending stack asserted.
   - It also says the question is "settled first by one dev launch": both conditions, PairedDrugs and one unequal-class instance, 10 dev seeds, with temperatures and within-class entropy logged. "If neither agent learns without C, this flips."
   - In the code now, both agents see C twice:
     - the drug embeddings are the rows of C (`ocmdp/agents/oc_sac.py`, `train`, the `Agent(...)` call);
     - the observation is mapped through C by `MechanismObservation` (`experiments/train_agent.py:make_env`, `ocmdp/envs/wrappers.py`).
   - The CLI cannot express the no-C condition. `ocmdp/envs/` is owner-gated. `ocmdp/agents/` and `experiments/` are not.
   - Decide:
     - Must the settling launch run before step 1?
     - Which condition, or conditions, does step 1 run in?
     - Which arms and instances may use the resulting L*?
     - What minimal code change gives the no-C condition without touching owner-gated paths?
3. **The convergence rule.** Today's rule (operations outcome §1(c); `experiments/check_dev_protocol.py`, `step1`):
   - g(L) = (uniform − agent) / (uniform − every-quarter replanner) per seed.
   - L* is the smaller of 12 000 and 24 000 such that, on every instance, the upper end of the 95% t-interval of mean(g(2L) − g(L)) over seeds is below 0.05, and no more than one fewer run learned at L than at 2L.
   - A review found four gaps:
     - it passes stagnation, for example g ≈ 0.3 at both lengths with no seed learned;
     - it is one-sided, so a large decline passes;
     - 24 000 → 48 000 is never checked when 12 000 → 24 000 passes, although 48 000 is run anyway;
     - with 10 seeds, passing needs mean(d) + 0.715·SD(d) < 0.05, and SD(d) is not measured. The only logged spread of the g level is a pre-merge diagnostic at 5 seeds: SD 0.166 and 0.106 at sigma 0 on PairedDrugs, in `runs/train/train_20261124_155259_10168.json`.
   - Decide:
     - the replacement rule;
     - what happens when no length learns;
     - whether the 48 000 runs are checked;
     - what the rule outputs when it lacks the power to decide.
4. **The fallback A/A rules when (a) or (b) fails** (operations outcome §1, "If (a) or (b) fails").
   - "Interval contains 0" is a null result read as equivalence.
   - The Fisher test of arm against CPU group cannot reject while pair jobs keep both arms on one host.
   - Decide their replacements.
5. **What qualifies a platform.**
   - (a) and (b) are near-certain to pass on one host. At fixed alpha each shorter run is an exact prefix of the longer one: the buffer never wraps, and alpha_lr is unused.
   - The test-launch check compares only a `--platform` label string (`experiments/train_agent.py:check_test_launch`).
   - The local backend's report records `platform.platform()` and the worker count, but not the Python version or the package versions.
   - Decide:
     - what a report must record;
     - what must match between val and test for a test launch to proceed;
     - whether step 1 (a)/(b) still has a role.
6. **The host for step 1.**
   - decisions/2026-11-24.md: "From today, training runs are launched on a second Modal account or a local GPU host, not from this machine."
   - The operations outcome names "the local host, if it passes step 1 (a) and (b)" as the claim platform.
   - Today the owner instructed "trainings on modal or rtx as is required".
   - The machine at hand has an RTX 4080 (unused, because torch is the CPU build), 32 logical cores, Windows 11 and Python 3.14. It has no Modal profile.
   - Installing a CUDA torch build is an install of external code, which is the owner's act.
   - Decide which host runs step 1 and under what label, and whether that host can become the claim platform.

## What it affects downstream

- Every later agent run uses L*.
- Whether val and test of any preregistration share an arithmetic.
- Whether the comparison the paper reports is the C-given configuration or the no-C one.
- Whether the rung-1 ordering in CLAUDE.md is respected.

## What makes one option clearly better

1. **It can fail.** Every rule names a concrete outcome that fails it, including stagnation, decline, lack of power and divergence.
2. **It is fixed before data.** No rule depends on a quantity seen only after step-1 outcomes exist, unless the proposal says how that quantity is fixed blind.
3. **It respects the hard rules.** These are CLAUDE.md, the claims ladder, and the owner-gated paths `ocmdp/eval/`, `ocmdp/envs/`, `ocmdp/core/`, `ocmdp/baselines/`, `ocmdp/theory/`, `seeds.yaml`, `budget.yaml` and `.claude/`. A proposal that needs an owner act names it.
4. **It is cheap.**
   - Today's budget is 40 launches.
   - A pair job at 12 000 decisions is minutes on one core.
   - The host has 32 cores.
   - The proposal states its launch count.
5. **It does not void later work.** No result built on step 1 needs rerunning because of a choice made here.

## Hard constraints

- Diverged runs are never excluded. Each counts as not learned (design outcome §1).
- "Learned" means more than half the reference's gain over uniform. The reference is the DP where it exists, otherwise the every-quarter replanner.
- Agents are compared on paired seeds, holding identical the network init seed, the environment reset sequence, the replay sampling order and the evaluation episodes.
- Test seeds are never touched.
- Nothing is pushed or committed by an agent.

## Evidence that would settle it empirically instead

None before step-1 data exist. The rules must be fixed blind to those data. The one cheap experiment that bears on decision 3, a measured SD(d), is a step-1 outcome itself and cannot be used to set the rule.
