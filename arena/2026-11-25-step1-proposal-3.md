# Proposal 3: readiness of operations step 1

Every threshold here is fixed before any step-1 run exists.

## Defects found while reading

- `check_dev_protocol.py:_runs` drops every diverged run, which breaks the never-excluded constraint.
- `train_agent.py:summarise` sets `runs_learned` against the plan-once `oracle`, not the every-quarter replanner. `step1` reads it.
- Restart agreement was never run at (8,5,4,2,1), and `planner_restarts.py` cannot express unequal classes.
- The local backend records the numpy, torch and gymnasium versions. It does not record the Python version, the BLAS or torch build, or the environment variables that change arithmetic.
- The CLI defaults are `--sigmas 0 0.2` and 5 seeds, so every step-1 command must pass `--sigmas 0` and `--seeds` explicitly.

## Decision 1: order against rung 1

**Rule text.**
> "Work in which entropy carries zero weight, and in which the two agents are verified bitwise identical, is pipeline qualification, not new method work. It may run before rung 1's verdict. Any run in which a temperature other than 0 enters either agent's objective is new method work, and it waits for rung 1's verdict."

- At α = 0 the two agents run the same arithmetic, and (a) checks it. The pushforward entropy is multiplied by 0, so step 1 cannot produce evidence for or against the entropy claim. It is a "diagnostic" in the CLAUDE.md dev-tier list.
- It waits for neither Gate 0 nor the verdict. Step 1 depends on nothing Gate 0 checks.

**May be concluded before the verdict:**
- host status on (a), (b) and the fingerprint;
- a provisional L*, labelled "qualified at temperature 0, noise 0, dev, two instances, pending rung 1";
- whether the α = 0 learner learns without C, which is recorded but not acted on.

**May not be done or concluded before the verdict:**
- no step-2 run, no learned-temperature settling run, and no val run;
- no OC-SAC-versus-SAC sentence;
- no number in a draft;
- no "the agent learns the task" outside `decisions/`;
- L* is never used as evidence for rungs 2 and 3, which use the published setup's own length.

**Owner act:** confirm the reading. If it is declined, step-1 reports wait unused, and nothing is voided.

## Decision 2: which configuration step 1 qualifies

**Settling launch before step 1?** No. It runs after or within step 1. Run first, at an unqualified length, "neither learns without C" could be a length artefact.

**Conditions.** Both, on both instances at all three lengths:
- **C-given:** `--inputs coupling`.
- **No-C:** `--inputs none`, meaning one-hot drug embeddings plus the raw drug-indexed observation.
- The temperature-0 half of the settling evidence is read from these.
- The learned-temperature half runs after the verdict, alongside step 2: both conditions, both instances, 10 dev seeds, c = 0.89, 4 launches, not counted today.

**Settling flip rule:**
- An agent "learns without C" on an instance iff at least half its no-C runs learned at L*, with diverged runs counted as not learned.
- If on either instance neither agent learns without C, both at α = 0 and at the learned temperature, scope item 4 flips to C-given. This is reported to the owner.

**Who may use L\*:**
- **Qualified:** OC-SAC and SAC in the conditions that passed, on (2,2) and (8,5,4,2,1), noise 0, α = 0, default `SACConfig`.
- **Used as the common training budget, labelled "convergence not shown":** every other arm (MinRed, hard-class pushforward, mechanism-level, two-level, class-size-corrected, PPO, DQN), every α ≠ 0, noise > 0, and the 22-drug Oz instance. Equal budget is what parity needs, so later results need no rerun.
- **One L\* for both conditions:** the largest of the per-condition L*s. Because the rule below requires every longer pair to pass, a condition qualified at 12 000 is also qualified at 24 000. If either condition yields none, there is no common L*, and the work stops for the owner.

**Minimal code change (nothing owner-gated):**
1. `ocmdp/agents/oc_sac.py`, `SACConfig`: `drug_embedding: str = "coupling"`, accepting "coupling" or "one_hot". `train` uses `torch.eye(n_drugs)` for "one_hot", and the assignment stays row-normalised C. One-hot followed by the first linear layer is a learned embedding table.
2. `experiments/train_agent.py`, `make_env`:
   - parameter `inputs: str = "coupling"`;
   - "none" returns `MechanismObservation(ToyOCMDPEnv(config), np.eye(n_drugs))`, the wrapper's documented argument;
   - assert in both conditions that `markov_observation` holds and that `observation_space.shape[0] == n_mech + k + 1 + k*(1+delay)`.
3. `main`: `--inputs {coupling,none}` sets both, passed through `instance` and `training_jobs`, and is recorded in `args` and `sac_config`. Baselines keep the default observation; they read `info`, so the g denominator is identical.
4. Tests:
   - the identity map gives the raw observation as float32;
   - the one-hot reachable range is [0, log K] for SAC;
   - the two conditions give different initial hashes.
- **Consequence:** with one-hot embeddings drugs are no longer tied, so T2 and T3 hold only in C-given, which stays B1's arm.

## Decision 3: the convergence rule

**Inputs.**
- Per (condition, instance, length): one report, α = 0, noise 0, dev seeds 0–19 (20 rather than 10, fixed now; same launch count).
- Per seed: g_s(L) = (uniform − agent)/(uniform − every-quarter replanner), using OC-SAC if (a) holds, otherwise the mean of the two agents.
- Learned means g_s > 0.5, computed in the checker against the replanner.
- The checker refuses if an instance's three reports differ in `evaluation_episode_seeds` or in their per-episode uniform or replanner values.

**Precondition.**
- `planner_restarts.py` gains `--class-sizes`, and a test shows it equals `ToyConfig.paired` for equal classes. It runs once at (8,5,4,2,1).
- If `spread_share_of_gain_over_uniform ≥ 0.02`, the result on that instance is REFERENCE NOT ESTABLISHED, and the work stops.

**Rule per pair (L, L′) ∈ {(12k,24k), (24k,48k), (12k,48k)}.**
- d_s = g_s(L′) − g_s(L), and CI = mean(d) ± t_{0.975,19}·SD(d)/√20, a half-width of 0.468·SD.
- First match:
  1. **DIVERGED:** any run of the cell diverged at L or L′. It is counted as not learned and never dropped.
  2. **UNDERPOWERED:** half-width ≥ 0.05, i.e. SD(d) ≥ 0.107. This is a fixed function of the data.
  3. **CONVERGED:** CI ⊂ (−0.05, 0.05) and |learned(L′) − learned(L)| ≤ 2 of 20. This is a two-sided equivalence test.
  4. **IMPROVING:** lower end > 0.
  5. **DECLINING:** upper end < 0.
  6. **NOT SHOWN:** anything else.

**Level clause:** LEARNS(L) iff learned(L) ≥ 10 of 20 on the instance, with diverged runs counted as not learned.

**Choosing L\* per condition:**
- 12 000 iff, on both instances, LEARNS(12k) holds and (12k,24k), (24k,48k) and (12k,48k) are all CONVERGED.
- Otherwise 24 000 iff, on both instances, LEARNS(24k) holds and (24k,48k) is CONVERGED.
- The 48 000 runs are always checked, and the 12k→48k span catches slow drift.
- Requiring every pair to pass is an intersection-union test, so no multiplicity correction is needed.

**Otherwise there is one stop label, by precedence:**
1. DIVERGED;
2. DECLINE;
3. NO LENGTH LEARNS (in no-C, also the α = 0 settling evidence);
4. LEARNS ONLY AT 48 000 (a 96 000 run needs the owner, under a new dated decision);
5. UNDERPOWERED (no L* is adopted, nothing defaults to 48 000, and the owner chooses more seeds, a longer budget or a wider margin in a new dated file);
6. NOT CONVERGED BY 48 000.

**Code:**
- `check_dev_protocol.py`, `step1`, rewritten, with `_gain_shares(report, variant)` (all seeds, diverged ones flagged) and `_pair_verdict(...)`.
- `_runs` keeps diverged runs, with a flag.
- The 0.05 margin is kept from the operations outcome.

## Decision 4: fallback A/A

- **Triage first:** record the first hash stage where the arms split.
  - A split at `initial` breaks the identical-initialisation rule. It is a bug: stop, no fallback.
  - A later split on one host at α = 0 is a defect to diagnose in a new dated decision. The fallback qualifies a platform only after that diagnosis names a cause that is not arm-specific.
- **Equivalence A/A** replaces "interval contains 0":
  - the paired OC-SAC − SAC expected failure rate at L* over all 40 pairs (20 seeds × 2 instances, per condition);
  - equivalent iff the 90% t-interval (TOST at 5%, `stats.equivalent`) lies inside (−0.01, 0.01), the design margin m;
  - a half-width ≥ 0.01 gives A/A UNDERPOWERED, and the platform does not qualify.
- **The Fisher test is replaced by a co-location assertion:** both runs of every pair have the same `pid`, consecutive `job_in_process`, and identical `machine`, `cpu`, `simd`, `numpy_cpu_features`, `torch_threads` and fingerprint. Any split pair fails.
- **If (b) fails:**
  - launch once more at L* on both instances (2 launches per condition);
  - the per-seed difference in g between original and repeat must be equivalent, with its 90% interval inside (−0.05, 0.05);
  - otherwise the platform is NOT REPRODUCIBLE.

## Decision 5: what qualifies a platform

**Record.** `arithmetic_fingerprint()` in `experiments/train_agent.py`, called inside `host()` in every job on every backend. It records:
- `sys.version` and the implementation;
- the machine and CPU;
- `numpy_cpu_features` and torch `get_cpu_capability()`;
- the numpy, torch and gymnasium versions;
- the SHA-256 of `torch.__config__.show()` and of `np.show_config(mode="dicts")`;
- `torch.version.cuda`, the torch threads and the deterministic flag;
- the environment variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `MKL_CBWR`, `ATEN_CPU_CAPABILITY`, `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED`;
- `fingerprint_sha256` over canonical JSON. `platform.platform()` is recorded but kept out of the hash, because the canary tests its effect directly.

**Val→test match:**
1. The `platform:` label.
2. An `arithmetic_sha256:` line in the preregistration, equal to the single fingerprint shared by every job of every val report.
   - A new `platform` subcommand in `check_dev_protocol.py` asserts it is single.
   - A test launch runs one probe job on its backend before `record_launch` and refuses on a mismatch.
3. A canary replay.
   - Every val launch, and every test launch before `record_launch`, also trains 2 000 decisions at learned temperature 0.89, seeds 0 and 1 (dev), both instances, both arms.
   - It records `canary_sha256` over the sorted (instance, variant, seed, final hash).
   - All val reports must agree, and the preregistration carries a `canary_sha256:` line.
   - A test launch refuses unless the canary matches bitwise.
   - Code: `canary_hash(...)` in `train_agent.py`, called from `main`; `check_test_launch` parses both lines.
   - Losing the match means a new dated preregistration that reruns val.

**Do (a) and (b) keep a role?** Yes, demoted.
- (a) is a code-correctness gate: the only arm difference has weight 0.
- (b) is a necessary condition: within-platform prefix determinism.
- Qualification needs (a), (b) and a single fingerprint across all step-1 jobs. The canary carries the val→test link.

## Decision 6: the host

- This machine, CPU torch `2.15.0+cpu`, `local-win-a`. It is the "rtx" host the owner named today, and the 2026-11-24 line ("not from this machine") was written on another machine.
- No Modal profile exists here, and Modal leaves the machine. GPU arithmetic would add nondeterminism.
- It can become the claim platform if it passes (a), (b) and the single-fingerprint check. It stays the claim platform only while the fingerprint and canary hold.
- A CUDA build is a different platform (`local-win-a-cuda`). CPU val never feeds a CUDA test.

## Launch count

- 12 training launches: 2 conditions × 2 instances × {12 000, 24 000, 48 000}, each with `--sigmas 0 --seeds 0 … 19 --set temperature=fixed alpha=0 --platform local-win-a`.
- 1 planner-only command.
- 13 in total. The fallback repeats add up to 4, for at most 17.

## Owner acts

1. Confirm the decision-1 reading.
2. Confirm `local-win-a` as the candidate claim platform, and optionally pause OS and package updates between val and test.
3. Acknowledge the settling flip rule and the common L*. A flip is a stop item.
4. Commit the code changes and the decisions lines.
5. Any CUDA install.

## Against each criterion

1. **It can fail:**
   - stagnation fails the level clause;
   - decline gives DECLINE;
   - divergence gives DIVERGED;
   - lack of power gives UNDERPOWERED;
   - a bad denominator gives REFERENCE NOT ESTABLISHED;
   - an initialisation split, a split pair, or a fingerprint or canary mismatch each fail.
2. **Fixed before data:** the margins (0.05, 0.01), the level (10 of 20), the count clause (2), the seed count (20) and the restart cut (0.02) are all set here. UNDERPOWERED is a fixed function of SD(d).
3. **Hard rules:**
   - no owner-gated path is touched;
   - diverged runs are kept;
   - learned uses the replanner;
   - pairs share everything;
   - the canary uses dev seeds.
4. **Cheap:** 13 launches. Doubling seeds costs compute, not launches.
5. **Does not void later work:** one L* across conditions with every longer pair checked, and unqualified arms are labelled rather than rerun.

## Strongest objection

**Objection 1.** With 20 seeds and a ±0.05 two-sided margin, the result may be UNDERPOWERED: bimodal seed outcomes give a large SD(d).
- Not fatal: UNDERPOWERED is honest, and defaulting to 48 000 would build every claim on an unverified length.

**Objection 2.** The canary lengthens every val launch, by about 4 pair jobs of 2 000 decisions.

## What would show this proposal wrong

- The canary or (b) mismatches with nothing changed.
- The per-condition L*s differ widely, or no-C alone stagnates, so a shared L* hides an efficiency confound.
- Learning curves keep rising after L* at the selected temperatures.
- (a) fails after initialisation on this one host.

## Prior work

- Schuirmann (1987), TOST, already implemented in `experiments/stats.py:equivalent`.
- Henderson et al. (2018) and Agarwal et al. (2021) on seed variance and IQM.
- PyTorch reproducibility notes on CUDA nondeterminism.
- No precedent was found for gating val→test on a bitwise replay of a short training run. These references are named from memory and are not verified.
