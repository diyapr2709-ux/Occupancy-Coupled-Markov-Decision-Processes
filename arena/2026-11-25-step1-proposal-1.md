# Proposal 1: readiness of operations step 1

All six decisions in `arena/2026-11-25-step1-criteria.md` are answered here. Every threshold is fixed without step-1 data. A quantity that depends on a step-1 outcome (stage 2, the settling length) does so only through a rule written here.

**Launch count:**
- 12 now.
- At most 12 more today, if stage 2 of decision 3 triggers.
- At most 4 later, if the scope-flip question needs them.
- Maximum 28, against a budget of 40.

## Decision 1: order against rung 1

**Rule text.**
> "New method work" is any run or code change that alters an agent's objective, or that produces a performance comparison between two methods or two entropy spaces.
> - Step 1 at fixed temperature 0 is not new method work. At α = 0 the two agents are the same program, and (a) asserts this bit for bit, so there is no method contrast to produce.
> - The no-C input path (decision 2) is not new method work. It implements an input condition the scope outcome already adopted, and it changes neither objective.
> - Any run at α > 0 is new method work for this purpose and waits for rung 1's verdict. This includes the settling launch's learned-temperature arm and the step-2 temperature dev check.

**Timing.**
- Step 1 runs now and waits for neither Gate 0 nor the verdict.
- Gate 0 tests the Weaver reimplementation, which shares no code, environment or learner with the toy agents.
- Priority rule: on any day when rung-1 work is ready to launch, it takes budget before step 1.

**May be concluded before the verdict:**
- L*_C and L*_noC as dev training-length settings;
- code-isolation status at full length, i.e. (a);
- host qualification status (decision 5);
- the no-C qualification, in its "confirms no-C" direction only.

**May not be concluded before the verdict:**
- any OC-SAC-versus-SAC sentence;
- any step-1 number in a draft, table, README or power calculation;
- any val-tier launch on the toy family, which is rung-4 work;
- any step-2 run.

If rung 1 fails, the ladder stops. Step-1 outputs remain valid dev facts.

## Decision 2: which configuration step 1 qualifies

**Settling launch before step 1?** No. Step 1 runs in both conditions, and its α = 0 arm is the settling launch's α = 0 arm: PairedDrugs and (8,5,4,2,1), dev seeds 0–9, within-class entropy logged, temperature a constant 0.

**Conditions.**
- **C-given:** today's code.
- **No-C:** one-hot drug embeddings and the raw drug-indexed observation, with the pending stack asserted.
- The network shapes differ (embedding width M against K), so each condition gets its own L*.

**Who uses which L\*:**
- L*_noC: OC-SAC and SAC in no-C (the main configuration and the B2 no-C ablation).
- L*_C: the B1 C-given arm and the SAC-with-C comparator.
- Both apply only at noise 0 on the toy family with K ≤ 20.
- These must pass the same decision-3 rule on dev before their val launch: noise above 0, the 22-drug Oz instance, and every other learned comparator.
- **Parity rule:** every arm of one preregistration trains for the same number of decisions, the maximum of its arms' qualified L*. This is fixed before any val launch of that preregistration.

**Settling verdict** (new subcommand `settle` in `experiments/check_dev_protocol.py`):
- An agent *learns* on an instance when at least 6 of its 10 runs learned (g > 0.5 against the every-quarter replanner; a diverged run counts as not learned).
- If, on both instances, the α = 0 agent learns in no-C at L*_noC (or at 48 000 if no L* was issued), **no-C stands**.
- Otherwise the α = 0 data cannot flip the scope on their own. After rung 1's verdict, 4 launches run: learned temperature at c = 0.89, OC-SAC and SAC, 2 conditions × 2 instances, seeds 0–9, at L*_c or 48 000. These read runs-learned only.
- Outcomes:
  - **Flip:** on some instance no agent learns in no-C, but some agent learns in C-given. To the owner as a scope change.
  - **No configuration qualified:** nothing learns in either condition. Stop.
  - **B1 arm unqualified:** no-C learns but C-given does not. Stop, for the owner.

**Minimal code change.** Nothing owner-gated is touched; `MechanismObservation` already accepts a `mechanism_map`.
- `ocmdp/agents/oc_sac.py`, `SACConfig`:
  - new field `inputs: str = "cross_resistance"`;
  - `__post_init__` rejects anything outside {"cross_resistance", "drug_indexed"};
  - the field parses through `--set inputs=drug_indexed`.
- `oc_sac.py`, `train`: `torch.eye(n_drugs)` as embeddings when `inputs == "drug_indexed"`. The assignment is unchanged.
- `experiments/train_agent.py`, `make_env`:
  - new keyword `observation: str = "mechanism"`;
  - `"drug"` returns `MechanismObservation(ToyOCMDPEnv(config), mechanism_map=np.eye(K))`.
- `train_and_evaluate`:
  - build `cfg` before `env` and pass `observation` accordingly;
  - in the drug condition, assert `env.mechanism_map == I`;
  - assert `env.observation_space.shape[0] == n_mech + K*(2 + delay) + 1`.
- `evaluate_baselines` is unchanged, so the denominators are identical across conditions.
- New `tests/test_no_c_inputs.py`:
  - embeddings equal I_K;
  - the observation equals the raw one, with the pending block present;
  - at α = 0 the agents are bitwise identical in no-C.
- T2 and T3 stay C-given only.
- **Order:** these changes and those of decision 5 land before any step-1 launch.

## Decision 3: the convergence rule

Implemented in `experiments/check_dev_protocol.py`, `step1`, rewritten. Applied per condition, per instance, at noise 0, at lengths 12 000, 24 000 and 48 000.

1. **Inputs.**
   - All six reports of a condition must share `git.commit` and `code_diff_sha256`, `inputs`, fixed α = 0 and `evaluation_episode_seeds`.
   - Their uniform and replanner per-episode values must be equal.
   - Otherwise the check refuses.
2. **Per seed.**
   - g_s(L) = (U − A_s(L)) / (U − R), with R the every-quarter replanner.
   - A diverged run gets g = 0 and counts as not learned. It is never dropped.
   - learned_s(L) is g_s(L) > 0.5. The report's `runs_learned` is not used.
   - n_L is the learned count.
   - The agent is OC-SAC if (a) holds. Otherwise both agents are checked and both must pass.
3. **Per doubling (L, 2L).**
   - d_s = g_s(2L) − g_s(L), with a 95% two-sided t-interval (TOST at 0.025 per side, Bonferroni over two looks) and k = 1 per 10 seeds.
   - First matching verdict:
     - **DIVERGED:** any run diverged at L or 2L.
     - **FLAT:** CI ⊂ (−0.05, 0.05) and |n_2L − n_L| ≤ k.
     - **IMPROVING:** lower end > 0, or n_2L − n_L > k.
     - **DECLINING:** upper end < 0, or n_L − n_2L > k.
     - **UNRESOLVED:** none of these, including every half-width ≥ 0.05.
4. **Qualification.**
   - L qualifies when n_L ≥ 6 of 10 and every measured doubling from L up to 48 000 is FLAT, on every instance and every checked agent.
   - L*_c is the smallest qualifying L in {12 000, 24 000}, so the 48 000 runs are always checked.
5. **Outputs:**
   - `L* = 12000` or `L* = 24000`;
   - `DIVERGES`, `DECLINES` or `NOT CONVERGED BY 48000`: stop for the owner, with no longer run without a new dated decision;
   - `DOES NOT LEARN`: FLAT, but n_L < 6 at every length (stagnation);
   - `UNRESOLVED AT n=10`: triggers stage 2;
   - `L* NOT DETERMINED AT n=30`: stop, with no default length.
6. **Stage 2.**
   - Fixed now, run once, only for a condition UNRESOLVED at n = 10.
   - Dev seeds 10–29 at all three lengths and both instances: 6 launches.
   - Recomputed on 30 seeds, with the same 0.025 per side and k = 3, at level n_L ≥ 16 of 30.
   - A definite failure at n = 10 never triggers stage 2.
- **Implied thresholds:**
  - FLAT needs |mean(d)| + 0.715·SD(d) < 0.05 at n = 10;
  - and |mean(d)| + 0.373·SD(d) < 0.05 at n = 30.
- **When no length learns:**
  - `DOES NOT LEARN` is reported as a negative result.
  - No L* is issued for that condition.
  - For no-C it feeds the settling rule.
  - If neither condition learns, stop. The owner may approve a pre-written learned-temperature variant in a new dated decision.

## Decision 4: fallback A/A rules

- **(a) fails on the claim host:**
  - this is a code defect, because at α = 0 the arms are one program in one process;
  - stop, find the first differing hash stage, fix it, and rerun step 1 under a new dated decision;
  - no statistical A/A substitutes for it.
- **(b) fails while (a) passes:** the host is not stable across processes. It does not qualify, and it fails closed.
- **Statistical A/A**, used only to characterise platforms that cannot be bitwise (new `aa` subcommand):
  - the 90% two-sided t-interval of the mean paired OC-SAC − SAC failure rate over 20 pairs at L* must lie inside ±0.01 (m), with |n_OC − n_SAC| ≤ 1 per instance;
  - inside ±0.01 means equivalent; excluding 0 means biased and rejected; anything else means unresolved and not qualified.
- **Fisher replacement:**
  - a design assertion: in every pair, both runs record the same `pid`, consecutive `job_in_process`, identical `host` fields and the same arithmetic canary;
  - a heterogeneity test that can only reject: with at least 2 CPU-feature groups of at least 5 pairs each, Welch's t on the paired differences at 0.05. Non-rejection qualifies nothing.

## Decision 5: what qualifies a platform

**Record per job, in `train_agent.py:host`:**
- `sys.version` and the implementation;
- `torch.__version__`, `torch.version.cuda`, and the SHA-256 of `torch.__config__.show()` and `parallel_info()`;
- the numpy version and the SHA-256 of `np.show_config(mode="dicts")`;
- the gymnasium version;
- `packages_sha256` over all installed distributions;
- the OMP, MKL and OPENBLAS thread variables;
- the interop threads and the deterministic-algorithms flag;
- `platform.platform()`;
- `arithmetic_canary`.

**`arithmetic_canary()`:**
- the final `parameter_sha256` of `train` on `make_env(0.0)`, C-given, PairedDrugs, seed 0, 2 000 decisions, fixed α = 0;
- cached per process and run under `fork_rng`;
- a test asserts that job hashes are unchanged with and without it.

**Val→test match.**
- The preregistration carries `arithmetic: <canary>` and `environment: <packages_sha256>`.
- Every job of every val report must share the canary, `packages_sha256`, the Python version, the torch build, `simd`, `numpy_cpu_features` and `torch_threads == 1`.
- New `check_dev_protocol.py platform`.
- `check_test_launch` computes the canary and digest in the driver before any job and refuses on a mismatch.
- It refuses `--backend modal` for test, because the driver cannot verify a container's arithmetic first. This narrows the operations outcome's Modal fallback.

**Role of (a) and (b).**
- (a) stays as the full-length T1 over 120 pairs, relabelled as a code check.
- (b) stays as the cross-process determinism check.
- The host qualifies when (a) and (b) pass and the canary is identical in every step-1 job and in a fresh driver process.

## Decision 6: the host

- This machine, on CPU, torch `2.15.0+cpu`, Python 3.14, Windows 11, `--platform local-win-a`.
- No CUDA: it would add nondeterminism to (a).
- It can become the claim platform under decision 5. If its environment changes the canary, `check_test_launch` refuses, and a new dated preregistration reruns val.
- Concurrency: 3 launches at once with `--workers 10`.
- Example command:
  `.venv/Scripts/python experiments/train_agent.py --tier dev --platform local-win-a --seeds 0 1 2 3 4 5 6 7 8 9 --sigmas 0 --class-sizes 2 2 --quarters 12000 --workers 10 --set temperature=fixed alpha=0 inputs=drug_indexed`

## Owner acts

1. Adopt this outcome.
2. Optional: commit the code before step 1.
3. Confirm the RTX host, CPU-only, as the candidate claim platform, and freeze `.venv` and OS updates between val and test.
4. Conditional: confirm a scope flip.
5. Conditional: approve any run beyond 48 000 or the learned-temperature variant.
6. Freeze preregistrations with their `arithmetic:` and `environment:` lines.
7. Accept that Modal is no longer eligible for test launches.

## Against each criterion

1. **It can fail:**
   - stagnation → DOES NOT LEARN;
   - decline → DECLINING;
   - late improvement → IMPROVING on 24→48;
   - lack of power → UNRESOLVED, then NOT DETERMINED;
   - divergence → DIVERGED.

   The A/A fails closed, and the canary can mismatch.
2. **Fixed before data:** δ = 0.05 and m = 0.01 come from earlier outcomes, and k, the level and stage 2 are set here. No measured SD enters any rule.
3. **Hard rules:**
   - only `ocmdp/agents/` and `experiments/` are touched;
   - diverged runs are counted;
   - learned uses the replanner;
   - only dev seeds 0–29 are used;
   - all α > 0 runs are deferred.
4. **Cheap:** 12 launches now, up to 28.
5. **Does not void later work:**
   - the code is final before step 1;
   - L* is condition-specific;
   - parity length is fixed per preregistration;
   - val and test are tied by fingerprint.
   - Known exception: deleting the reserve critic reruns step 1.

## Strongest objection

**Objection 1.** Qualifying at α = 0 with a level requirement conflates "converged" with "learns", and it does so at the agent least likely to learn. If α = 0 stagnates, no L* is issued.
- Not fatal: the rule stops rather than issue a wrong length.

**Objection 2.** The decision-1 reading is an interpretation.
- If the owner reads the sentence strictly, step 1 simply waits.

## What would show this proposal wrong

- Stage 2 is still UNRESOLVED at n = 30.
- The canary agrees across an environment change while full-run hashes differ.
- (b) passes, yet a fresh-process rerun differs.
- No-C qualifies at α = 0 but fails step 2's tracking.
- (a) passes, yet T2 and T3 fail at the same commit.

## Prior work

- Nagarajan, Warnell and Stone, arXiv 1809.05676 (deterministic DQN).
- Pham et al., ASE 2020 (variance from implementation nondeterminism).
- Colas, Sigaud and Oudeyer, arXiv 1806.08295 and 1904.06979 (seed power).
- Patterson et al., JMLR 2024, arXiv 2304.01315 (empirical design).
- Schuirmann 1987 (TOST); intersection-union; Bonferroni.
- No prior rule was found that chooses RL training length by a two-sided equivalence rule with a learning-level guard. This rests on one search pass.
