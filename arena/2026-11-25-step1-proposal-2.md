# Proposal 2: readiness of operations step 1

## Decision 1: order against rung 1

**Reading.**
- Step 1 as specified (fixed temperature 0, both agents) is not "new method work" under the rung-1 sentence.
- The method's only new content is the entropy term, and at α = 0 that term has no weight. OC-SAC and SAC are therefore one computation, which step 1(a) checks bitwise.
- The no-C change below is input plumbing shared by both arms.

**Timing.**
- Step 1 may run now and waits for neither Gate 0 nor the verdict. Gate 0 checks the Weaver environment and shares no code path with step 1.
- On any day when both are ready, rung-1 launches go first in the budget.

**May be concluded before the verdict:**
- the defect checks of decision 3, step 1 ((a), (b), shared process per pair, canary, identical baselines). Bugs they expose are fixed under the usual authority;
- L* and the settling verdict, written as provisional values in `decisions/` with the report names.

**May not be concluded or used before the verdict:**
- L* in any val launch, preregistration or step 2. Step 2's nonzero temperatures are the method's content;
- any g value, learned count or agent-versus-planner number in the README, a draft or a table;
- anything about OC-SAC against SAC, which are identical by construction, so any difference is a bug;
- anything as evidence for Claim A, B1 or B2.

**After the verdict:**
- If rung 1 reproduces, the provisional values become usable without a rerun.
- If it fails, the ladder stops, and the step-1 reports stay as diagnostics.
- Step 1 is rerun only if the training path has changed, which every later launch checks through the arithmetic canary (decision 5).

**Owner act:** confirm this reading. If it is declined, step 1 waits for rung 1's verdict and nothing else changes.

## Decision 2: which configuration step 1 qualifies

**(a) Settling launch first?** No. It is folded into step 1: every step-1 launch is a settling launch at one length (both conditions, PairedDrugs and an unequal instance, 10 dev seeds). The temperature-dependent half of the scope question is read from step 2, run in every condition S2 keeps.

**(b) Conditions.** Both:
- **c-given:** rows of C as embeddings, plus `MechanismObservation`.
- **no-c:** one-hot embeddings, plus the raw drug-indexed observation with the pending stack asserted.

**Rule S2, settling (fixed now).**
- Read per instance, at the longest length run, over all seeds run.
- no-c "does not learn" on an instance if its learned count k ≤ k_max(n). k_max(n) is the largest k whose two-sided 95% Clopper–Pearson upper bound on the learned share is below 0.5; k_max(10) = 1, and k_max(30) is computed in code from n alone.
- **Flip:** no-c does not learn on some instance while c-given learns in at least ⌈n/2⌉ seeds there. This reverses scope item 4, so it stops for the owner.
- **NOT_LEARNING:** neither condition reaches ⌈n/2⌉ on an instance. Stop.

**(c) Who may use L\*.**
- L* is the decision budget for every learned agent on the toy family (all entropy variants, the α = 0 control, SAC-with-C), whatever its condition.
- All arms share one L* = the maximum over retained cells, because B2's intersection-union test compares c-given and no-c arms directly.
- Comparators trained by other code (PPO, DQN, MinRed with a learned inverse model) get the same budget for parity. Convergence is shown only for the SAC family, and reports say so.
- L* is not qualified at σ > 0, at temperatures above 0, or for K > 20, which includes Oz at K = 22. B2's preregistration either runs S1 there on dev or states L* as unqualified.
- Rungs 2 and 3 use the published 500-episode budget.

**(d) Minimal code change.** Nothing owner-gated.
- `ocmdp/agents/oc_sac.py`, `SACConfig`: new field `drug_embedding: str = "mechanism"`, accepting "mechanism" or "one_hot".
- `train`:
  - embeddings are C if "mechanism", else `torch.eye(n_drugs)`;
  - `assignment` is unchanged;
  - the default is unchanged, so c-given hashes stay bit-identical. A test asserts the T1 hashes are unchanged.
- `experiments/train_agent.py`:
  - `CONDITIONS = {"c-given": ("mechanism", "mechanism"), "no-c": ("drug", "one_hot")}`;
  - `make_env(..., inputs="mechanism")`: "drug" returns the unwrapped `ToyOCMDPEnv`.
- New `assert_pending_occupancy(env)`:
  - deep-copies, resets, gives drug 0 for one quarter;
  - checks that the trailing pending block (raw, or mapped through `mechanism_map`) equals `state()["pending"]` front-padded, and is nonzero when delay ≥ 1;
  - raises `RuntimeError`, so the check survives `python -O`.
- New required flag `--condition {c-given,no-c}` on every tier, carried in the job tuple (one line in `modal_backend`).
- `_parse_overrides` refuses `drug_embedding`.
- Baselines keep the mechanism env.
- The report gains `"condition"`, and `REPORT_SCHEMA` becomes 3.
- Tests:
  - the default reproduces T1;
  - the no-c observation is raw and passes the assertion;
  - a mutated pending block fails;
  - a no-c pair at α = 0 is bitwise equal;
  - `--set drug_embedding` is refused.

## Decision 3: the convergence rule (S1)

**Cells:** {c-given, no-c} × {(2,2), (8,5,4,2,1)}, σ = 0, fixed α = 0, 10 evaluation episodes, L ∈ {12 000, 24 000, 48 000}, plus 96 000 under extension E. Seed set S = {0..9}, or {0..29} after stage 2.

**Definitions.**
- U and R are uniform and `oracle[replans every quarter]` on the report's episodes. They must agree within 1e-12 across all reports of an instance.
- If a DP is built later, the learned counts are recomputed from the logged reports with no relaunch.
- g_s(L) = (U − A_s)/(U − R) for OC-SAC; (a) has passed, so both agents give the same numbers.
- A diverged run gets g := min(0, the lowest non-diverged g in that cell and length). It counts as not learned and is never excluded.
- Learned ⇔ g > 0.5. k_c(L) is the learned count, D_c(L) the diverged count, and a(n) = max(1, ⌊n/10⌋).
- d_s = g_s(2L) − g_s(L), with I the two-sided 95% t-interval and δ = 0.05.
- Classification:
  - EQUIV if −δ < lo and hi < δ;
  - RISING if lo ≥ δ;
  - DECLINING if hi ≤ −δ;
  - INCONCLUSIVE otherwise.
- **A step L→2L passes** iff all four hold:
  1. EQUIV;
  2. |k(2L) − k(L)| ≤ a(n);
  3. k(L) ≥ ⌈n/2⌉ (anti-stagnation);
  4. D(L) ≤ a(n) and D(2L) ≤ a(n).
- **Cell length L*_c:**
  - 12 000 iff 12→24 and 24→48 both pass;
  - otherwise 24 000 iff 24→48 passes;
  - otherwise 48 000 iff E ran and 48→96 passes;
  - otherwise none.
- **L\*** = the max of L*_c over the retained cells. It is unset if any retained cell has none.

**Order.**
1. **DEFECT,** a stop with no fallback, on any of:
   - (a) or (b) fails;
   - the runs of a pair don't share a pid with consecutive `job_in_process`;
   - the canary or fingerprint differs between jobs;
   - baselines differ across reports of an instance;
   - seed sets differ;
   - commit or `code_diff_sha256` differs.
2. **S2 settling.**
3. **Continuation:**
   - L* set → CONVERGED.
   - L* unset, no stage 2 yet, and any comparison INCONCLUSIVE → NEEDS_STAGE_2: seeds 10–29 at all 12 (cell, length) combinations, re-evaluated on n = 30, with stage-1 classifications discarded.
   - Every failing cell fails only by RISING on 24→48 or a count rise → NEEDS_96K: 4 launches.
4. **Stop verdicts:**
   - DECLINING;
   - RISING after E (not converged by 96 000);
   - INCONCLUSIVE at n = 30 (UNDECIDED);
   - EQUIV with k < ⌈n/2⌉ (FLAT_BELOW_MAJORITY, i.e. stagnation);
   - divergence above a(n) (DIVERGING).

**Underpowered output:** `{"verdict": "UNDECIDED", "length": null}` with every interval. It never outputs a default length.
- Derived bounds at zero mean change: n = 10 needs SD(d) < 0.070, and n = 30 needs SD(d) < 0.134.
- SD(d) is not measured.

**Exit codes:** 0 for CONVERGED, 2 for NEEDS_*, 1 for a stop.

**Launches:** stage 1 is 12; stage 2 adds 12; E adds 4; the maximum is 28.

**Code** (`experiments/check_dev_protocol.py`):
- `step1` rewritten, with `_defects`, `_gain_shares` (including diverged runs, with learned measured against the replanner), `_classify(d, n)`, `_step_passes` and `_settle`.
- It accepts several reports per (cell, length) with disjoint seeds.
- `experiments/stats.py` gains `clopper_pearson_upper` and `k_max`.
- Synthetic-report unit tests for stagnation, decline, a wide interval, the 24→48 block, the divergence imputation, the S2 flip and each DEFECT path.

## Decision 4: fallback A/A

- **4.1:**
  - pair jobs keep both arms in one process, so at α = 0 a failed (a) is a DEFECT on every platform;
  - the report names the first differing hash stage;
  - fix, and rerun step 1 in full.
- **4.2:** a deterministic assertion on every report of every tier: both runs of each seed share a pid, have consecutive `job_in_process`, and have equal host blocks.
- **4.3:**
  - on this host, a failed (b) is a DEFECT;
  - a multi-container platform becomes a claim platform only through a repeat-launch A/A: the step-1 12 000 cells, both instances, seeds 0–9, launched twice;
  - pass iff the 90% t-interval of the per-seed difference in g lies inside ±0.05 (TOST) and the learned counts differ by at most 1;
  - this costs 2 launches, only after an owner act choosing such a platform.

## Decision 5: what qualifies a platform

**(a) Every job records** (in `experiments/train_agent.py:host`):
- the Python version and implementation;
- `platform.platform()`;
- the CPU string;
- numpy CPU features and torch SIMD;
- the torch version, git version, cuda, and the SHA-256 of `torch.__config__.show()`;
- the deterministic flag, device and threads;
- the numpy version and the SHA-256 of `show_config`;
- the gymnasium version;
- the SHA-256 of the sorted installed-distribution list, with the full list in the launch record;
- `arithmetic_canary`: four trainings of 2 000 decisions (OC-SAC and SAC, c-given and no-c), PairedDrugs, σ 0, seed 0, at *learned* temperature so the entropy and temperature paths run, under `fork_rng` at one thread. The value is the SHA-256 of the four final hashes;
- `platform_fingerprint` over these fields.

**(b) Val→test match:**
- one `platform_fingerprint` across every job of the cited val reports and of the test launch, equal to the preregistration's `platform_fingerprint:` line;
- `git diff --quiet <val_commit> HEAD -- ocmdp experiments/train_agent.py experiments/stats.py` is empty;
- the driver computes the fingerprint before `record_launch`, so a mismatch cannot burn the test launch;
- each job also checks its own fingerprint against the job tuple;
- the `--platform` label is still recorded but no longer decides anything.

**(c) Role of (a) and (b):** zero-cost defect detectors. They do not qualify a platform.

## Decision 6: the host

- This Windows host, CPU, `2.15.0+cpu`, Python 3.14, `--platform local-win-a`. No CUDA install.
- It can become the claim platform if S1 returns no DEFECT, the fingerprint is recorded, and the venv is frozen from the first val launch to the test launch.
- Any change that alters the canary loses the platform, and the existing rule then applies.
- A later CUDA build is a new platform. L* carries over to it, but the defect checks are rerun.
- **Owner act:** confirm that "rtx" covers CPU training on the RTX host. The 2026-11-24 line is read as naming the machine that then launched runs, which is an interpretation.

## All owner acts

1. Confirm the decision-1 reading.
2. Confirm the decision-6 host reading.
3. Confirm any flip.
4. Decide what follows any S1 stop.
5. Review the `check_test_launch` change and the two new preregistration lines.
6. Commit the code before step-1 launches.
7. Note that schema 3 changes `runs_learned` in `summarise` to the replanner reference, with diverged runs counted. Earlier reports are not rewritten; if any existing table uses `runs_learned`, that is a stop item.

## Against each criterion

1. **It can fail:**
   - stagnation → FLAT_BELOW_MAJORITY;
   - decline → DECLINING;
   - lack of power → UNDECIDED;
   - divergence → DIVERGING plus imputation;
   - non-determinism → DEFECT;
   - no learning without C → flip or NOT_LEARNING.
2. **Fixed before data:** δ comes from the operations outcome. a(n), ⌈n/2⌉ and k_max(n) are functions of n. Stage 2 always covers all 12 combinations.
3. **Hard rules:** only `ocmdp/agents/`, `experiments/` and `tests/` are touched. Diverged runs are counted, and learned uses the replanner.
4. **Cheap:** 12 to 28 launches.
5. **Does not void later work:** one shared L* avoids a rerun when B2 mixes conditions, and a later DP means recomputation, not relaunch.

## Strongest objection

S1 may end UNDECIDED. Training is chaotic, so SD(d) could approach the spread of the g level. Above 0.134, even 30 seeds cannot pass.
- Not fatal: UNDECIDED is true, and defaulting to 48 000 is the gap the review found.
- The decision-1 reading loosens an owner sentence. It is mitigated by quarantine and by owner confirmation.

## What would show this proposal wrong

- SD(d) above the derived bounds.
- Val learning curves at L* still rising over the last quarter at α > 0 or σ > 0.
- Equal canaries with unequal final hashes on the same job.
- No-c passing S2 while step-2 learned-temperature runs in no-c fail to learn.
- Frequent canary mismatches caused by rung-1 work.

## Prior work

- Nagarajan, Warnell and Stone 2018 (arXiv 1809.05676).
- Colas et al. 2018 (arXiv 1806.08295).
- Henderson et al. 2018 (arXiv 1709.06560).
- Patterson et al. JMLR 2024 (arXiv 2304.01315).
- Agarwal et al. 2021.
- Schuirmann 1987.
- Berger 1982 (intersection-union).
- Machado et al. JAIR 2018 (reporting at fixed budgets).
- No prior rule was found that fixes RL training length by equivalence across doublings with an anti-stagnation floor.
