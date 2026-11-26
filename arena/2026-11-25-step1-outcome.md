# Outcome: readiness of operations step 1

**How it was judged.**
- The judge worked blind, against `2026-11-25-step1-criteria.md`.
- **Ranking:** proposal 2 first, proposal 1 a close second, proposal 3 third.
- **Adopted:** proposal 2, with the judge's eleven amendments.

**Why proposal 3 placed last.** It loses on criterion 2: when the result is UNDERPOWERED, it offers the owner more seeds or a wider margin after the data are seen. CLAUDE.md forbids that.

**Code claims the judge confirmed:**
- `_runs` drops diverged runs.
- `summarise.runs_learned` counts against `oracle`, not the every-quarter replanner.
- The local backend already records the numpy, torch and gymnasium versions. The criteria file said it did not.
- The CLI defaults are `--sigmas 0 0.2` and 5 seeds.
- `MechanismObservation` accepts a `(K, k)` map.
- One-hot embeddings untie the drugs in `reachable_entropy_range`.

## Adopted

### 1. Order against rung 1
- **What runs now.** Step 1 at fixed α = 0 is not new method work, and neither is the no-C input plumbing. Step 1 runs now and waits for neither Gate 0 nor rung 1's verdict.
- **Budget priority.** On any day when rung-1 work is ready to launch, rung 1 takes budget first.
- **Before the verdict:**
  - L* and the settling verdict go into `decisions/` as provisional values, with their report names.
  - No α > 0 run, including the settling's temperature half and step 2.
  - No val launch or preregistration uses L*.
  - No g value, learned count or other number goes into the README, a draft, a table or a power calculation.
  - No sentence compares OC-SAC with SAC.
  - Nothing counts as evidence for Claim A, B1 or B2.
- **After the verdict.**
  - The provisional values become usable without a rerun if the canary and the `git diff` over the code paths are unchanged.
  - If rung 1 fails, the ladder stops and the reports stay as diagnostics.
- **Owner act:** confirm this reading of the rung-1 sentence. If it is declined, step 1 waits, and nothing else changes.

### 2. Configuration
- **Settling is folded into step 1.**
  - Its α = 0 half is step 1 itself: both conditions, (2,2) and (8,5,4,2,1), dev seeds 0–9.
  - Its temperature half is a separate run, after rung 1's verdict (amendment 1).
- **"Does not learn" without C.** No-C "does not learn" on an instance when k ≤ k_max(n). k_max(n) is the largest k whose two-sided 95% Clopper–Pearson upper bound on the learned share is below 0.5.
- **Settling outcomes:**
  - **Flip:** on some instance no agent learns in no-C, at α = 0 or at the learned temperature, while some agent reaches ⌈n/2⌉ in C-given. Stop for the owner.
  - **NOT_LEARNING:** neither condition reaches ⌈n/2⌉. Stop.
  - **B1 arm unqualified:** no-C learns but C-given does not. Stop.
- **Who uses L\*.**
  - One L* applies to every learned toy-family agent, in either condition: the maximum over the retained cells.
  - Comparators trained by other code get the same budget. Reports say that convergence was shown only for the SAC family.
  - L* is unqualified at σ > 0, at α > 0 and for K > 20 (Oz). B2's preregistration either runs S1 there on dev or states L* as unqualified.
  - Rungs 2 and 3 use the published budget.
- **Code changes:**
  - `SACConfig.drug_embedding ∈ {"mechanism", "one_hot"}`.
  - `train` uses `torch.eye(n_drugs)` for `one_hot`; the assignment is unchanged.
  - `make_env(..., inputs)`, as in amendment 5.
  - A required `--condition {c-given,no-c}` flag, carried in the job tuple.
  - `_parse_overrides` refuses `drug_embedding`.
  - Baselines keep the mechanism environment.
  - The report gains `condition`, and the schema becomes 3: `runs_learned` is counted against the replanner, with diverged runs counted.
  - T2 and T3 stay C-given only.
  - New tests:
    - the default embedding reproduces the T1 hashes;
    - the no-C observation is the raw one and passes the pending assertion;
    - a mutated pending block fails that assertion;
    - a no-C pair at α = 0 is bitwise equal;
    - `--set drug_embedding` is refused.

### 3. Convergence (S1)
- **Cells:** {c-given, no-c} × {(2,2), (8,5,4,2,1)}, at σ = 0, α = 0 and L ∈ {12k, 24k, 48k}.
- **Per seed:**
  - g_s = (U − A)/(U − R), with R the every-quarter replanner. U and R must agree within 1e-12 across reports.
  - A diverged run gets g = min(0, lowest non-diverged g of that cell and length). It counts as not learned.
  - Learned means g > 0.5.
  - a(n) = max(1, ⌊n/10⌋).
- **Classifying d = g(L′) − g(L).** Take the two-sided 95% t-interval of mean d. This is TOST at 0.025 per side, a Bonferroni split over two looks. With δ = 0.05:
  - EQUIV if the interval lies inside (−δ, δ);
  - RISING if lo ≥ δ;
  - DECLINING if hi ≤ −δ;
  - INCONCLUSIVE otherwise.
- **A step L→L′ passes** when all of these hold:
  - it is EQUIV;
  - |Δk| ≤ a(n);
  - k(L) ≥ ⌈n/2⌉;
  - D ≤ a(n) at both lengths.
- **Cell length L\*_c:**
  - 12k iff 12→24, 24→48 and 12→48 all pass;
  - else 24k iff 24→48 passes;
  - else 48k iff the 96k extension ran and 48→96 passes;
  - else none.
- **Order of checks:**
  1. Reference precondition: restart spread at (8,5,4,2,1) below 0.02 of the gain, else REFERENCE NOT ESTABLISHED.
  2. DEFECT.
  3. S2 settling.
  4. Definite stops: DECLINING, DIVERGING, FLAT_BELOW_MAJORITY. A definite stop never triggers stage 2.
  5. NEEDS_STAGE_2: seeds 10–29 at all lengths, recomputed on n = 30. Only for cells that are INCONCLUSIVE without a definite failure.
  6. NEEDS_96K: 4 launches, only when every failing cell fails by RISING or by a count rise.
  7. Final stops:
     - UNDECIDED at n = 30, output as `{"verdict": "UNDECIDED", "length": null}`;
     - not converged by 96k.
- **Never:** a default length.
- **Exit codes:** 0 converged, 2 NEEDS_*, 1 stop.

### 4. Fallback A/A
- **A failed (a) is a DEFECT on every platform**, because both arms share one process.
  - The report names the first hash stage that differs.
  - Fix it and rerun step 1 in full. No statistical arm A/A replaces it.
- **The Fisher test is replaced by a deterministic assertion** on every report of every tier. Both runs of each seed must:
  - share a pid;
  - have consecutive `job_in_process`;
  - have equal host blocks.
- **A failed (b) on this host is a DEFECT.**
- **Multi-container platforms.** One qualifies only through a repeat-launch A/A that the owner chooses:
  - the step-1 12k cells, seeds 0–9, launched twice;
  - it passes iff the 90% interval of the per-seed difference in g lies inside ±0.05, and the learned counts differ by at most 1.

### 5. Platform qualification
- **Recorded per job:**
  - the Python version and implementation;
  - `platform.platform()`, recorded but not hashed;
  - the CPU string, numpy CPU features and torch SIMD;
  - the torch version, git version and CUDA version, plus the SHA-256 of `torch.__config__.show()`;
  - the deterministic flag, intra-op and interop thread counts;
  - the numpy version and the SHA-256 of `np.show_config(mode="dicts")`;
  - the gymnasium version;
  - the SHA-256 of the sorted installed-distribution list;
  - the environment variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_CBWR`, `ATEN_CPU_CAPABILITY`, `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED`;
  - `arithmetic_canary`.
- **The arithmetic canary:**
  - four trainings of 2 000 decisions each: OC-SAC and SAC, in both conditions, PairedDrugs, σ 0, dev seed 0, learned temperature;
  - cached per process, run under `fork_rng`, and not counted in `job_in_process`;
  - a test asserts that job hashes are unchanged with and without it.
- **`platform_fingerprint`** is a hash over all of the above, except `platform.platform()`.
- **What must match between val and test:**
  - every val job carries one fingerprint, equal to the preregistration's `platform_fingerprint:` line;
  - `git diff --quiet <val_commit> HEAD -- ocmdp experiments/train_agent.py experiments/stats.py` is empty;
  - the driver computes the fingerprint before `record_launch`, with a probe job on non-local backends;
  - each job checks its own fingerprint;
  - the `--platform` label is recorded but decides nothing.
- **(a) and (b)** stay as zero-cost defect detectors.
- **The host qualifies** when:
  - S1 returns no DEFECT;
  - every step-1 job has one fingerprint;
  - a fresh driver process gives the same canary.

### 6. Host
- This machine, on CPU: torch `2.15.0+cpu`, Python 3.14, Windows 11, `--platform local-win-a`. No CUDA install.
- At most 3 concurrent launches, each with `--workers 10`.
- Every command passes `--sigmas 0` and `--seeds` explicitly.
- It may become the claim platform under decision 5. The venv and OS updates are frozen from the first val launch to the test launch.
- A CUDA build would be a different platform (`local-win-a-cuda`), and CPU val never feeds a CUDA test.
- Deleting the reserve-critic path reruns step 1.
- **Owner act:** confirm that "rtx" covers CPU training on this host.
- **Launches:**
  - 12 now, plus 1 planner command;
  - up to 12 more for stage 2, and 4 for the 96k extension;
  - the 4 settling launches come after rung 1's verdict.

## Where the proposals disagreed
The judge's record is in full in the hand-back. The disagreements:
- Agent-level "learns": 6 of 10, ⌈n/2⌉ or 10 of 20.
- Divergence: stop on any diverged run, or a tolerance with imputation.
- Seeds: 10 then 30, or 20 in a single stage.
- What follows 48k: an owner stop, or a pre-specified 96k extension.
- The canary's scope.
- Whether `platform.platform()` goes into the hash.
- Modal at test: refused, a repeat A/A, or a probe.
- The 2026-11-24 host line: asserted, or put to the owner.
- One L*, or one per condition with a per-preregistration maximum.

## Missed by all three
- **L* also sets `alpha_lr`.** `SACConfig.__post_init__` derives it from `total_quarters`: 3e-3 at 12k, 2e-3 at 24k, 1e-3 at 48k. Step 1 at α = 0 cannot qualify that, so L* is unqualified for learned temperatures in a second way, beyond the α > 0 label.
- **(b) says little beyond the first window.** At fixed α a shorter run is a bitwise prefix of a longer one, and nobody uses that as a full-length consistency check. Likewise, a run that diverges at L must also diverge at 2L.
- **Criterion 5 is met by labelling, not by a test.** Transfer to α > 0 and to σ > 0 is still the operations outcome's open item.

## Owner acts
1. Confirm the decision-1 reading.
2. Confirm that this CPU host is "rtx".
3. Commit the code before the step-1 launches, so reports cite a commit.
4. Confirm a flip if S2 calls for one.
5. Rule on any S1 stop.
6. Add the `platform_fingerprint:` and `val_commit:` lines to future preregistrations.
