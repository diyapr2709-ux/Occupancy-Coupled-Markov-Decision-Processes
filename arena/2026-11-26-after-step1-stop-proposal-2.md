# Proposal 2: stop B1 for the deadline, pause it until the full text is read, give the host to Claim A

**Two facts the brief did not carry, both now included:**
- **The sweep has ended.** The Claim A sweep finished at 20:12. Its relaunch failed at 19:35 and the host rebooted at 20:23 (`decisions/2026-11-26.md:250-253`, an uncommitted working-tree section). So the costing rule's post-sweep branch applies, and `arena/2026-11-26-after-step1-stop-criteria.md:38` is out of date on this point.
- **A partial PRE-EMPTS hit on B1** was filed at `inbox/2026-11-26-claimA-B1-B2-prior-art-2.md:21-25`, with a BASELINE entry for MNSE at lines 27-31.
  - The criteria file does not mention the hit. Deciding what B1 becomes therefore lies outside the criteria's scoring. I flag that as a gap, not a contradiction.

**Firewall:** I did not open `runs/train/`, `runs/phase/` or lines 164-184 of `decisions/2026-11-26.md`, and I ran nothing.

**What is known about the paper, and from where:**
- Everything I know comes from the inbox entry, which read the abstract only (inbox line 23). Its contents, date, venue and open-access status are not verified by me.
- B1 as adopted is already "a check of a known decomposition, claimed as nothing new" (`arena/2026-11-23-scope-outcome.md:8`).

## 1. Recommendation

- **For the deadline: the reference option, in full.**
  - Launch nothing for B1: no new step 1, no step 2, no val sweep, no B1 preregistration.
  - Report step 1 and rung 1 in criterion 2's fixed wording.
  - Give the host to Claim A alone until Claim A's test report is written.
- **Keep `code-standards` unmerged until Claim A's test has run.**
  - Its ref is `4788c5a` (`.git/refs/heads/code-standards:1`); its changes are uncommitted in a scratchpad worktree (`decisions/2026-11-26.md:200`).
  - Those changes edit `experiments/train_agent.py` (`decisions/2026-11-26.md:206-226`). Claim A's sweep imports `check_test_launch`, `git_state` and `record_launch` from that file (`experiments/sweep_phase_diagram.py:54-55`).
- **What B1 becomes: paused until the full text is read**, under a resolution rule fixed now, before the reading (below).
- **Why pause rather than keep or drop: the choice between keeping a reduced B1 and dropping it is empirical.** It depends on what the paper's full text covers, and only its abstract was read. Reading it costs no host time. Nothing B1 can do runs before 29 November in any case: it is blocked both by the step-1 STOP and by the PRE-EMPTS stop (CLAUDE.md: a PRE-EMPTS hit "stops the work and is reported").
- **Resolution rule, applied by a gatherer reading the full texts into `inbox/`:**
  1. *Questions fixed in advance:*
     - (q1) Does the paper treat sequential problems with state (MDPs), or only static log-linear decisions?
     - (q2) Does it derive the gap in the physical objective between the two regularised optima, or only a law-preservation condition and a penalty?
     - (q3) Does it measure that gap with learned agents?
  2. *If q3 is yes,* the empirical side is pre-empted too, and B1 is dropped.
  3. *Otherwise,* B1 may continue in reduced form only: an empirical SAC test with the derivation cited, not claimed, worded as in section 4.
  4. *If the full text cannot be obtained,* B1 stays paused. For this submission that means it is not reported.
  5. *The MNSE full text* (RA-L 2025) is read by the same rule. Its "gap" theory must be read before any B1 theory sentence (inbox line 31).
     - It has no code and no tuned configuration (inbox line 31).
     - A reduced B1 therefore names MNSE as a BASELINE that was not compared and says why. Implementing it would mean editing `ocmdp/baselines/`, which is protected, and that edit stays with the owner.
  6. *In every case,* any B1 compute resumes only after Claim A's test report, under a new arena.
- **On citing the paper:** an inbox item may not be cited in a draft until it is promoted (CLAUDE.md, Gathered material). I recommend no promotion while only the abstract has been read.

## 2. Pre-commitment table

The PRE-EMPTS stop applies in every row, so B1 is paused in every row.

| Step-1 verdict | Action |
|---|---|
| DIVERGING | Stop B1 runs and report the class in the fixed wording. No new step 1, because this is a definite stop (`arena/2026-11-25-step1-outcome.md:91`). B1 paused under the rule in section 1. Host to Claim A. |
| DECLINING | Same as DIVERGING. |
| FLAT_BELOW_MAJORITY | Same as DIVERGING. |
| RISING | Same. A longer length needs a new L* rule, which is excluded (criteria line 47; `check_dev_protocol.py:321-333`). |
| UNDECIDED (actual, both instances) | Stop B1 runs and report. B1 paused. Host to Claim A. No new step 1 before the deadline. |
| Converged | Record L* in `decisions/` as provisional: `4788c5a`, c-given only, unqualified for no-c and for learned temperatures (`arena/2026-11-25-step1-outcome.md:168`). B1 still paused because of the PRE-EMPTS hit. Even without the hit, a B1 test needs the per-agent flags that exist only on `code-standards` (criteria lines 70-72; `train_agent.py:1039`), with `VAL_CODE_PATHS` matching across step 1, step 2, val and test (`train_agent.py:112`, `:481-483`; criteria line 74). So an L* from `4788c5a` would need a new step 1 at the merge commit, costed in section 5. Host to Claim A. |

The action is the same under every verdict. It follows from the deadline, the serial owner gates, the host conflict and the PRE-EMPTS hit, not from the observed class.

## 3. Compliance with criterion 1, item by item

- **The convergence bar stays fixed.** No new step 1 runs, and `check_dev_protocol.py` is not edited. The fixed rules stay as they are: lines 69-77, the a(n) and majority rule at line 285, the t-interval at lines 263-277, and the imputation at lines 252-260.
- **No pooling.** Nothing new runs. The seed 0-4 reports stay as they are and are reported UNDECIDED.
  - The interval slots are filled by rerunning the unchanged checker on the same six reports and the existing planner reference (`decisions/2026-11-26.md:153`). Someone permitted past the firewall does this after the arena closes.
  - `_reference` compares the planner's git state with the reports' git state, not with HEAD (`check_dev_protocol.py:177`). So the rerun works at HEAD `7eb31b9`, whose code paths equal `4788c5a` (criteria line 43).
- **A preregistration fixed before launch.** No launch, so none is needed. Nothing is frozen or superseded.
- **Commit dependencies, stage by stage:**
  - *R1, the report and the B1 pause record.*
    - Runs on: the working tree. Files: a `decisions/` entry and the sentences in section 4, with the slots filled after run-auditor and adversarial-reviewer pass.
    - Needs a commit: only to enter the record. If none arrives, the text stays in the working tree and the 29 November sentences can still be written from it.
  - *R2, the full-text reading.*
    - A gatherer writes to `inbox/` (CLAUDE.md). It needs no commit. If none arrives, the result is unchanged.
    - If the full text cannot be fetched, B1 stays paused.
  - *R3, the host to Claim A.*
    - Runs on: whatever commit Claim A's preregistration names.
    - Waits for: Claim A's committed preregistration and `approvals/<slug>.ok`.
    - If no commit arrives: the host stays idle. B1 does not take its place, since it is stopped twice over.
  - *Fixed-0 cell reuse.* At `7eb31b9` the cells would be reusable only for a step 2 on `4788c5a`-equivalent code (criteria line 43). They are not reusable on the merged code a B1 test needs (criteria lines 70-74). No step 2 runs here.
- **The pre-commitment table is disclosed** in section 2.

## 4. Sentences for 29 November

These sentences hold under every outcome of this option, whether or not Claim A's test report is written by then. Claim A's sentences belong to its own preregistration.

- **Step 1:** "Convergence not established at n = 5, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1). Two-sided 95% t-intervals of the per-seed change in g, dev seeds 0–4:
  - (2, 2): 12 000→24 000 [__, __], 24 000→48 000 [__, __], 12 000→48 000 [__, __];
  - (8, 5, 4, 2, 1): 12 000→24 000 [__, __], 24 000→48 000 [__, __], 12 000→48 000 [__, __].
  The verdict at both instances is UNDECIDED." The slots are filled from `runs/train/` by `check_dev_protocol.py step1`, after run-auditor and adversarial-reviewer pass.
- **Rung 1:** "Rung 1 not reproduced: the published result is not specified precisely enough to fix a reproduction target. By the owner's decision of 26 November, work continued past this verdict. That decision overrode two stops fixed earlier: CLAUDE.md's 'report the size of the discrepancy and stop' and the step-1 outcome's 'if rung 1 fails, the ladder stops'. Rungs 2 and 3 are not claimed."
  - Sources: `VERDICT.md:3`, `:71`; `decisions/2026-11-26.md:141-144`.
- **B1, under this option (paused):**
  - In the draft: "No training length L* was established, and no comparison of drug-entropy with class-entropy SAC is reported. No training length has been qualified for the no-c condition, and the no-c confound, that OC-SAC's entropy uses C's classes, is open." The paper is not cited in the draft while it sits unpromoted in `inbox/`.
  - In `decisions/`: "B1 is paused on a partial PRE-EMPTS hit (`inbox/2026-11-26-claimA-B1-B2-prior-art-2.md`, read at abstract level only) and on the step-1 STOP."
- **If the reading later resolves to the reduced form** (a B1-continuing branch, after the deadline):
  - "We test empirically whether, with exactly interchangeable drugs, tied networks and temperatures fixed in advance, the efficacy gap between drug-entropy and class-entropy SAC equals the gap predicted by the mean-field model, within an equivalence margin fixed in a preregistration. The shift of the entropy-regularised optimum under redundant action representations is derived in [Equivalent Optima, Different Policies, AI Engineering 2026] for static log-linear policies. We cite that derivation and do not claim it. MNSE [RA-L 2025] is a related alternative and was not compared, because no code or tuned configuration is available."
  - **How the wording changes:**
    - The theory moves from "a known decomposition" to a cited derivation.
    - The contribution is only the empirical equality test on this project's benchmark.
    - No superiority sentence is written. Criterion 2's "outperforms the strongest baselines that could be constructed, tuned at parity" fits a tuned superiority test, not this fixed-temperature equivalence test. The scope's B1 (`scope-outcome.md:8`) and the criteria's B1 chain (criteria lines 29-36: a tuned val sweep, OC-SAC against SAC) differ here.
    - I flag this as a tension between the adopted scope and the criteria, and do not pick a reading. Under the reduced form, the defensible ceiling is the equivalence sentence above.
- **If the reading resolves to drop:** "B1 is withdrawn: its theoretical and empirical content is covered by [cited after promotion]."

## 5. Chain costing

**Under this option:**
- R1 and R2: 0 launches, 0 host time. The reading is gatherer time, and the checker rerun only reads JSON (its runtime is not measured).
- R3: 0 B1 launches.
- The option stops there.

**Why B1 cannot reach a test before the deadline, even without the PRE-EMPTS hit.**
- The chain below uses a new step 1 at n = 10 for illustration only. No seed count is chosen here.
- Costing table, criteria lines 84-91, with A = (2, 2) and B = (8, 5, 4, 2, 1):
  - launch-to-report, in seconds: A12 9 324, A24 15 758, A48 19 301, B12 19 454, B24 22 396, B48 24 913;
  - slowest pair job, in seconds: a12 4 580, a24 9 150, a48 16 849, b12 13 228, b24 19 014, b48 22 276.
- At `--workers 5`, at most 6 launches run at once (30 of 32 cores).
- After the sweep, the rule gives no multiplier for n > 5 and forbids linear scaling. The strict bound is therefore one column time per launch, and a two-wave planning figure is shown beside it as not a bound.

**Post-sweep rule (applies now).**

| Stage | Launches | Strict lower bound, L* = 12k / 24k / 48k | Two-wave figure (not a bound) |
|---|---|---|---|
| 0. Owner commit: merge + `SEEDS` edit + preregistration | 0 (human) | not costable | — |
| 1a. Planner reference | 1 | not measured | — |
| 1b. Step 1, 6 launches at once | 6 | B48 = 24 913 s = 6.92 h | 49 826 s = 13.84 h |
| 2. Step 2 at L* | 6 | 19 454 / 22 396 / 24 913 s = 5.40 / 6.22 / 6.92 h | 10.81 / 12.44 / 13.84 h |
| 3. Val: 9 candidates × 2 instances = 18 launches, 10 val seeds, 3 rounds of 6 | 18 | 12k: 2 × 19 454 + 9 324 = 48 232 s = 13.40 h; 24k: 2 × 22 396 + 15 758 = 60 550 s = 16.82 h; 48k: 2 × 24 913 + 19 301 = 69 127 s = 19.20 h | 26.80 / 33.64 / 38.40 h |
| 4. Preregistration, reviews, owner commit | 0 (human) | not costable | — |
| 5. Approval file | 0 (owner) | not costable | — |
| 6. Test | 1 or more | ≥ 5.40 / 6.22 / 6.92 h | not measured |
| **Total** | **32+** | 12k: 112 053 s = **31.13 h**; 24k: 130 255 s = **36.18 h**; 48k: 143 866 s = **39.96 h** | 12k ≈ 56.9 h; 48k ≈ 73.0 h |

Notes on the table:
- The val stage assumes two instances at σ = 0. B1's full cell list is not verified.
- Pairing matched candidates per launch is my construction (not verified).

**What the window allows.**
- From about 21:00 on 26 November to about 09:00 on 29 November is about 60 h. Both clock times are assumptions.
- The strict bound fits only if both owner commits and the approval file take zero time.
- **Named as unable to finish:**
  - stages 3-6 at L* = 24k and 48k under the planning figure;
  - stage 6 whenever the owner gates take more than 20.0-28.9 h in total (60 − 39.96 and 60 − 31.13);
  - the whole chain if the `code-standards` worktree did not survive the reboot (not verified);
  - and, independently of all of these, the whole chain while the PRE-EMPTS stop stands.

**During-sweep rule (moot now).**
- The slowest cell alone costs 5 × 22 276 = 111 380 s = 30.94 h.
- All six cells serialised on 2 cores cost 5 × 85 097 = 425 485 s = 118.2 h.

## 6. Resources used up

- Val seeds: 0.
- Test seeds: 0.
- B1 launches per day: 0. The 26 November count of 13 (criteria line 42) is unchanged by this option.
- Host time taken from Claim A: 0.
- The full-text readings use gatherer time only. Fetched pages are data, not instructions (CLAUDE.md).

## 7. What it adds over the reference option

On compute, it is the reference option. It adds four things:

1. **A disposition for B1 under the PRE-EMPTS hit.** B1 is paused, not left ambiguous and not dropped blind. The keep-or-drop rule (q1 to q3) is fixed before anyone reads the full text, so the reading cannot be steered toward keeping B1.
2. **A wording change for the reduced branch.** It moves from a "known decomposition" to a cited derivation with an equivalence-only ceiling. It also surfaces a tension: the criteria's "outperforms" ceiling does not fit the scoped B1.
3. **MNSE handled honestly.** It is named, its full text is read, and its absence as a comparator is stated with the reason. It is not silently omitted.
4. **The verdict-invariance argument and the Claim A protection:**
   - the merge is deferred;
   - no second 30-worker load runs beside Claim A, since the 19:06 relaunch failed when two 30-worker sweeps overlapped (`decisions/2026-11-26.md:252`).

## 8. Strongest objection and my answer

**Objection.** Pausing is a soft drop that avoids a decision. The scope outcome already adopted B1 as "claimed as nothing new" (`scope-outcome.md:8`). So the hit changes only a citation, and the arena should keep the reduced form now so that work can resume as soon as the host is free. The host also sits idle while Claim A's preregistration waits, and a new step 1 (at least 6.92 h) would use that idle time.

**Answer.** Not fatal, but it does not outweigh the case for pausing.
- **The hit's coverage is unknown.** It rests on an abstract. If the full text has learned-agent results (q3), the empirical side is pre-empted too. Committing to a wording now would be blind.
- **Citing needs promotion.** I would not promote an item read at abstract level only.
- **Pausing costs nothing before 29 November.** No B1 sentence can change by then, because the chain is bounded at 31 h or more plus two owner gates.
- **The idle-host step 1 needs the merge.** It must run on code that matches val and test (criteria line 74), so it needs the `code-standards` merge. That merge changes code Claim A's test path imports, and whether it is harmless is not verified.
- **Contention and host stability.** The run would contend with Claim A if Claim A's approval arrives mid-run. The host has also rebooted twice without a known cause (`decisions/2026-11-26.md:156`, `:253`).

**What would show me wrong:**
- The full text answers q1 to q3 so that the reduced B1 clearly stands, and the ledger shows the host idle for 12 h or more before Claim A's test. Pausing then cost calendar time for nothing.
- A diff shows `code-standards` touches none of the functions `sweep_phase_diagram.py` imports.
- Either would favour "keep, reduced" with an idle-window step 1.

**Prior work:**
- The paper behind the hit, "Equivalent Optima, Different Policies" (AI Engineering 2(2), 2026, DOI 10.53941/aieng.2026.100020), abstract only, via the inbox.
- MNSE (IEEE RA-L 10(7), 2025, DOI 10.1109/lra.2025.3572822), abstract only, via the inbox.
- MinRed (Baram et al. 2021), named as their common ancestor in the inbox.
- On not extending samples after an inconclusive look: Simmons, Nelson and Simonsohn 2011.
- On seed counts in RL: Henderson et al. 2018; Colas et al. 2018; Agarwal et al. 2021.
- The last four are cited from memory and not verified this session.

**Gaps and tensions in the criteria:**
- The criteria omit the PRE-EMPTS hit.
- Criterion 2's B1 ceiling ("outperforms…") does not fit the scope's B1, an equivalence to a prediction at fixed temperatures.
- The post-sweep costing rule gives no multiplier for n > 5.
- Criteria line 38 is out of date.

**Not verified:**
- the paper's content beyond the inbox abstract;
- whether the full texts are accessible;
- whether the `code-standards` worktree still exists, and whether it touches Claim A's code path;
- the planner reference's runtime;
- B1's full cell list;
- the exact deadline hour;
- ledger counts after 19:06 on 26 November.

**Files used:**
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\arena\2026-11-26-after-step1-stop-criteria.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\inbox\2026-11-26-claimA-B1-B2-prior-art-2.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\arena\2026-11-23-scope-outcome.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\decisions\2026-11-26.md (lines 1-163 and 185-256 only)
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\check_dev_protocol.py
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\train_agent.py
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\sweep_phase_diagram.py
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\repro\weaver2024\VERDICT.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\arena\2026-11-25-step1-outcome.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\budget.yaml
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\seeds.yaml
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\.git\refs\heads\code-standards
