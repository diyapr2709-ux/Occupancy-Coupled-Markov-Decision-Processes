# Ranking of the three proposals: what follows the step-1 stop

**Firewall.** I did not open anything under `runs/train/` or `runs/phase/`, did not run the checker, and did not read `decisions/2026-11-26.md` lines 164–184. I read lines 1–163 and 185–274; line 185 begins "Code standards pass", a different heading. To check two proposals' claims about the `code-standards` merge, I also read the uncommitted worktree's `experiments/check_dev_protocol.py` (lines 60–348) and `experiments/train_agent.py:94-97`. The worktree is at `C:\Users\soumi\AppData\Local\Temp\claude\56127e9efb711afe\3161ddb6-cc92-4b37-95d1-4781a2e2b497\scratchpad\wt-code-standards\`. That is code, not run output.

## 1. Gate (criterion 1)

| Gate item | P1 | P2 | P3 |
|---|---|---|---|
| Convergence bar fixed; seeds and n stated as chosen after UNDECIDED | PASS. Only `SEEDS` changes to 5–14 and every other rule is listed at its `4788c5a` value (§3). The post-verdict choice is stated (§1). It foresees the merge's schema edit inside `_step1_cells` and restores it. | PASS, trivially: no new step 1 and no checker edit (§3). | PASS. Only `SEEDS` changes and the post-verdict choice is stated (§3). But it says "only `SEEDS` changed" without seeing that the merge edits `_step1_cells` (amendment A1). |
| No pooling | PASS. Line 129 rejects other seed sets; 0–4 stay UNDECIDED. | PASS | PASS |
| Preregistration fixed before launch; terminal rule can fail; no extension | PASS, marginally. "Any verdict other than CONVERGED stops B1". But the preregistration's "no further step 1 *before 29 November*" limits the ban in time. | PASS (no launch) | PASS, marginally. "No further step 1 … *under this decision*" is a similar opening. |
| Commit dependencies named, with fixed-0 reuse | PASS. M1, M2, stages 1a/1b/2, and an action if no commit arrives. 0–4 cells not reusable. It omits that the branch's changes are not committed yet, so M1 cannot be a plain merge. | PASS. R1–R3. Reuse stated for `7eb31b9` and for merged code. | PASS. C1 is spelled out: commit the worktree, merge, `SEEDS` edit, preregistration. Reuse stated. |
| Pre-commitment table, all six verdicts | PASS | PASS (identical action in every row, with reasons) | PASS, plus a second table for the new check's verdict. Its "Converged" row is internally inconsistent (see criterion 3). |
| PRE-EMPTS hit addressed (criteria §"Added after…") | Paused, with a DROP / KEEP REDUCED / PAUSED rule | Paused, with rule q1–q3 | Paused, with a drop/keep rule |

All three pass the gate. P1 and P3 pass only marginally on the terminal rule (amendment A4).

## 2. Scores (1–5) and against the reference option

| Criterion | P1 | P2 | P3 |
|---|---|---|---|
| 2. Licensed wording | 3 | 4 | 4 |
| 3. Severity | 4 | 5 | 4 |
| 4. Feasibility | 4 | 4 | 4 |
| 5. Resources | 3 | 5 | 4 |
| 6. Information per compute | 3 | 4 | 4 |

**Criterion 2, wording.**
- **P1 (3).**
  - Good: S1–S3 match the fixed wording, and O0–O4 cover every outcome of the second check. It is the only proposal that discloses the two-look error bound in its sentences (O1).
  - Defect: its KEEP REDUCED sentence (S4) uses "at equal treatment cost". `arena/2026-11-23-scope-outcome.md:9` dropped that phrase because "no file defines a treatment cost". `decisions:218` leaves the primary metric open.
  - Its reduced-B1 sentence also frames B1 as superiority on efficacy. The inbox (line 25) says only the equality of measured and predicted gaps stays novel.
  - Its sentences do not say the slots wait for run-auditor and adversarial-reviewer.
- **P2 (4).**
  - Good: the fixed wording appears in full, and the slots are tied to run-auditor and adversarial-reviewer passing (§4). It is the only proposal to flag a real tension: scope-outcome:8 defines B1 as equivalence to a mean-field prediction, while criterion 2 gives an "outperforms…" ceiling.
  - Defect: its draft sentence for a paused B1 does not say B1 is paused on prior art. That fact goes only into `decisions/`.
  - Its reduced-branch ceiling departs from criterion 2's fixed phrase. It flags this but does not resolve it.
- **P3 (4).**
  - Good: the fixed wording appears in full. O1–O5 cover the deadline outcomes. O5 qualifies L* for no-c, α > 0, σ > 0 and `alpha_lr`. The "strongest baselines" phrase is made conditional on MNSE.
  - Defect: no two-look disclosure, and no auditor/reviewer gate on the slots.

**Criterion 3, severity.**
- **P1 (4).** The bar is fixed and the wording is narrowed honestly. Its "Converged" row reruns all of step 1 at M2, which is correct.
- **P2 (5).** Nothing runs, so nothing is weakened.
- **P3 (4).**
  - Its "Converged at `4788c5a`" row (counterfactual) pairs an L* found at `4788c5a` with step-2 cells at merged code. Criteria:74 requires step 1 itself to run on code that matches val and test.
  - "MNSE … shown not to apply" is a second route to the "strongest" claim without implementing MNSE.

**Criterion 4, feasibility.** All three cost every stage under both rules and name the stages that cannot finish. I rechecked their arithmetic and it is correct (§7).
- **P1.** Its PF figures are labelled as not bounds.
- **P2.** It gives a strict-bound table plus a two-wave figure. It calls the while-sweep rule "moot now", which ignores the 21:30 Claim A rerun (§6).
- **P3.** It calls its n = 10 extension a "floor" while admitting it is not a bound. Its val figure divides slot-seconds by 6, which approaches the forbidden linear scaling.

**Criterion 5, resources.**
- **P1 (3).**
  - It takes no host time from Claim A by gate (b), and it orders owner requests after Claim A's.
  - Its kill-and-relaunch rule fails in practice. Claim A becoming "ready" means a preregistration commit has landed, so any B1 relaunch is a new step with a new planner reference, a 13 h restart.
  - It does not address that the merge edits `train_agent.py` functions that `sweep_phase_diagram.py:54-55` imports.
- **P2 (5).** Zero on every count, and it keeps the merge off until Claim A's test has run, citing that import path (§1).
- **P3 (4).** Condition 3 makes overlap with Claim A impossible, and it never kills a run. But it requests C1 "after the pause lifts", which could put the merge ahead of Claim A's test.

**Criterion 6, information per compute.**
- **P1 (3).** It is the only option that can produce new data before 29 November: one convergence verdict for at least 6.9 h (LB) or about 13 h (PF). It licenses no B1 sentence, and n = 10 has no power basis.
- **P2 (4).** Zero compute, buying a disposition rule.
- **P3 (4).** In practice zero compute before the deadline, since condition 3 waits on Claim A's test. It adds a resume design fixed behind the firewall.

**Against the reference option.**
- P2 is the reference option plus a pause rule and the merge deferral.
- P3 is the reference option plus a pause rule, a resume design fixed blind, and two blockers no other proposal names: the open primary metric and the stale docstring.
- P1 is the reference option plus a pause rule and a conditional pre-deadline step 1, at the cost of two owner commits and up to about 26 host-hours.

## 3. Ranking

1. **P3.**
2. **P2,** a close second.
3. **P1.**

**Why P3 and P2 lead.** Before the deadline both use no host time and no launches, and both leave B1 paused.
- **For P3:** a resume design is fixed by proposers who cannot see the step-1 numbers (criteria:20-24). It also identifies two blockers no one else does: the open primary metric (`decisions:218`) and the stale docstring.
- **For P2:** it is the only proposal that protects Claim A's imported code from the merge. It is also the only one that got the sweep facts right.
- **What would separate them.** On the stated criteria the gap is narrow. The tie-break is how much one values committing now, blind, to a resume design for a line that may be dropped, against not pre-committing. The criteria do not weigh that. Both of P3's main defects (merge timing and the counterfactual row) can be fixed by amendment.

**Why P1 is last.**
- It spends two owner commits and up to about 13 h of host time per stage before 29 November on a prerequisite that licenses no B1 sentence.
- Its reduced-B1 wording reinstates "at equal treatment cost", which the scope outcome dropped.
- Its kill-and-relaunch rule restarts a 13 h step whenever Claim A's preregistration commit lands.
- Its terminal rule is limited in time.

**Where P1 is better than the others.**
- It foresaw the merge's schema edit inside `_step1_cells`. That edit is real (§7).
- It is the only proposal to disclose the two-look bound.

## 4. Points of disagreement

1. **New step 1:**
   - P1 fixes one and may run it before the deadline.
   - P3 fixes one but runs it only after Claim A's test or Claim A's own stop.
   - P2 fixes none, and any B1 compute waits for a new arena after Claim A's test.
2. **Cutoff:**
   - P1 has a launch cutoff of 12:00 on 28 November, and the full-text read must also be done by then.
   - P3 has an implied latest launch time of about 18:53 on 28 November.
   - P2 has none.
3. **Killing B1 when Claim A needs the host:** P1 kills and relaunches; P3 never kills.
4. **Merge timing:**
   - P2 defers it until after Claim A's test.
   - P1 makes it a precondition for launch, with no tie to Claim A.
   - P3 requests it once the pause lifts.
5. **Owner commits:**
   - P1 asks for two (M1, M2) and treats M1 as a plain merge.
   - P3 asks for one (C1), including first committing the worktree.
   - P2 asks for none.
6. **What triggers DROP:**
   - P1: learned sequential RL agents comparing action and class entropy under redundancy, in that paper.
   - P2: q1–q3, dropping only on q3 (learned agents measuring the gap); MNSE is read under the same rule.
   - P3: learned agents in either paper, MNSE included.
7. **Reduced-B1 claim and ceiling:**
   - P1: superiority on efficacy at equal treatment cost, narrowed to "action-entropy SAC; MNSE not compared".
   - P2: equivalence to the mean-field prediction, with no superiority sentence.
   - P3: superiority, but "strongest" only if MNSE is implemented or shown not to apply.
8. **Promotion of the paper:** P2 recommends against promoting on the abstract alone; P1 treats promotion as the arena's call; P3 requires a promoted reference.
9. **Where the prior-art pause is disclosed:** P1 and P3 in the draft; P2 only in `decisions/`.
10. **Claim A sweep status:** P2 accepts the correction at `decisions:250-253`; P1 and P3 treat it as unverified and as contradicting `:229`.
11. **"Converged" counterfactual:**
    - P1 reruns all of step 1 at M2.
    - P2 needs a new step 1 at the merge commit.
    - P3 reruns only the L* fixed-0 cells at merged code.
12. **Why n = 10:**
    - P1: it was the pre-data stage-1 count.
    - P3: arithmetic and time (n = 20 does not fit).
    - P2 chooses no n.
13. **Items raised by one proposal only:**
    - P1: the line-123 schema check and the two-look bound.
    - P3: the primary-metric blocker and the stale docstring.
    - P2: the scope-versus-criteria tension over B1 and the import path from the sweep script.

## 5. Amendments P3 needs before adoption (each checkable)

- **A1, schema check.** The worktree edits `_step1_cells` to `if report.get("report_schema") not in STEP_SCHEMAS:` (worktree line 126), and drops `REPORT_SCHEMA` from the import (line 68). C1 must:
  - restore the `4788c5a` text `!= REPORT_SCHEMA`;
  - re-add `REPORT_SCHEMA` to the import.
  
  Check: a diff of the seven step-1 functions between `4788c5a` and C1 is empty.
- **A2, merge timing.** Request C1 only after Claim A's test report is written or its line has stopped. Alternatively, the standing reviewer first shows that the merge leaves these unchanged: `check_test_launch`, `git_state`, `record_launch`, `launch_arguments`, `prereg_digest`, `check_seed_file`, `load_tiers` and `cross_resistance`. Check: C1's timestamp, or a function-level diff.
- **A3, "Converged" row.** Replace it with a full step 1 at merged code (criteria:74). Check: the table text.
- **A4, terminal rule.** Drop "under this decision". The preregistration must state that any later step 1 is a third look and must disclose both earlier looks and the combined error bound. Check: the preregistration text.
- **A5, two-look disclosure.** Add P1's disclosure to O4 and O5: two-look error at most 1 − (1 − α)², because the second check was chosen after an UNDECIDED. Check: the sentence text.
- **A6, Claim A facts.**
  - Replace the "unverified coordinator" paragraph with `decisions:250-253`.
  - Add the Claim A bitwise reproduction rerun (`decisions:266-270`) to the resource and condition-3 accounting.
  
  Check: the section 0 text.
- **A7, interval slots.** Every slot is filled only after run-auditor and adversarial-reviewer pass, by an agent outside the firewall, after the arena closes. Check: the §4 text.
- **A8, costing labels.** Relabel the §5 "floor" figures for n = 10, and the divide-by-6 val figure, as planning figures. Only the 5-seed launch-to-report column is a lower bound. Check: the §5 labels.
- **A9, which B1.** Before any B1 preregistration, record which B1 is meant: the scope's equivalence to a prediction (scope-outcome:8) or the criteria's tuned OC-SAC against SAC (criteria:29-36). Then fix the ceiling sentence to match. Check: a `decisions/` line.
- **A10, docstring.** Record a ruling on the stale `check_dev_protocol.py:7` docstring ("dev seeds 0-4"). It may stay, with the preregistration noting it, or change in C1, with the reviewer confirming it lies outside the step-1 functions. Check: a `decisions/` line.

## 6. What all three missed

- **The Claim A reproduction rerun.** `decisions:257-273` records a bitwise reproduction rerun of Claim A's dev map, launched at about 21:30. The git status shows its launch record, which I did not open. Claim A's use of the 18:40 map is blocked until that rerun passes.
  - None of the three counts this host load or this block.
  - So all three describe Claim A's "test path" as further along than the record shows.
  - P2's "while-sweep rule moot now" may be wrong at the time of writing.
  - Criterion 5's "host time taken from Claim A" is therefore not fully met by any proposal.
- **The pre-data remedy for this verdict was 30 seeds.** The pre-data design sent an inconclusive result to 30 pooled seeds (`arena/2026-11-25-step1-outcome.md:92`). Neither P1 nor P3 says that its n = 10 is below that count. A hostile reviewer would ask why.
- **MinRed SAC.** The adopted scope lists MinRed SAC in both forms, plus mechanism-level and two-level SAC, as comparators for the entropy tests (scope-outcome:11). The inbox names MinRed as the common ancestor. No proposal's reduced-B1 ceiling mentions MinRed, only MNSE.
- **Method pre-emption.** The abstract's "exact correction recovers the physical objective" could be the same objective as OC-SAC's entropy over resistance mechanisms. If so, the method is pre-empted, not just the theory. None of the three drop rules tests this; they drop only on learned-agent experiments.
- **The costing rule.** It gives no post-sweep bound for n > 5 beyond the 5-seed column. All three note this; no proposal can fully satisfy criterion 4 for n = 10 as the rule is written. That is a gap in the criteria.

## 7. Factual claims checked

**Confirmed:**
- `check_dev_protocol.py` at HEAD:
  - lines 69–78 hold the constants;
  - line 129 rejects other seed sets;
  - line 133 rejects uncommitted code;
  - lines 139–144 enforce the length set;
  - lines 177–178 tie the planner reference to the reports' git state, not HEAD;
  - lines 239–240 raise the commit DEFECT;
  - line 285 holds a(n) and the majority;
  - lines 321–333 hold the L* rule.
- `train_agent.py`:
  - line 112, `VAL_CODE_PATHS`;
  - line 205, `CODE_PATHS`, with notes outside them (lines 211–224);
  - line 1039, `--set` applies to both variants;
  - line 1069, git state read at launch;
  - lines 1108–1110, `ProcessPoolExecutor`;
  - line 1127, evaluation episodes set by tier and count.
- The worktree has `REPORT_SCHEMA = 4` (line 96), as P3 says.
- P1's suspicion is confirmed: the merge changes a step-1 function (worktree lines 80, 126). I found no other change in `_step1_cells`, `_reference`, `_defects`, `_gain_shares`, `_classify`, `_step` or `step1` (worktree lines 122–344).
- `.git/refs/heads/code-standards` is `4788c5a…`. `.git/worktrees/wt-code-standards/gitdir` points into another session's scratchpad.
- `sweep_phase_diagram.py:54-55` imports `check_test_launch`, `git_state` and `record_launch` from `train_agent`.
- `decisions` lines 24, 27, 140, 141–144, 153–162, 198–226, 229, 239–245 and 250–253 say what the proposals cite.
- The step-1 outcome file: `:37` (seeds 0–9), `:91`, `:92`, `:93`, `:168`. scope-outcome `:8`. `VERDICT.md:3`. `budget.yaml:3` (40 per day). `seeds.yaml` (dev 0–99, so seeds 5–14 are dev). The inbox, lines 21–31.
- All arithmetic in the three costing sections:
  - 85 097 and 425 485;
  - 47 189, 41 410 and 32 682;
  - 13 904, 24 908 and 36 150;
  - 69 879, 99 477 and 125 009;
  - 169 204, 210 472 and 244 300;
  - 112 053, 130 255 and 143 866;
  - 48 232, 60 550 and 69 127;
  - the EQUIV limits 0.070 at n = 10 and 0.0403 at n = 5;
  - "4 of 10 learned" at n = 10;
  - "18:53 on 28 November" (08:00 minus 13.11 h).

**Found wrong or out of date:**
- **P2,** "The criteria file does not mention the hit". Wrong for revision 6 (criteria:150-163). P2 was evidently written against revision 5.
- **P1 and P3,** the Claim A sweep facts "not verified" and "contradicting `decisions:229`". The working-tree correction at `decisions:250-253` records them. They are not unverified.
- **P3,** "the only edit is line 73". Wrong once the merge is included: the worktree also changes line 126 inside `_step1_cells`.
- **P1,** "n = 10 is the stage-1 count fixed before any data". Accurate for stage 1, but it leaves out that the pre-data remedy for an inconclusive result was n = 30 (outcome:92).

**Not checked:**
- P3's claim that `preregistrations/` is empty (I have no directory listing).
- Whether planner launches count against the daily cap.
- Whether the full texts can be obtained.

