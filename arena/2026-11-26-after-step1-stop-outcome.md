# Outcome: what follows the step-1 stop

## How it was decided

- **Criteria:** `2026-11-26-after-step1-stop-criteria.md`, revision 6. The standing reviewer passed revision 5. Revision 6 added the B1 PRE-EMPTS section after the arena opened.
- **Proposals.** Three proposers had identical briefs and the information firewall. Their proposals are in `-proposal-1.md` to `-proposal-3.md`: taken verbatim from each final hand-back, numbered in shuffled order, with revision markers removed. Two proposers learned of the PRE-EMPTS hit by a follow-up message. Proposal 2 was written partly against revision 5.
- **The judge** worked blind, with `-judgement.md`. It ranked proposal 3 first, proposal 2 a close second and proposal 1 third.
- **The standing reviewer failed that ranking.** On the judge's own scores, proposal 2 is at least equal to proposal 3 on every criterion and higher on two: severity, because proposal 3's "Converged" row breaks the code-state rule, and resources, because proposal 3 can put the merge ahead of Claim A's test. The judge's tie-break, the value of committing now to a resume design, lies outside the criteria, as the judge itself says.
- **Adopted: proposal 2, with amendments P2-1 to P2-9.** The arena's scoring decides, under the owner's delegation (`decisions/2026-11-26.md`, "Owner instructions of 2026-11-26 evening: the arena decides"). A ranking that does not follow from that scoring does not.

## Adopted

**B1 is stopped for the 29 November deadline and paused.**
- No new step 1, no step 2, no val launch and no B1 preregistration before the deadline.
- The decision is the same under every step-1 verdict. It follows from the deadline, the owner's commits that would have to come one after another, the host's commitment to Claim A, and the PRE-EMPTS stop.

**The host goes to Claim A.** Claim A's use of its 26 November dev map waits for the bitwise reproduction rerun `phase_20261126_213058_19724` (decisions, "Claim A dev map: which report…").

**No commit or merge of `code-standards` until Claim A's test report is written or its line has stopped (P2-2).**
- Claim A's sweep imports `check_test_launch`, `git_state`, `record_launch`, `ENV_KWARGS`, `PAIRED_DRUGS`, `ROOT` and `TIERS` from `experiments/train_agent.py`, which the branch edits.
- The branch's uncommitted worktree is preserved as `pending/code-standards-worktree-2026-11-26.patch`, SHA-256 `351ef40d05b333b07502b47e333423d8e8bd78246e85f07d37b167552aa2a10f`. It covers 6 files and applies cleanly to `4788c5a`.

**The pause ends only through a full-text read, under this rule fixed now (proposal 2's q1–q3, plus P2-3):**
1. Does the paper treat sequential problems with state, or only static decisions?
2. Does it derive the gap in the physical objective between the two regularised optima?
3. Does it measure that gap with learned agents? If yes, B1 is dropped.
4. Does it propose the corrected objective as a regulariser, and over which reference measure on classes? If yes, a reduced B1 cites OC-SAC's objective to that paper and to MinRed, and claims only the sequential test with learned agents.

The same rule applies to MNSE. The standing reviewer reviews the read before the pause lifts. If the full text cannot be read, B1 stays paused. Lifting a PRE-EMPTS stop rests on the owner's delegation, as read in decisions.

**Fixed wording for 29 November:**
- **Step 1:** "Convergence not established at n = 5, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1)", with each step's 95% interval in a slot. The slots are filled from `runs/` through the checker, only after run-auditor and adversarial-reviewer pass.
- **Rung 1:** VERDICT.md's sentence, quoted exactly, with the owner's override of the two stops disclosed.
- **B1 (P2-1):** "B1 (OC-SAC against SAC) is paused pending a prior-art check: a report filed on 26 November, read at abstract level only, may already derive its theory component. No OC-SAC-against-SAC comparison was run, and none is claimed. No training length has been qualified for the no-c condition, and the no-c confound is open." The paper is not named until it is promoted from `inbox/`.
- **If a reduced B1 is later resumed (P2-4),** its ceiling sentence names both MNSE and MinRed as not compared.

## Open items that bind any later B1 work

- **P2-5. Which B1.** It must be recorded before any B1 preregistration: the scope's equivalence to a prediction (`arena/2026-11-23-scope-outcome.md:8`), or the criteria's tuned OC-SAC against SAC.
- **P2-7.** The primary metric for "at equal treatment cost" is open (decisions, "Not changed, for the owner"). `check_dev_protocol.py:7` still says "dev seeds 0-4".
- **P2-8. Constraints on any later B1 arena:**
  - It costs launches under guard v2: one guarded launch at a time, about 24 planner workers after a cold boot, and training-worker commit memory not measured.
  - It justifies n against the 30-seed pre-data remedy (`arena/2026-11-25-step1-outcome.md:92`).
  - It runs under this arena's firewall.
- **From the judge's and the reviewer's "missed" lists:**
  - **Method pre-emption.** The abstract's "exact correction" may be OC-SAC's class-level objective. They coincide for equal class sizes and differ by a policy-dependent term for (8, 5, 4, 2, 1). Question 4 above tests this.
  - **The resume design** with merge restoration (proposal 3's A1, the `_step1_cells` schema check), the two-look bound at α = 0.025 per side (at most about 0.049, derived), and the A2 import list are inputs to that arena, not decisions taken here.

## Corrections recorded (P2-6)

- **Criteria revision 6, line 38,** says the relaunch `phase_20261126_190606_33244` "has run since 19:06". It failed at 19:35. The sweep `phase_20261126_184038_27744` wrote its report at 20:12 (decisions, "Correction…").
- **The proposals' Claim A accounting** omits the reproduction rerun launched at 21:30. That rerun is host load and blocks Claim A until it passes.

## Where the proposals disagreed

- **A new step 1:** before the deadline (1), after Claim A (3), or none (2).
- **Merge timing:** before launch (1), after the pause lifts (3), or after Claim A's test (2).
- **Reduced-B1 claim:** superiority (1, 3) or equivalence to a prediction (2).
- **Drop trigger:** learned-agent comparisons in the paper (1), the measured gap with learned agents (2), or learned agents in either paper (3).
- **Stopping B1 for Claim A:** kill and relaunch (1) or never kill (3).
- **Where the pause is disclosed:** in the draft (1, 3), or only in decisions (2), now in the draft too by P2-1.

## Convergence

All three paused B1 and gave the host to Claim A. Identical briefs give correlated answers, so this is weak evidence.

## Owner acts

- **PRE-EMPTS report (P2-9).** Reported in session on 26 November and recorded in decisions.
- **Commits** of the working tree:
  - `decisions/`, `arena/`, `tools/`, `tests/test_compare_phase_reports.py` and `pending/`;
  - the 20:12 phase report, and the rerun's report when written.
- **The rebase** of `step1-readiness` onto `origin/main` (`738519c`).
- **Promotion** of the PRE-EMPTS paper from `inbox/`, after the full-text read.
- **A fixed page-file size, if wanted.** Without one, guard v2 admits about 24 planner workers after a cold boot.
