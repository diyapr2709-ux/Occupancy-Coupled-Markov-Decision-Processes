# Proposal 3: pause B1 on the PRE-EMPTS hit, give the host to Claim A, and fix now, behind the firewall, one n = 10 step-1 check that runs only if the pause lifts and Claim A has released the host

**Before the sections, five facts that change the decision:**
- **The coordinator's update is not verified.** It says the phase sweep is no longer running, that `phase_20261126_184038_27744` finished at 20:12, that `_190606_33244` failed at 19:35 with no report, and that the host rebooted at 20:23. I could not check any of it without opening `runs/phase/`. It also conflicts with `decisions/2026-11-26.md:229`, which says `184038` wrote no report and the host "was not running it". My option does not depend on which account is right, because B1 never runs beside a Claim A launch. I cost both rules anyway.
- **The code the criteria require for a new step 1 is not committed anywhere.** The `code-standards` branch ref is still `4788c5a` (`.git/refs/heads/code-standards:1`). The merge's content (`--set-oc-sac` at worktree `train_agent.py:1102`, `REPORT_SCHEMA = 4` at `:96`) exists only in an uncommitted worktree in another session's temp scratchpad (`.git/worktrees/wt-code-standards/gitdir:1`; `decisions/2026-11-26.md:198-200`). It survived until now. A cleanup or reboot can delete it.
- **The B1 chain has a blocker that compute cannot remove.** The primary metric for "at equal treatment cost" is open, and its efficacy threshold sits in protected `ocmdp/eval/` (`decisions/2026-11-26.md:218`).
- **No preregistration exists yet,** not even Claim A's. A Glob of `preregistrations/*` returns nothing.
- **The PRE-EMPTS hit and MNSE are unread beyond their abstracts** (`inbox/2026-11-26-claimA-B1-B2-prior-art-2.md:23-25, 29-31`). Nothing in inbox/ can be cited until it is promoted (CLAUDE.md, "Gathered material").

## 1. Recommendation

- **B1 is paused, not dropped and not reduced yet.**
  - CLAUDE.md says a PRE-EMPTS hit stops the work. The hit is partial and covers only the theory (inbox file, line 25). It was judged from an abstract alone, so dropping B1 on it, or citing its derivation, would be premature.
  - **The pause ends with a full-text read** of the PRE-EMPTS paper and of MNSE (RA-L 2025, inbox lines 29-31). A gatherer does the read. It uses no host time and no launches.
  - **The rule that ends the pause is fixed now:**
    - B1 is **dropped** if either paper already measures learned agents under action-class or redundancy-aware entropy against action entropy.
    - Otherwise B1 is **kept in reduced form**: an empirical test of OC-SAC against SAC, with the static conditional-divergence derivation cited and not claimed.
    - If the full text cannot be read before the deadline, B1 **stays paused** and is reported that way.
- **The host goes to Claim A**, as in the reference option.
- **One new step 1 is fixed now**, by proposers who cannot see the step-1 numbers. It runs only if all four of these hold:
  1. the pause lifts with B1 kept;
  2. the owner commits C1;
  3. Claim A has no launch running, and its test report is written or its line has stopped by its own rule;
  4. no commit lands between the planner reference and the sixth launch.
- **The new step 1:**
  - c-given, dev seeds 5–14 (n = 10), lengths {12 000, 24 000, 48 000};
  - every rule at its `4788c5a` value, with only `SEEDS` changed;
  - a terminal rule with no extension clause.
- **If it converges,** step 2 runs on seeds 5–14 at L*. The option then stops for a new arena on MNSE as a comparator and on the primary metric. There is no val sweep, no B1 preregistration and no test before 29 November.

## 2. Pre-commitment table

**On the original step-1 verdict.** Under every row, B1 is paused on the PRE-EMPTS hit and the host goes to Claim A.

| Step-1 verdict (seeds 0–4) | Action if the pause lifts with B1 kept |
|---|---|
| DIVERGING | No new step 1. B1's learned-agent line stops and is reported. A definite stop never triggers more seeds (`arena/2026-11-25-step1-outcome.md:91`). |
| DECLINING | Same as DIVERGING. |
| FLAT_BELOW_MAJORITY | Same as DIVERGING. |
| RISING | Stop and report. The only pre-data remedy was the 96 000 extension (outcome `:93`), and the lengths are now fixed (criteria `:47`). |
| UNDECIDED (the actual verdict) | The new step 1 at n = 10 on seeds 5–14. UNDECIDED is the only class the pre-data design sent to more seeds (outcome `:92`). That stage is not reinstated as pre-specified: the seeds are new, nothing is pooled and there is a new preregistration. |
| Converged at `4788c5a` | Step 2 at the merged code. The `4788c5a` fixed-0 cells cannot be reused, because the merge changes `train_agent.py`, which is in `VAL_CODE_PATHS` (`train_agent.py:112`). Step 2 would add two fixed-0 cells at L* on seeds 0–4: 8 launches. Then stop for the val-design arena. |

**On the new step-1 verdict (n = 10):**
- **Converged:** record L* as provisional in `decisions/`. Run step 2 (6 launches) on seeds 5–14 at L*, reusing the new step 1's fixed-0 cells. Then stop for the arena.
- **DIVERGING, DECLINING, FLAT_BELOW_MAJORITY, RISING or UNDECIDED, and also DEFECT or REFERENCE NOT ESTABLISHED:** B1's learned-agent line stops and is reported in the fixed wording at n = 10. There is no further step 1 and no rerun under this decision.

## 3. Compliance with criterion 1

**The bar stays fixed.**
- Unchanged at the `4788c5a` values:
  - `DELTA = 0.05`, `SPREAD_LIMIT = 0.02`, `MIN_PLANNER_SEEDS = 5`, `BASELINE_TOLERANCE = 1e-12`, `INSTANCES` and `CONDITION` (`check_dev_protocol.py:69-78`);
  - the t-interval (`:263-277`);
  - a(n) and the majority (`:285`);
  - the imputation (`:252-260`);
  - the lengths and the L* rule (`:321-333`).
- **The only edit** is line 73: `SEEDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`. Line 129 requires `args.seeds` to equal it in that order.
- **The seeds and the count were both chosen after the verdict was UNDECIDED.** The preregistration says so.
- **Why n = 10.** The choice uses only arithmetic and the time budget, no step-1 number:
  - at zero mean change, EQUIV needs SD(d) < 0.05·√10 / t(0.975, 9) = 0.05 × 3.162 / 2.262 ≈ 0.070, against 0.0403 at n = 5 (`decisions/2026-11-26.md:24`);
  - a(10) = 1 and ⌈10/2⌉ = 5;
  - n = 10 takes 2 rounds at `--workers 5`. n = 20 takes 4 rounds, about 25.5 h at the floor (section 5), which leaves no room for Claim A and a reboot.
  - These are derived figures, not measured ones.
- **A contradiction in the criteria, flagged rather than resolved:**
  - the criteria allow only `SEEDS` and the merge's changes in the checker (`:48-50`);
  - the module docstring says "dev seeds 0-4" (`check_dev_protocol.py:7`), and line 73's comment says "the owner's step-1 seeds";
  - CLAUDE.md requires docstrings to describe what the code does.
  - My reading: line 73's trailing comment goes with `SEEDS`, and the stale docstring is recorded in `decisions/` and left as it is. The judge should rule on this.

**No pooling.** Seeds 5–14 are disjoint from 0–4. The 0–4 reports stay UNDECIDED and are not re-checked.

**The preregistration is fixed before launch.** It is a new dated file in `preregistrations/`, written into the working tree and committed in C1. It holds:
- the design above;
- the launch arguments exactly as the criteria fix them (`:56-61`): `--tier dev --condition c-given --platform rtx-1 --seeds 5 … 14 --sigmas 0 --episodes 10 --workers 5 --backend local --set temperature=fixed alpha=0`, plus `--class-sizes` and `--quarters` (flags at `train_agent.py:1023-1036`);
- the terminal rule: anything but CONVERGED stops B1 and is reported, with no extension clause;
- step 2's seed set (5–14) and its six launches (`decisions/2026-11-26.md:26-28`);
- the sentence "this check was chosen because the first was UNDECIDED; its result stands on its own data only".

**Commit dependencies, stage by stage:**

| Stage | Runs on | Waits for | If no commit arrives |
|---|---|---|---|
| Full-text read | nothing | no commit | not applicable |
| Planner reference and step 1 | one launch commit containing C1 | C1, the owner's act: (i) commit the worktree changes onto `code-standards`; (ii) merge into `step1-readiness`; (iii) the `SEEDS` edit; (iv) the preregistration. The standing reviewer diffs the step-1 functions against `4788c5a` over the merge first (criteria `:51`). | Nothing launches. The 29 November sentence is O1 or O3, and the plan stands after the deadline. |
| Step 2 | any HEAD whose `VAL_CODE_PATHS` content equals C1's (merged checker, worktree `check_dev_protocol.py:353-363`) | no new commit | not applicable. If that content has changed, step 2 does not launch. |
| Val sweep, B1 preregistration, test | not in this option | a new arena; owner commit C2; `approvals/<slug>.ok` | stops |

- **One launch commit.** The launch commit is HEAD at the planner reference. `_reference` requires the reference to come from the same commit and code (`:177-178`), and `_defects` rejects reports from different commits (`:239-240`).
- **If a commit lands before the sixth launch,** the planner reference is rerun at the new HEAD before any cell is launched.
- **The working tree may stay dirty.** "Dirty" covers only `CODE_PATHS` (`train_agent.py:205-224`), so the uncommitted `decisions/` and `arena/` files do not block a launch.
- **Reuse of fixed-0 cells.** The `4788c5a` cells cannot be reused: they were on other seeds and other code. The new step 1's cells are reused by step 2.

**The table is disclosed** in section 2.

## 4. Sentences for 29 November

**Always written:**
- "Step 1: convergence not established at n = 5, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1); both instances UNDECIDED (95% intervals of the mean change in g for 12 000→24 000, 24 000→48 000 and 12 000→48 000: (2, 2) [·,·], [·,·], [·,·]; (8, 5, 4, 2, 1) [·,·], [·,·], [·,·], from runs/ through check_dev_protocol.py)."
- "Rung 1 not reproduced: the published result is not specified precisely enough to fix a reproduction target. On 26 November 2026 the owner overrode the two stops this verdict would have triggered, CLAUDE.md's 'report the size of the discrepancy and stop' and the step-1 outcome's 'if rung 1 fails, the ladder stops', so later steps were allowed to proceed."

**For B1, under each outcome:**
- **O1, paused through the deadline:** "B1 (OC-SAC against SAC) is paused pending a prior-art check: a report filed on 26 November, read at abstract level only, may already derive its theory component. No OC-SAC-against-SAC comparison was run, and none is claimed."
- **O2, dropped after the full text:** "B1 is dropped: [promoted reference] already [derives / derives and measures] … . No OC-SAC-against-SAC comparison was run."
- **O3, kept in reduced form, check not launched or not reported:** "B1 continues as an empirical test only. The result that redundant action representations shift the entropy-regularised optimum by a conditional-divergence term is [promoted reference]'s, cited and not claimed. A second dev convergence check, preregistered in [file] after the first was UNDECIDED, had [not been launched / not reported] by 29 November."
- **O4, the second check does not converge:** "A second dev convergence check, chosen because the first was UNDECIDED and judged on its own data only: convergence not established at n = 10, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1) ([instance]: [class], intervals [·,·] …). B1's learned-agent line stops here."
- **O5, the second check converges:** "…established convergence at n = 10 … with L* = [·] decisions. L* is unqualified for no-c, for learned temperatures (L* also sets the temperature learning rate, `arena/2026-11-25-step1-outcome.md:168`), for α > 0 and for σ > 0. The no-c confound (OC-SAC's entropy uses C's classes) is open. A step can pass with the longer length's learned count one below the majority (`decisions/2026-11-26.md:140`). No OC-SAC-against-SAC comparison was run."
  - Optionally: "Step 2's temperature checks at L* on dev seeds 5–14 [passed / failed at …]."

**How the wording changes if B1 continues.**
- The theory sentence is cited, never claimed.
- Criterion 2's ceiling, "outperforms the strongest baselines that could be constructed, tuned at parity", is licensed only if MNSE was either implemented and tuned at parity or shown not to apply to discrete, interchangeable actions.
- Otherwise the most a later test licenses is: "outperforms action-entropy SAC tuned at parity; MNSE, a direct alternative, was not compared (no code or tuned configuration published)."

## 5. Chain costing

**The rules.**
- Anchor: 21:00 on 26 November. The deadline is assumed to be 08:00 on 29 November, which gives 59 h; "morning" is not defined.
- **After-sweep rule.** For n = 10 at `--workers 5` I use: the cell's 5-seed launch-to-report time, plus (⌈10/5⌉ − 1) = 1 × its slowest-pair time. This is the form of the criteria's own ⌈n/k⌉ × slowest-pair rule. It is my extension, and it is not a strict bound for seeds no one has run.
- At most 6 launches run at once (30 workers on 32 cores).

**Step 1 (6 launches, plus 1 planner reference).** Under the after-sweep rule:

| Cell | Floor |
|---|---|
| (2, 2), 12 000 | 9 324 + 4 580 = 13 904 s |
| (2, 2), 24 000 | 15 758 + 9 150 = 24 908 s |
| (2, 2), 48 000 | 19 301 + 16 849 = 36 150 s |
| (8, 5, 4, 2, 1), 12 000 | 19 454 + 13 228 = 32 682 s |
| (8, 5, 4, 2, 1), 24 000 | 22 396 + 19 014 = 41 410 s |
| (8, 5, 4, 2, 1), 48 000 | 24 913 + 22 276 = **47 189 s = 13.11 h** |

- The step costs its slowest cell: 13.11 h.
- The planner reference's time is not measured. On 26 November it was followed by step-1 launches within about 3 minutes (`decisions/2026-11-26.md:153-154`).
- Under the sweep-running rule (⌈10/2⌉ = 5 × slowest pair, six cells sharing 2 cores): the largest cell alone is 5 × 22 276 = 111 380 s = 30.9 h, and the sum over cells is 5 × 85 097 = 425 485 s = 118.2 h. It cannot finish before the deadline.
- The latest launch that finishes at the after-sweep floor is about 18:53 on 28 November.

**Step 2 (6 launches, at L*).** These floors use the α = 0 times, because α > 0 runtimes are not measured.
- Floor = the (8, 5, 4, 2, 1) cell at L*: 32 682 s = 9.08 h at 12 000, 41 410 s = 11.50 h at 24 000, 47 189 s = 13.11 h at 48 000.
- Step 1 and step 2 together at the floor: 22.19 h to 26.22 h.
- Under the sweep-running rule it cannot finish. Three temperatures × (22 900 + 66 140) s at 12 000 = 267 120 s = 74.2 h.

**Val sweep (not in this option, costed for completeness).**
- Count: 9 candidates × 2 instances = 18 launches at minimum (the cells are not fixed by any preregistration), 10 val seeds each.
- Floor: slot-seconds ÷ 6.
  - At 12 000: (9 × 32 682 + 9 × 13 904) / 6 = 419 274 / 6 = 69 879 s = 19.41 h.
  - At 24 000: (9 × 41 410 + 9 × 24 908) / 6 = 596 862 / 6 = 99 477 s = 27.63 h.
  - At 48 000: (9 × 47 189 + 9 × 36 150) / 6 = 750 051 / 6 = 125 009 s = 34.72 h.

**Test.** Floor at 5 seeds, (8, 5, 4, 2, 1): 19 454 / 22 396 / 24 913 s. The seed count is not fixed.

**The whole chain at the floor,** with zero owner delay and Claim A taking nothing:

| L* | Chain | Against the 59 h window |
|---|---|---|
| 12 000 | 47 189 + 32 682 + 69 879 + 19 454 = 169 204 s = 47.0 h | 12 h slack |
| 24 000 | 210 472 s = 58.5 h | 0.5 h slack |
| 48 000 | 244 300 s = 67.9 h | over |

Even the 12 000 case ignores the primary-metric blocker and the MNSE question.

**Stages that cannot finish before the deadline:**
- every stage, under the sweep-running rule;
- the val sweep, the B1 preregistration and the B1 test under the after-sweep rule, given Claim A priority and owner latency;
- step 1 itself, if it is launched after about 18:53 on 28 November.

## 6. Resources used up

- **Val seeds (100–199):** none.
- **Test seeds (1000–1299):** none.
- **Dev seeds:** 5–14.
- **Launches:** at most 7 B1 launches on the day step 1 starts, and 6 on the day step 2 starts. That is within the 40-per-day cap alongside Claim A, whose launch count I do not know.
  - The counts at `decisions/2026-11-26.md:162` and `:230` fit planner launches not being counted against the cap. Not verified: I did not read `runs/ledger.txt`.
- **Host time taken from Claim A:** zero by construction. Resume condition 3 forbids any overlap, and a running B1 launch is never killed, because a relaunch after any owner commit would trip `:239-240`. If Claim A needs a host run while B1 is running, B1 cannot have launched, by condition 3.
- **Owner time:** one commit, C1, requested only after the pause lifts.

## 7. What it adds over the reference option

- **It handles the PRE-EMPTS hit by its own rule.** The work stops, a zero-cost full-text read resolves it, and the decision rule is fixed in advance. The reference option either drops B1 on an abstract or leaves it undefined.
- **It fixes the resume design behind the firewall.** The seeds, n and terminal rule are chosen by proposers who cannot see the step-1 intervals. Someone choosing after the deadline may have seen the printout (criteria `:20-24`), so fixing it later could not be done as cleanly.
- **It uses only host time Claim A has released.** It costs Claim A nothing.
- **It can resolve whether B1 is viable at all.** At the margin it can still put one convergence sentence (O4 or O5) into the 29 November record.
- **When none of the resume conditions hold, it is the reference option.**

## 8. Strongest objection, and the answer

**The objection.** "This is the reference option plus paperwork:
- the resume will almost certainly not run before 29 November;
- n = 10 has no power basis, so it may come back UNDECIDED and waste 13 h;
- it pre-fixes a design for a line that a prior-art hit has just stopped;
- it depends on owner time and on a merge that lives in a temp directory."

**The answer.** Not fatal.
- Before the resume, the option costs no host time and no launches. It asks for C1 only after the pause lifts.
- n cannot be sized for power without step-1 numbers (the firewall). The terminal rule makes another UNDECIDED a reported result, not a reason to extend.
- Fixing the design now is the firewall's whole point.
- The worktree risk exists under every option that keeps B1, so it is not specific to this one.

**What would show I was wrong:**
- **Claim A was slowed.** Any overlap between B1 and Claim A launch records, or a Claim A stage that waited on owner attention spent on C1.
- **B1 should have been dropped now.** The full-text read shows the paper already tests learned agents. The cost would be one draft preregistration.
- **The resume was not worth fixing.** The full text cannot be obtained before the deadline. The option then reduces to the reference, at no host cost.

**Prior work.** The B1 prior-art items in `inbox/2026-11-26-claimA-B1-B2-prior-art-2.md`:
- "Equivalent Optima, Different Policies", AI Engineering 2(2), 2026, DOI 10.53941/aieng.2026.100020: PRE-EMPTS, partial;
- MNSE, IEEE RA-L 10(7), 2025, DOI 10.1109/lra.2025.3572822: BASELINE;
- both found through the citers of MinRed (Baram et al. 2021).

All three are inbox only and unpromoted, and none was read in full. I made no further search.

**Not verified:**
- the coordinator's account of the sweep and the reboot;
- whether the worktree survives;
- whether planner launches count against the ledger;
- the planner reference's runtime;
- α > 0 runtimes;
- whether the PRE-EMPTS paper's full text is accessible;
- whether MNSE applies to discrete actions.
