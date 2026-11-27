# Proposal 1: pause B1, read the full text under a rule fixed now, and run the fresh-seed step 1 only if B1 survives in reduced form

It adds the partial PRE-EMPTS hit and the MNSE BASELINE entry from the 26 November prior-art file. The eight sections and the firewall are unchanged: I did not open `runs/train/` or `runs/phase/`, did not read the step-1 verdict section, and ran nothing.

## 0. Facts added since the brief

- **Claim A sweep (coordinator, not verified by me).** `phase_20261126_184038_27744` finished at 20:12. The relaunch `_190606_33244` failed at 19:35 with no report. The host rebooted at 20:23.
  - This contradicts the record: `decisions/2026-11-26.md:229` says `_184038_27744` "wrote a launch record at 18:40 and no report".
  - Whether Claim A has a dev report therefore decides whether it needs the host again. If it does, the identical-relaunch precedent applies (`decisions:230`).
  - Both costing rules are still worked through below.
- **Partial PRE-EMPTS hit on B1** (`inbox/2026-11-26-claimA-B1-B2-prior-art-2.md:21-25`).
  - The paper is "Equivalent Optima, Different Policies", AI Engineering 2(2) art. 20, 21 Oct 2026, DOI 10.53941/aieng.2026.100020. Only the abstract page was read (line 23).
  - It derives that redundant representations shift the entropy-regularised optimum by a nonnegative conditional-divergence term, and that an exact correction recovers the physical objective. It has no RL agents (line 24).
  - The scout's tag: the mean-field side of B1 is pre-empted. The measured gap between learned SAC agents stays novel "if it stands", and "the derivation itself should be cited, not claimed" (line 25).
  - CLAUDE.md lines 16-17 make a PRE-EMPTS hit a stop.
  - Inbox items cannot be cited in a draft until they are promoted to `references/` (CLAUDE.md, "Gathered material").
- **BASELINE: MNSE** (`inbox/...prior-art-2.md:27-31`), IEEE RA-L 10(7), 2025.
  - It is "a direct alternative to class-entropy SAC". It has no code and no tuned configuration, its full text is not open access, and its "gap" theory "must be read in full before B1's theory is claimed as new" (line 31).
  - Adding it as a comparator would mean code in `ocmdp/baselines/`, a protected path that only a human may edit (criteria:141; `decisions:245`).
- **The B1 headline claim.** CLAUDE.md lines 3-5: "entropy over resistance mechanisms preserves drug efficacy better than entropy over drugs, at equal treatment cost." The hit does not touch this empirical claim. It does touch any theoretical account of why it should hold.

## 1. Recommendation

- **B1 status: paused until the full text is read, under an assessment rule fixed now, before anyone reads it.**
  - **DROP** if the full text reports learned sequential RL agents, of any algorithm, comparing entropy over actions with entropy over action classes or decoder fibres under redundancy, with a measured performance comparison. B1's empirical side would then be pre-empted too.
  - **KEEP REDUCED** if every result is static or log-linear, planned or optimised, with no learned RL agent. That is what the abstract and the scout's line 24 say. B1 then becomes an empirical test of OC-SAC against action-entropy SAC on this benchmark. The derivation is cited, after promotion to `references/`, and never claimed. MNSE is named as a published alternative that was not compared.
  - **Stays PAUSED** if no full-text assessment, by a lit-scout read reviewed by the standing adversarial reviewer, exists by 12:00 on 28 November. B1 is then reported as paused.
  - The read is a gatherer task. It uses no host compute, and whether the full text can be accessed is not verified.
- **Only under KEEP REDUCED: one preregistered dev step 1 on fresh dev seeds 5–14 (n = 10).**
  - Every rule, length and launch argument stays at its `4788c5a` value.
  - It runs at one launch commit, M2, on the merged `code-standards` code. Its fixed-0 cells and a step 2 on seeds 5–14 then share code with any later val sweep and test.
- **Launch gates.** All of these must hold:
  - (a) the owner's commits M1 (the merge) and M2 (the checker's `SEEDS` edit plus the new preregistration) exist;
  - (b) no Claim A launch is running or ready;
  - (c) the time is no later than 12:00 on 28 November;
  - (d) the assessment says KEEP REDUCED.
- **Terminal rule.** Any verdict other than CONVERGED stops B1 and is reported.
- **Not attempted before 29 November:** the val sweep, the B1 preregistration, the approval and the B1 test. They cannot finish before the deadline under the planning figures (section 5).
- **If any gate fails,** the option becomes the reference option, with B1's status reported as paused or dropped.
- **How n and the seeds were chosen.** Seeds 5–14 and n = 10 were chosen after the step-1 verdict was UNDECIDED. n = 10 is the stage-1 count the 2026-11-25 design fixed before any data (`arena/2026-11-25-step1-outcome.md:37`).

**Not taken:**
- *Keep B1 reduced at once:* it would cite or plan around an abstract.
- *Drop B1 at once:* the scout says the empirical side stays novel, and dropping on an unread abstract throws that away without evidence.

## 2. Pre-commitment table (action under each original step-1 verdict)

Under DROP or PAUSED, every row is the reference option, and B1's status is reported. The rows below apply under KEEP REDUCED.

| Step-1 verdict (seeds 0–4) | Action under KEEP REDUCED |
|---|---|
| DIVERGING | Reference option. A definite stop never triggered more seeds (`arena/2026-11-25-step1-outcome.md:91`). |
| DECLINING | Reference option. |
| FLAT_BELOW_MAJORITY | Reference option. |
| RISING | Reference option. A longer length would need a new L* rule (criteria:47; `check_dev_protocol.py:326-331`). |
| UNDECIDED (actual) | Fresh-seed step 1 at n = 10 on seeds 5–14, gates (a)–(d). |
| Converged | Merge (M1), then M2 with `SEEDS` left at 0–4, then an identical relaunch of step 1 at M2. A different verdict or L* is a stop. Then step 2 on seeds 0–4. The `4788c5a` reports cannot be reused at merged code (`train_agent.py:112`; `decisions:206-222`). The val sweep and test come after the deadline. |

## 3. Compliance with criterion 1, item by item

**The bar stays fixed.**
- The only edit at M2 is `check_dev_protocol.py:73`, changing `SEEDS` to `[5, …, 14]`, plus the merge's changes outside the step-1 functions.
- These stay at their `4788c5a` values:
  - `DELTA` (line 71);
  - the 95% t-interval (lines 263-277);
  - a(n) and the ⌈n/2⌉ majority (line 285);
  - the imputation (line 259);
  - `SPREAD_LIMIT` and `MIN_PLANNER_SEEDS` (lines 75-76);
  - `BASELINE_TOLERANCE` (line 77);
  - `INSTANCES`, `LENGTHS` and `CONDITION` (lines 69-70, 74);
  - the L* rule (lines 321-333).
- The reviewer diffs the seven step-1 functions against `4788c5a` across the merge (criteria:51).
- **Not verified:** whether the branch edited line 123 inside `_step1_cells` to accept schemas 3 and 4 (`decisions:214`).
  - If it did, M2 restores line 123's `4788c5a` text.
  - That is harmless: line 123 compares against the imported `REPORT_SCHEMA`, which is 4 at merged code, and every new report is schema 4.
- **Launch arguments,** as fixed by criteria:56-63:
  - `--tier dev --condition c-given --platform rtx-1 --class-sizes <I> --quarters <L> --seeds 5 … 14 --sigmas 0 --episodes 10 --workers 5 --set temperature=fixed alpha=0`;
  - delay and lam at their defaults, the local backend.
- **What EQUIV needs at n = 10.** At a mean change of zero, SD(d) < 0.05·√10/t(0.975, 9) ≈ 0.070. This is derived by arithmetic; SD(d) is not measured.

**No pooling.**
- Line 129 rejects any report not on exactly `SEEDS`, so the seed 0–4 reports cannot enter.
- Those reports stay UNDECIDED (criteria:136).

**A preregistration fixed before launch.** A new dated file in M2. It fixes:
- the seven commands and the launch commit;
- the seeds and n, with when they were chosen;
- the terminal rule: no extension, and no further step 1 before 29 November;
- step 2 on seeds 5–14 at L*, with the three launches per instance at `decisions:27`;
- the cutoff;
- the Claim A precedence rule and the reboot relaunch rule (`decisions:158-162`);
- the section 4 sentences;
- the statement that it stands on its own data (criteria:138), and the two-look bound (≤ 1 − (1 − α)², derived);
- B1's reduced wording;
- the KEEP REDUCED assessment, by file name.

**Commit dependencies.**
- **Stage P, the full-text read.** No commit is needed: the gatherer writes to `inbox/`.
  - Promoting the paper to `references/` is needed before any draft names it. Promotion is an owner choice, now the arena's.
  - If there is no read by the cutoff, B1 stays PAUSED and nothing launches.
- **Stage 0, M1 then M2.** If either is missing at the cutoff, or the reviewer finds a step-1-function change that cannot be removed, the option becomes the reference option.
- **Stage 1a, planner reference.** `planner_restarts.py --class-sizes 8 5 4 2 1` at its defaults and at least 5 seeds, at M2 (lines 177-178).
- **Stage 1b, the six launches.** All six start at M2 within minutes of 1a. Reports stamp the commit at launch (`train_agent.py:1069`), so only a commit between 1a and the last start spends the step (criteria:67).
  - Notes under `decisions/` and `runs/` never make a launch dirty (`train_agent.py:211, 219-220`).
  - If a commit lands in that window and no report exists yet, rerun at the new HEAD before the cutoff. Otherwise stop.
- **Stage 2, step 2.** It needs step 1 to exit 0. It runs at M2, or at a commit with unchanged `VAL_CODE_PATHS` content (`decisions:225`).
  - It reuses the new fixed-0 cells.
  - The seed 0–4 cells are not reusable: they are on other seeds and ran on other code.
- **Stages 3–6** are not attempted before the deadline.

**Claim A precedence.**
- B1 does not launch while a Claim A launch is running or ready.
- If Claim A becomes ready during a B1 run, the B1 processes are stopped. No output is read, the records stay, and the stop is logged in `decisions/`. B1 relaunches only before the cutoff.

**PRE-EMPTS handling.**
- The hit edits nothing.
- B1 stays stopped except under the KEEP REDUCED branch.
- That the arena may lift CLAUDE.md:17's stop rests on the owner's delegation (`decisions:239`). This is an inference of the same kind as `decisions:241`.

## 4. Sentences for 29 November (fixed wording)

**Always written:**
- (S1) "Step 1 on dev seeds 0–4: convergence not established at n = 5, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1); 95% intervals: [six slots, from runs/ through the checker]."
- (S2) "Rung 1 not reproduced: the published result is not specified precisely enough to fix a reproduction target. The owner's decision of 26 November overrode the two stops this outcome would have triggered: CLAUDE.md's 'report the size of the discrepancy and stop' and the step-1 outcome's 'if rung 1 fails, the ladder stops'."
- (S3) "No comparison of OC-SAC with SAC has been made; B1 is untested. The no-c confound is open; any L* is unqualified for no-c."

**B1 status sentence (S4), one of:**
- *PAUSED:* "B1 is paused. A prior-art report filed on 26 November, from an abstract only, indicates that B1's theory component, that redundant action representations shift the entropy-regularised optimum by a conditional-divergence term, has been derived in a static log-linear setting; the full text has not been assessed."
  - The paper is not named until it is promoted.
- *KEEP REDUCED:* "B1 is retained as an empirical test only: whether entropy over resistance mechanisms (OC-SAC) preserves drug efficacy better than entropy over drugs (SAC), at equal treatment cost, with both tuned at parity on this project's benchmark. The derivation that redundant representations shift the entropy-regularised optimum is [citation, promoted] and is not claimed here. MNSE, a published alternative to class-entropy regularisation, is not compared: no code or tuned configuration is available, and adding it would edit a protected path."
- *DROP:* "B1 is dropped: the full text of [citation] reports [the learned-agent comparison that triggered DROP]."

**How the B1 wording changes under KEEP REDUCED.** The most a future B1 test licenses narrows from criterion 2's "outperforms the strongest baselines that could be constructed, tuned at parity" to "outperforms [or: does not outperform] action-entropy SAC tuned at parity on this benchmark; MNSE not compared". A known alternative was left out, so "strongest" cannot be claimed. No sentence attributes a mechanism or a theory to B1 as its own.

**Step-1 outcome sentence, added only under KEEP REDUCED, one of:**
- *O0, not launched:* "A second step 1 on fresh dev seeds 5–14 was proposed and not run before the deadline, because [reason]."
- *O1, CONVERGED:* "A second step 1, preregistered and run because the first was UNDECIDED, on fresh dev seeds 5–14 (n = 10), with no pooling and the unchanged rule, established convergence at L* = __ decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1); 95% intervals: [slots]. The result rests on its own data; the two-look error rate is at most about twice the per-look rate. L* is unqualified for no-c, for learned temperatures (whose alpha_lr is set by L*, `arena/2026-11-25-step1-outcome.md:168`) and for σ > 0. Under the adopted rule an instance can converge at 24 000 with 4 of 10 runs learned at 48 000 (`check_dev_protocol.py:290`)." Then one of: "Step 2 at L* on seeds 5–14 passed every check", "failed [check]", or "was not run before the deadline".
- *O2, STOP:* "A second step 1, preregistered and run because the first was UNDECIDED, on fresh dev seeds 5–14: convergence not established at n = 10, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1); reason at the 24 000→48 000 step: [instance: class]; 95% intervals: [slots]. Under its preregistration no further step 1 follows."
- *O3:* "A second step 1 on dev seeds 5–14 stopped on [REFERENCE NOT ESTABLISHED / DEFECT: text] and gives no convergence verdict."
- *O4:* "A second step 1 on dev seeds 5–14 was launched at __ and had not written all six reports by the deadline; its verdict is not measured."

The words "did not converge" and "unstable" appear nowhere.

## 5. Chain costing

The figures come from criteria:84-91.
- **LB** is the rule's launch-to-report figure, a lower bound.
- **PF** is 2 × the slowest pair plus the n = 5 overhead. It is derived, not a bound. Ten pair jobs on `ProcessPoolExecutor(5)` take at least two waves (`train_agent.py:1108-1110`).

**Stage P, the full-text read.** Zero host time and zero launches. How long it takes is not measured.

**Stage 1a, planner reference.** 1 launch. Its duration is not measured. The names at `decisions:153-154` suggest about 2–3 minutes (derived).

**Stage 1b, step 1 at n = 10.** 6 launches, 30 workers.
- After the sweep:
  - LB = 24 913 s ≈ 6.9 h.
  - PF for (8, 5, 4, 2, 1) at 48 000: 2 × 22 276 + 2 637 = 47 189 s ≈ 13.1 h.
- While a sweep runs:
  - the slowest cell alone: 5 × 22 276 = 111 380 s ≈ 30.9 h;
  - all six cells together: 5 × 85 097 = 425 485 s ≈ 118 h;
  - **cannot finish.** Gate (b) means it does not launch while a sweep runs.

**Stage 2, step 2 at L*.** 6 launches. Run times at α > 0 are not measured.

| L* | LB | PF |
|---|---|---|
| 12 000 | 19 454 s | 2 × 13 228 + 6 226 = 32 682 s ≈ 9.1 h |
| 24 000 | 22 396 s | 2 × 19 014 + 3 382 = 41 410 s ≈ 11.5 h |
| 48 000 | 24 913 s | 47 189 s ≈ 13.1 h |

- While a sweep runs, at 48 000: 5 × 3 × 39 125 = 586 875 s ≈ 163 h. **Cannot finish.**

**Stage 3, val sweep.** 18 launches: 9 candidates × 2 instances, 10 val seeds each, 180 pair jobs. That the cells are the two instances at σ = 0 is not verified.
- LB is about 24 913 s.
- PF: (90 × 22 276 + 90 × 16 849)/32 ≈ 110 039 s ≈ 30.6 h.
- **Cannot finish under PF.**

**Stages 4–5, preregistration, commit and approval.** These are owner acts; their latency is not known. **Cannot be placed before the deadline under PF.**

**Stage 6, test.** At least 1 launch, LB ≥ 24 913 s. **Cannot finish under PF.**

**Whole chain.**
- About 59.5 h are available, from about 20:30 on 26 November to about 08:00 on 29 November. The deadline hour is my assumption.
- LB sum: 4 × 24 913 = 99 652 s ≈ 27.7 h, plus at least three owner acts.
- PF sum: 13.1 + 13.1 + 30.6 + 6.9 ≈ 63.7 h, which is more than 59.5 h. This option therefore stops after stage 2.

## 6. Resources used up

- **Seeds.** Val: none. Test: none. Dev: seeds 5–14, and only under KEEP REDUCED.
- **Launches per day.**
  - At most 13 on one day: 1 + 6 + 6. I count the planner launch, which may not count against the cap (not verified).
  - At most 20 with one full relaunch, against the 40-per-day cap (`budget.yaml:3`).
  - Under DROP or PAUSED: zero.
- **Host time taken from Claim A.** None, by design: gate (b) plus the stop rule.
  - The B1 runs take 30 of 32 cores for about 13 h per stage under PF, at least 6.9 h under LB. That is at most about 26 h of idle host time.
  - Claim A's test run time is not measured.
- **Owner attention.** Two extra commits, M1 and M2, which compete with Claim A's preregistration and approval. Requests for them should come after Claim A's own requests.

## 7. What this adds over the reference option

- **The partial PRE-EMPTS hit is settled on evidence, not on an abstract,** under a rule fixed before the reading, at no host cost.
  - Stopping B1 on an unread abstract, as the reference option does, leaves it unclear whether the empirical claim in CLAUDE.md:3-5 survives.
  - Reviewers would ask that first.
- **Under KEEP REDUCED, one fresh convergence verdict at the merged code,** the only code a later B1 val sweep and test can use. At mean zero its equivalence tolerance in SD terms is about 0.070, against 0.0403 at n = 5 (derived; `decisions:24`). It runs in host time Claim A is not using.
- **The B1 wording is corrected in advance:**
  - the derivation is cited, not claimed;
  - MNSE is disclosed as not compared;
  - the "strongest baselines" sentence is narrowed.
- **What it does not add:** no OC-SAC-against-SAC sentence on 29 November.

## 8. Strongest objection, and answer

- **The objection.**
  - With B1's theory pre-empted and MNSE uncompared, the residual B1 is a narrow empirical comparison that cannot be tested before the deadline.
  - Spending two owner commits and up to about 26 host-hours on its first prerequisite buys nothing for 29 November.
  - A retry after UNDECIDED invites a charge of convergence-shopping.
  - The reference option, plus the full-text read, dominates.
- **The answer: largely right on the deadline view, and the option is built to collapse toward it.**
  - The read runs first and costs no host time.
  - DROP and PAUSED are the reference option.
  - The compute starts only under KEEP REDUCED, with a committed M2, an idle host and the cutoff not passed.
  - On shopping: the data are fresh and not pooled, n is a pre-data count, every rule is frozen, a third look is forbidden, and the two-look bound is disclosed. Two-stage equivalence designs (Potvin et al. 2008) pool stages and adjust α; this option does not pool, so it states the bound instead.
  - **Where the objection stands.** If the arena counts only sentences dated 29 November, the judge should choose the reference option with the full-text read, which is this option with gate (d) treated as never met.

## What would show this proposal was wrong

- The full text cannot be obtained, so PAUSED becomes permanent.
- The full text shows learned-agent experiments, so the result is DROP.
- The merge must change a step-1 function.
- M2 has not arrived by the cutoff.
- Claim A needs the host through 28 November.
- Step-1 launches run longer than PF.

## Prior work

- **"Equivalent Optima, Different Policies"**, AI Engineering 2026, DOI 10.53941/aieng.2026.100020. The partial PRE-EMPTS above. Seen only through the inbox entry, which is not yet promoted.
- **MNSE**, IEEE RA-L 2025, DOI 10.1109/lra.2025.3572822. A BASELINE in the same file. Its abstract only was reconstructed.
- **Potvin, DiLiberti, Hauck, Parr, Schuirmann and Smith,** "Sequential design approaches for bioequivalence studies with crossover designs", *Pharmaceutical Statistics* 7:245–262 (2008), DOI 10.1002/pst.294. Seen through a search record and a citing paper only: [Bohrium record](https://www.bohrium.com/paper-details/sequential-design-approaches-for-bioequivalence-studies-with-crossover-designs/811968577555922944-12039), [citing paper, AAPS J](https://link.springer.com/10.1208/s12248-014-9622-7).
- **Colas et al. 2018** ("How many random seeds?") and **Agarwal et al., NeurIPS 2021:** few-seed RL comparisons are unreliable. From memory, not fetched.

## Not verified, and contradictions

- **The coordinator's Claim A facts** are not verified, and they contradict `decisions:229`.
- **The checker merge.** Whether the branch touches line 123 or any other step-1 function is not verified.
- **The PRE-EMPTS paper** has not been read in full by anyone, me included. I did not fetch it, so that the assessment rule above stays fixed before the reading.
- **Smaller items, not verified:**
  - whether planner launches count against the daily cap;
  - the planner's run time;
  - val cells beyond the two instances;
  - run times at α > 0;
  - the deadline hour.
- **Contradiction between the criteria and an earlier decision.** `decisions:27` sets step 2 on seeds 0–4; criteria:69 supersedes it for a fresh step 1. I followed the criteria.
- **Contradiction in the record.** CLAUDE.md:17 says a PRE-EMPTS hit stops the work for the owner. Under the delegation at `decisions:239` that stop is the arena's to resolve. Reading the delegation that way is an inference, not the owner's words.

## Files relied on

- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\arena\2026-11-26-after-step1-stop-criteria.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\inbox\2026-11-26-claimA-B1-B2-prior-art-2.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\check_dev_protocol.py
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\train_agent.py
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\decisions\2026-11-26.md (verdict section not read)
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\arena\2026-11-25-step1-outcome.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\experiments\repro\weaver2024\VERDICT.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\CLAUDE.md
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\budget.yaml
- c:\Users\soumi\Occupancy-Coupled-Markov-Decision-Processes\seeds.yaml

Sources:
- [Sequential design approaches for bioequivalence studies with crossover designs (Bohrium record)](https://www.bohrium.com/paper-details/sequential-design-approaches-for-bioequivalence-studies-with-crossover-designs/811968577555922944-12039)
- [Citing paper, AAPS J](https://link.springer.com/10.1208/s12248-014-9622-7)
