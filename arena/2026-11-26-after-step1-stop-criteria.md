# Criteria: what follows the step-1 stop

Revision 6. Revisions 1 to 4 failed the standing reviewer's check with six, four, two and one blocking findings, and revision 5 passed. Revision 6 adds the B1 pre-emption section below, which was missed when the arena opened.

## The decision

- **The stop.** Step 1 (the owner's six-launch design, `decisions/2026-11-26.md`, "Step 1 is the owner's six-launch design") ended in STOP, with no L*. Both instance verdicts are UNDECIDED. Step 2 was not launched.
- **Rung 1.** `experiments/repro/weaver2024/VERDICT.md` records: "Rung 1 not reproduced: the published result is not specified precisely enough to fix a reproduction target."
  - The owner's decision of 2026-11-26 overrides two stops (`decisions/2026-11-26.md`, "The owner's decision overrides two stops"):
    - CLAUDE.md's "report the size of the discrepancy and stop";
    - the step-1 outcome's "if rung 1 fails, the ladder stops".
  - So steps 1 and 2 and later rungs may proceed after rung 1. That also lifts rung 1's gate on Claim A's test launch.
- **The design's own rule, fixed before any data:** "the checker stops for the owner instead of asking for more seeds or length".
- **Authority.** The owner made the agentic arena the sole decision maker on 26 November. The instruction is recorded verbatim in `decisions/2026-11-26.md`, "Owner instructions of 2026-11-26 evening: the arena decides". That delegation covers scope, so the arena adopts. CLAUDE.md's own text still says scope changes stop for the owner; the recorded instruction supersedes it for this decision.

## Information firewall

- **What proposers and the judge get:** "STOP, no L*, both instance verdicts UNDECIDED". Nothing else from the step-1 reports: no step classes, intervals, g values, learned counts, spreads or per-seed runtimes. A diverged run ends early, so per-seed runtimes carry outcome information.
- **Enforced by instruction, not by access.** Proposers and the judge can read the repository. They must not open the six step-1 reports under `runs/train/` (listed in `decisions/2026-11-26.md` under "Step-1 verdict") or the step-1 verdict section of that file. They must not run `experiments/check_dev_protocol.py`.
- **Who has seen more.**
  - The checker's 26 November run, recorded in `decisions/2026-11-26.md` under "Step-1 verdict", was made in an earlier session. Its printout carries the intervals, and it is stored nowhere in the repository.
  - The session that wrote these criteria did not run the checker, and its context does not hold that printout. It has read the verdict section's text: the exit code, the verdict and the class of each length step, but no number. Whether the same person or agent saw the printout in the earlier session is not established.
  - The statistics verifier recomputed the verdict on 26 November and saw the numbers.
  - The standing reviewer read the reports' `args`, `seconds`, `variant`, `seed`, `pid` and `job_in_process` fields. Its searches also returned, unrequested, two summary `diverged` fields from one report. It did not report their values, and they informed none of its findings.
- **The verdict class is itself an outcome**, and withholding numbers cannot hide it. So every proposal must include a pre-commitment table (criterion 1).

## What the decision affects downstream

- **B1 (OC-SAC against SAC) is the only learned-agent claim in scope.**
  - B2 was cut on 26 November (`decisions/2026-11-26.md`, "Deadline and cuts"). Reinstating it is outside this arena.
  - B1's route to a test launch is the whole chain:
    1. a qualified L*;
    2. step 2: six launches, decisions "Step 2 runs at L*";
    3. a val sweep over CLAUDE.md's candidate set: fixed temperatures {0, 0.01, 0.03, 0.1, 0.3, 1, 3} and learned temperatures at c ∈ {0.77, 0.89}, for each agent, on 10 val seeds, over every cell;
    4. a preregistration carrying the full table;
    5. the owner's commit and `approvals/<slug>.ok`;
    6. one test run.
- **Claim A is planner-only and independent of L*.** Its dev phase sweep `phase_20261126_190606_33244` has run since 19:06 on 26 November with `workers: 30`. Its end time is not measured. Its test launch needs its own committed preregistration and approval file.
- **The deadline** is the morning of 29 November. These criteria were written on the evening of 26 November.
- **Host and budget.**
  - One Windows host with CPU torch. It has 32 logical cores (`nproc` on this host).
  - `budget.yaml` allows 40 training launches per day across all tiers; 13 were used on 26 November (`runs/ledger.txt`).
- **Code state.** Step 1 ran at `4788c5a`. HEAD is `7eb31b9`, committed at 19:08 on 26 November; it adds only `decisions/` and `runs/` files to `d847dc5`. `git diff 4788c5a 7eb31b9 -- ocmdp experiments pyproject.toml seeds.yaml`, over all of `CODE_PATHS` (`experiments/train_agent.py:205`), is empty.
- **A new step 1 needs a code edit and a human commit.**
  - `experiments/check_dev_protocol.py:69-73` hard-codes `INSTANCES`, `LENGTHS = (12_000, 24_000, 48_000)`, `SEEDS = [0, 1, 2, 3, 4]` and `CONDITION`.
  - It rejects reports on other seeds (line 129), on any other set of lengths (lines 139-144), and from uncommitted code (line 133).
  - **The lengths stay at {12 000, 24 000, 48 000}.** `step1()` writes the three steps and the L* rule into its code (lines 321-333), so any other length set would need a new L* rule. Writing one now would be a design choice made after the outcome.
  - **The checker may receive only these changes:**
    - `SEEDS`;
    - the `code-standards` merge's changes outside the step-1 functions.
  - **Everything else in the step-1 functions stays identical to `4788c5a`:** `_step1_cells` (apart from `SEEDS`), `_reference`, `_defects`, `_gain_shares`, `_classify`, `_step` and `step1`. The standing reviewer diffs those functions against `4788c5a` before the commit, over the `code-standards` merge as well.
  - The edit and the new preregistration must then be committed.
  - **No minimum on U − R.** The bar is exactly the `4788c5a` bar, which raises a defect when U ≤ R (lines 202-203) and sets no other floor.
    - A new minimum cannot be fixed in advance here. Evaluation episodes depend only on the tier and the episode count (`experiments/train_agent.py:1127`), and U and R must agree across reports to within 1e-12 (lines 229-236).
    - So a new dev step 1 has the same U − R as the logged step-1 reports, and those reports already decide any minimum.
- **Launch arguments stay as the step-1 reports recorded them:**
  - `--episodes 10` and `--sigmas 0`;
  - delay and lam at their defaults (`null` in `args`);
  - `--condition c-given`;
  - `--set temperature=fixed alpha=0`;
  - `--workers 5`, the local backend, and the instances and lengths above.
  
  Changing `--episodes` after a verdict that spread made UNDECIDED would be an outcome-conditioned choice on the same footing as the seed count, so it is fixed too.
- **One launch commit per new step 1.**
  - `_reference` requires the planner-restart report to come from the step-1 commit and code (lines 177-178). So a new step 1 needs one more launch: a planner reference rerun at its commit.
  - `_defects` raises a DEFECT if the reports' commits differ (lines 239-240).
  - Any commit between the planner reference and the last step-1 launch, even one that adds only `decisions/` and `runs/` files (as `7eb31b9` did), therefore spends the step.
  - Each option names one launch commit for its planner reference and all its step-1 launches.
- **Step 2 runs on step 1's seeds.** Step 2 reuses step 1's fixed-0 cells, which needs the same seeds and the same code (`decisions/2026-11-26.md`, "Step 2 runs at L*", sets dev seeds 0–4 for the original step 1). A new step 1 on disjoint seeds therefore needs step 2 on those same seeds, and each option gives step 2's seed set.
- **A B1 test needs per-agent temperatures, so it needs a merge.**
  - At `4788c5a`, `--set` applies to both variants (`experiments/train_agent.py:1039`).
  - `--set-oc-sac` and `--set-sac` exist only on the unmerged `code-standards` branch. That branch also changes how diverged runs are imputed and the report schema.
  - `train_agent.py` is in `VAL_CODE_PATHS` (`train_agent.py:112`), so val and test must run on the same merged code.
  - Step 2 must share code with step 1 at L*, so step 1 and step 2 must also run on code that matches val and test in `VAL_CODE_PATHS`.
- **Committer.** Commits and merges are the owner's act and are not delegated (CLAUDE.md, "I make every commit"). The owner last committed at 19:08 on 26 November (`7eb31b9`). Whether the owner will commit again before 29 November, and how quickly, is not known.
- **Every option names, for each stage of its chain:** the commit it runs on, the commits and merges it waits for, and its action if no commit arrives.

### Costing rule (fixed here, before any proposal)

- **Step-1 wall clock**, measured from launch-record `started` to the report's file-modification time, for 5 seeds at `--workers 5`.
  - The (2, 2) 12 000 report ran in the 02:16 batch, beside the five launches lost at the reboot, with 30 workers in total. Its 04:52 finish is also recorded in `decisions/2026-11-26.md`.
  - The other five ran together from 05:25 with 25 workers, and fewer as jobs finished. Their times come from file modification times.

  | Cell (class sizes, `args.quarters`) | Launch to report (s) | Slowest pair job (s) |
  |---|---|---|
  | (2, 2), 12 000 | 9 324 | 4 580 |
  | (2, 2), 24 000 | 15 758 | 9 150 |
  | (2, 2), 48 000 | 19 301 | 16 849 |
  | (8, 5, 4, 2, 1), 12 000 | 19 454 | 13 228 |
  | (8, 5, 4, 2, 1), 24 000 | 22 396 | 19 014 |
  | (8, 5, 4, 2, 1), 48 000 | 24 913 | 22 276 |

  "Slowest pair job" is the largest, over seeds, of the sum of that seed's two run `seconds`, a derived figure. A pair job trains OC-SAC and then SAC in one process. The launch-to-report time also covers the per-process canary and the driver's fingerprint.
- **Costing rules, one for every proposal:**
  - **While the Claim A sweep runs,** about 2 cores are free. A launch at the fixed `--workers 5` shares them. So a cell of n seeds costs at least ⌈n/2⌉ times its slowest-pair time, and the option states how much it slows the sweep (criterion 5).
  - **After the sweep writes its report,** cost from the launch-to-report column, also as a lower bound. The sweep's end time is not measured, so each chain is costed under both rules.
  - Assume no linear scaling across lengths or seed counts.
  - Count launches against the 40-per-day cap.

## What would make one option clearly better

1. **Compliance, a pass-or-fail gate.** An option fails if any of these does not hold:
   - **The convergence bar stays fixed.** Any new step 1 keeps every rule at its `4788c5a` value in `experiments/check_dev_protocol.py`:
     - `DELTA = 0.05`;
     - the two-sided 95% t-interval;
     - a(n) = max(1, ⌊n/10⌋) and the ⌈n/2⌉ majority (line 285);
     - the divergence imputation;
     - the reference precondition (`SPREAD_LIMIT = 0.02`, `MIN_PLANNER_SEEDS = 5`);
     - `BASELINE_TOLERANCE = 1e-12`;
     - `INSTANCES` and `CONDITION`.
     
     The lengths stay at {12 000, 24 000, 48 000}, with the L* rule of `step1()`. The only free choices are the seeds (dev, disjoint from 0–4) and the seed count, and the proposal states that both were chosen after the verdict was UNDECIDED. Changing any fixed item, or any part of the checker beyond the changes listed in the code-state section, fails the option.
   - **No pooling.** No new seed or length is pooled with the seed 0–4 reports, and those reports are not re-checked under a changed rule.
   - **A preregistration fixed before launch.** Any new step 1 runs under a new dated preregistration, fixed in full before launch. Its terminal rule can fail, with no extension clause, so another UNDECIDED means stop and report.
   - **The commit dependencies are named,** as the code-state section requires, including whether step-1 fixed-0 cells can be reused at that commit.
   - **The pre-commitment table is disclosed.** It gives the option's action under each step-1 verdict: DIVERGING, DECLINING, FLAT_BELOW_MAJORITY, RISING, UNDECIDED and converged. The table is a disclosure, not the test; the fixed bar above is the test.
2. **What it licenses, in fixed wording.** The option states the exact sentence the project can write on 29 November under each outcome.
   - Step 1 is reported as "convergence not established at n = 5, δ = 0.05, over 12 000–48 000 decisions, c-given only, at σ = 0 and fixed temperature 0, on instances (2, 2) and (8, 5, 4, 2, 1)". It is never "did not converge" or "unstable". The wording leaves a slot for each step's interval, filled later from `runs/` through the checker.
   - Rung 1 is reported in VERDICT.md's own sentence, quoted above, with the owner's override of the two stops disclosed.
   - Any L* is stated as unqualified for no-c, and the no-c confound is stated as open.
   - The most a B1 test licenses is "outperforms the strongest baselines that could be constructed, tuned at parity".
3. **Severity preserved.** The option does not weaken comparators, lower the convergence bar or narrow scope to make a pass likelier.
4. **Feasibility of the whole chain.** For each stage up to a B1 test, the option gives its launch count and wall-clock cost under the costing rule. Each stage that cannot finish before the deadline is named.
5. **Resources it uses up.** The option states:
   - val seeds (100–199) and test seeds (1000–1299);
   - launches per day;
   - host time taken from Claim A.
   It must not slow or pre-empt the Claim A sweep or Claim A's test path without saying so and justifying it.
6. **Information per unit of compute** for the claims in scope.

**The reference option** is to stop the B1 line, report step 1 and rung 1 in the fixed wording, and give the host to Claim A. Every other option is scored against it, and must show what it adds over it.

## Hard constraints

- **No tuning toward reproduction or toward a win.** Rung 1's track is closed, and its authors' code is not run.
- **The step-1 seed 0–4 reports stay as they are,** reported as UNDECIDED.
- **The superseded 2026-11-25 stage 2 cannot be reinstated as pre-specified.** It was defined against a stage 1 of seeds 0–9 in both conditions, and seeds 5–9 and no-c never ran.
- **Any new step 1 is chosen because the step-1 verdict was UNDECIDED.** So its result stands on its own data only, and its report says so.
- **Human acts stay human:**
  - no test-seed launch, `approvals/` file, commit, push or message;
  - no edit to protected paths (`ocmdp/eval`, `envs`, `core`, `baselines`, `theory`, `seeds.yaml`, `budget.yaml`, `.claude/`, `CLAUDE.md`), to a committed preregistration, or to anything under `runs/`.
- **Seed tiers:** dev for diagnostics, val for selection, test once per preregistration.
- **Null and negative results are reported as results,** with intervals, never as trends.
- **No number goes into a draft, table or abstract** until run-auditor and adversarial-reviewer both pass (CLAUDE.md, Authority table).

## Evidence that would settle it empirically

None cheap. The question is which line of work to pursue. The one empirical move, a new step 1, is itself one of the options. It costs hours of host time under the costing rule, and it would be chosen because of the observed verdict. So the arena is held.

## Added after the arena opened: B1 has a partial PRE-EMPTS hit

- **The hit.** At 16:44 local on 26 November, a literature scout filed a partial PRE-EMPTS hit on B1: `inbox/2026-11-26-claimA-B1-B2-prior-art-2.md`, section "PRE-EMPTS".
  - The paper is "Equivalent Optima, Different Policies" (AI Engineering 2(2), 21 October 2026).
  - In a static log-linear setting, it derives that redundant action representations shift the entropy-regularised optimum by a conditional-divergence term. That is B1's theory component.
  - Only the abstract was read, and the paper has no RL experiments.
  - The same file lists MNSE (IEEE RA-L 2025) as a BASELINE whose theory must be read before B1's is claimed.
- **Its effect.** CLAUDE.md says a PRE-EMPTS hit stops the work and is reported. B1 is stopped on it, separately from the step-1 stop. The arena decides both.
- **Every option states what B1 becomes:**
  - kept in reduced form, as an empirical SAC test with the derivation cited rather than claimed;
  - dropped;
  - or paused until the full texts are read.
  An option that continues B1 gives its changed wording. An option that ignores the hit fails criterion 2.
- **Nothing in `inbox/` is cited in a draft until it is promoted** (CLAUDE.md, "Gathered material").
