# Proposal 2

Three findings change the decisions as the arena framed them:
- **The dev check cannot score one of its own pass criteria today.** It requires |H − target| / (H_max − H_floor) < 0.1 in the last window. But `train()` in `ocmdp/agents/oc_sac.py` logs only the temperature per window (`alpha_path`), not the batch entropy per window. Without that log the criterion would quietly be scored on evaluation-time entropy, which is a different quantity.
- **The absolute local path appears in more files than the eight train reports.** It is also in:
  - `runs/phase/phase_20261123_184302_13496.json` and `runs/phase/phase_20261123_190013_15752.json` (`seeds_file`);
  - nine `arena/2026-11-23-*-proposal-*.md` files (whether these are committed is unchecked);
  - `.claude/settings.json` (locked).
- **Rewriting history would break the provenance.** Every report records a git commit, and the `provenance/` patches are built on `7eb042f`. A history rewrite invalidates both.

The criteria do not contradict each other.

---

## Decision 1. Run order before val

**1. Recommendation.** Do the work in three steps, all on one platform (the "claim platform", see Decision 5). The separate 20-seed A/A launch is dropped: step 1 produces 60 A/A pairs as a by-product.

- **Step 0: code and tests, no launches.** This happens in the owner-side session, within proceed authority.
  - (a) Add `entropy_path`: the mean of the detached batch entropy over each window of `dual_update_every` updates, in the agent's own entropy space. It goes into `TrainResult` and into each job record in the report. It draws no random numbers, and a test confirms that training is unchanged bit for bit, as the checkpoint test already does.
  - (b) Add the isolation tests from Decision 3.
  - (c) Make the checkpoint and platform code changes from Decisions 4 and 5.
  - Pass: `.venv/bin/python -m pytest -q && .venv/bin/python experiments/validate.py --quick` is green, and the owner commits.
  - Steps 1 and 2 must run from that one code state. The reports' `code_diff_sha256` and commit fields make this checkable.
- **Step 1: convergence check, merged with the A/A test and a reproducibility check.** Six launches:
  - Command: `.venv/bin/python experiments/train_agent.py --tier dev --platform <label> --seeds 0 1 2 3 4 5 6 7 8 9 --sigmas 0 --class-sizes <CS> --quarters <Q> --set temperature=fixed alpha=0`
  - CS is `2 2` or `8 5 4 2 1`; Q is 12000, 24000 or 48000.
  - That is 120 jobs, or about 280 default-job equivalents. As a projection only: at the 426–649 s per default job logged on Modal in `runs/train/train_20261123_190717_20498.json`, it is roughly 35–50 CPU-hours. Not measured on the local host.
  - Six launches fit the 40-per-day limit in `budget.yaml`.
  - It must show three things:
    - (a) **Bitwise A/A.** At alpha 0 the two variants run the same algorithm. On every (seed, Q, instance), OC-SAC's and SAC's `parameter_sha256.final` must be equal: 60 of 60 pairs. A failure means that platform is not deterministic within a launch, because Step 0's test has already ruled out a leak through the variant label.
    - (b) **Reproducibility across launches.** For each (variant, seed, instance), the `initial` and `after_first_window` hashes must agree across the three Q launches. Run length changes nothing before the replay buffer wraps. A failure means seeds on that platform are draws.
    - (c) **Convergence.**
      - Define g(L) per seed as (uniform − agent) / (uniform − replanner), using the sampled expected failure rate on the same dev evaluation episodes. It is computed from the per-episode baseline values the report stores.
      - Use OC-SAC's run per seed if (a) passes; otherwise use the mean of the two agents per seed.
      - L* is the smallest L in {12k, 24k} where the upper end of the 95% Student-t interval for the mean of g(2L) − g(L) over seeds is below 0.05 on both instances. `runs_learned` at L must also be no more than one below its value at 2L.
      - If 24k→48k also fails, the check fails: training has not converged by 48k. Stop and take it to the owner.
    - If (a) or (b) fails, switch to the statistical A/A. Over the 20 pairs at L*, the t-interval of the mean OC-SAC − SAC difference must contain 0. A Fisher test must also find no imbalance between each arm and the CPU-feature group its jobs ran on (recorded in `host`). Failure means the platform and the arm are confounded, and the submission order has to be randomised before val.
- **Step 2: the dev check at L*,** as adopted in `arena/2026-11-23-temperature-outcome.md`.
  - Six new launches: fixed 0.1, fixed 3, and learned 0.89 with `alpha_lr` from Decision 2, on both instances, dev seeds 0-4, sigma 0.
  - The alpha-0 arm reuses step 1's L* launches (seeds 0-4 are a subset) only if the code hash is identical. Otherwise rerun all eight.
  - Pass criteria:
    - the adopted ones (widening rule, off the clamp, floor gap), with the last-window entropy criterion now scored from `entropy_path`;
    - the settling criterion from Decision 2;
    - Decision 3's tests passing at the same commit.
  - If L* = 48k, show the owner the projected cost of the val grid before val starts. The rule does not change.

**2. Against the criteria.**
- Validity: length, learning rate and widening are fixed by rules written before any outcome. They apply identically to both agents and never read the comparison between them. Everything runs on one platform.
- Fails loudly: each step has a named failure and a named consequence.
- Cost: the A/A costs nothing extra, and the dev check saves two launches by reuse.
- Authority: nothing locked is touched.
- Simplicity: the brief is 12 command lines with fixed arguments.

**3. Strongest objection.** Convergence measured at alpha 0 may not hold for high-temperature arms. And with 10 seeds the 0.05 bound may be underpowered, which pushes L* upward and makes val up to four times as expensive. Not fatal: the error runs in the conservative direction, and step 2 logs learning curves for every arm.

**4. What would show me wrong.**
- Step 2 learning curves at L* still falling in the last quarter for the fixed 3 or learned arms.
- A bitwise A/A pass on the local host followed by non-identical reruns later. That would mean hidden state, such as process reuse.

**5. Prior work.**
- A/A tests as a platform check before A/B tests: Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* (2020). Citation unchecked this session.
- Seed variance large enough that two groups of 5 seeds differ significantly: Henderson et al., "Deep RL that Matters", AAAI 2018.
- Choosing the number of seeds: Colas et al., arXiv 1806.08295.
- IQM and probability of improvement: Agarwal et al., NeurIPS 2021.
- A one-bit change producing full run-to-run variance: Summers & Dinneen, ICML 2021 (arXiv 2103.04514).

## Decision 2. Learned-temperature reachability

**1. Recommendation.**
- Change `alpha_lr` by a rule and keep `alpha_init` at 0.05.
- Rule: `alpha_lr` = 28 / n_updates, rounded up to two significant figures, where n_updates = L* − `learning_starts` + 1. Adam then moves log alpha by up to about 28 nats over a run, which is enough to cross the full clamp width (14 nats, from −12 to 2) within the first half of updates.
- Values: 12k → 2.6e-3; 24k → 1.3e-3; 48k → 6.0e-4. Pass it to both agents as `--set alpha_lr=<value>`.
- Do not run 3e-4. Its reach, about [0.002, 1.36], is arithmetic, not a question to measure. It excludes the fixed grid point 3, which a 0.89-of-range target may need (a conjecture, unchecked).
- The design is contingent, stated in advance:
  - **Settling criterion.** Over the last 25% of windows, the range of window-mean log alpha is below 1 nat.
  - **One rerun, at most.** If the learned arm fails that criterion, or the 0.1 entropy criterion, while log alpha oscillates around the target (the sign of H − target alternates), rerun the learned arm once at half the rate.
  - **Clamp.** If the temperature ends at a clamp, the target is unreachable in practice. By the adopted rule that is a bug: stop.
  - **Second failure.** Stop and take it to the owner.

**2. Against the criteria.**
- Validity: the rule depends only on run length and clamp width, is the same for both agents and is fixed before any outcome. The rerun reads only entropy, never the primary metric.
- Fails loudly: the settling criterion and the 0.1 criterion can both fail, and each failure has a meaning.
- Cost: zero extra launches, or two if the rerun is needed.
- Authority: `ocmdp/agents/` is not locked.
- Simplicity: one `--set` value.

**3. Strongest objection.** A rate about 9× larger makes alpha noisier, and that noise can feed back into the actor. This is not fatal: the settling criterion catches it, and the reference implementation lets log alpha travel much further than this rule does.

**4. What would show me wrong.** Use `alpha_path` and `entropy_path` from step 2.
- If the learned temperature crosses most of the clamp range within the first 10% of updates and then oscillates, the rate is too fast.
- If it is still trending in one direction at the end, the gradient sign is not consistent and the rate is not the problem.

**5. Prior work.**
- CleanRL `sac_atari.py` (fetched): log alpha starts at 0, Adam at 3e-4 with eps 1e-4, 5M steps, updates every 4 steps after 20k. That is about 1.2M updates, roughly 370 nats of possible travel against about 3.3 here.
- Haarnoja et al., arXiv 1812.05905 (alpha learning rate 3e-4); Christodoulou, arXiv 1910.07207 (SAC-Discrete). Details unchecked this session.

## Decision 3. Isolation check

**1. Recommendation.** Replace the floor-comparison rule with three pytest tests and one training-level requirement. The paired difference at matched alpha on (2,2) is still reported, with its interval, but descriptively only.
- **T1, bitwise at alpha 0.** In one process, train both variants with fixed alpha 0 for 2000 decisions, seeds 0 and 1, on both instances. Weight hashes must be equal at `initial`, `after_first_window` and `final`.
  - Failure: the variant label reaches something other than a zero-weighted entropy term, for example a NaN from 0·log 0.
- **T2, tied equal classes (2,2), alpha in {0.1, 3}.** Same weights (state loaded across the agents) and the same batch.
  - Actor-loss gradients are equal within rtol 1e-5.
  - Action entropy minus class entropy equals log 2 per sample, within 1e-6.
  - Soft critic targets differ by exactly gamma·(1 − done)·alpha·log 2.
  - Note: actor gradients are not equal after a full update, because the critics update first on targets that differ by that constant. This is why the test holds the critic fixed.
- **T3, unequal classes (8,5,4,2,1).** SAC's gradient minus OC-SAC's equals the gradient of alpha·Σ_c q_c log n_c, the named reward bonus, and the critic targets differ by gamma·(1 − done)·alpha·Σ_c q_c(s′) log n_c.
  - Failure: the list of named differences between the agents is incomplete.
- **Mutation check.** Each test must fail when the action entropy is scaled by 1.01.
- **Training level.** Decision 1's check (a) is the bitwise A/A requirement at training scale.

**2. Against the criteria.** Every check can fail, and each failure has a stated meaning. They cost CPU-seconds, sit within authority (tests: proceed) and need nothing from the collaborator.

**3. Strongest objection.** These tests verify that the code is equivalent, not that trained outcomes are. Not fatal: at matched alpha, trained outcomes are draws, so no outcome-level test with fixed thresholds can isolate the agents.

**4. What would show me wrong.** T2 passing while step 1's bitwise A/A fails on a deterministic host. That points to a difference outside the update, such as evaluation or replay.

**5. Prior work.** The identity "SAC = OC-SAC + alpha log n_class" was confirmed by the theory-checker (`decisions/2026-11-23.md`). Testing an implementation against a second one in this way is standard differential testing. I found no RL-specific citation.

## Decision 4. Checkpoint storage

**1. Recommendation.** Keep saving every agent; commit only the test-tier checkpoints; archive the rest outside git with their hashes; cap intermediate checkpoints.
- **Layout.** The driver writes to `runs/train/checkpoints/<tier>/<report>/…`.
- **`.gitignore`.** Replace `!runs/train/checkpoints/**` with `!runs/train/checkpoints/test/**`.
- **Archive.** After each dev or val launch, pack the folder with `tar -cf <report>.tar -C runs/train/checkpoints/<tier> <report>` and record the tar's SHA-256. The owner uploads it as a GitHub Release asset. Uploading is the owner's act. Assets must be under 2 GiB each, and GitHub sets no cap on total size or bandwidth.
- **Verification.** Add `experiments/verify_checkpoints.py <report>`. It recomputes every file's SHA-256 against the report and fails on any mismatch or missing file.
- **Intermediate cap.**
  - `--checkpoint-every` is allowed only on val, at no less than quarters / 4, which means at most 3 intermediates per agent.
  - It is used only when a preregistration names checkpoint choice.
  - There is no evaluator for intermediate checkpoints yet. Without one they cannot inform any choice. Build it before using the flag.

**2. Against the criteria.**
- Validity: the reports and launch records stay in git, and every file is tied to its report by hash. The test evidence is cloned with the repository.
- Fails loudly: the verifier fails on any missing or mismatched file.
- Cost: no checkpoint has been committed yet (`decisions/2026-11-24.md`), so none of this touches history. Growth is test-tier only, at about 1 MB per agent (993,394 bytes, measured with `checkpoint_bytes`, not a run).
- Authority: no installation needed. The upload and the owner's earlier instruction to keep Modal checkpoints in the repository are owner calls; this reverses that instruction for dev and val, so it needs the owner's confirmation.
- Simplicity: no LFS, so no pointer files to confuse a session with no memory of the project.

**3. Strongest objection.** Dev and val checkpoints exist only on the launching machine until they are uploaded, and could be lost. Not fatal: the reports keep their hashes. On a bitwise-reproducible claim platform they can also be retrained exactly.

**4. What would show me wrong.** Any archived dev or val checkpoint that is later needed for a claim. If that happens, val checkpoints belong in git as well.

**5. Prior work and second choice.** Git LFS is the second choice. Current GitHub docs give 10 GiB of free storage and 10 GiB of free monthly bandwidth on personal accounts, but every clone needs git-lfs installed. GitHub recommends keeping a repository ideally under 1 GB and under 5 GB, and blocks files over 100 MiB.

## Decision 5. Platform assignment

**1. Recommendation.**
- **Claim platform: the local host,** provided it passes step 1's checks (a) and (b). It is one CPU type, so both agents of a pair run on identical arithmetic, and seeds can reproduce bit for bit. Everything that feeds a preregistration runs there: step 1, the dev check, val, and test.
- **Second Modal account:** exploratory dev work outside the claim path (phase sweeps, diagnostics) and optional statistical replications after a test.
- **Fallback.** If the local host fails (a) or (b), the claim platform becomes the Modal account. Use the statistical A/A, and check the balance between arm and CPU group in every launch.
- **Val and test of one preregistration must share a platform.** "Same platform" means the same `--platform` label and the same host fingerprint: CPU features, package versions, thread count.
  - The preregistration records both.
  - `train_agent.py` refuses a test launch whose label does not match.
  - If the platform is lost between val and test, write a new dated preregistration that reruns val on the new platform before any test run.

**2. Against the criteria.**
- Validity: pairing within one arithmetic, and no interaction between platform and configuration between selection and test.
- Fails loudly: the label check refuses, and step 1 qualifies the platform.
- Cost: a single host is slow, but feasible at L* ≤ 24k.
- Authority: the check sits in `experiments/`, which is not locked.
- Simplicity: one platform for the whole chain.

**3. Strongest objection.** The local host's capacity limits throughput: a 48k val grid could take about a day of wall time. And if the platform is lost, val costs twice. Not fatal: validity ranks first.

**4. What would show me wrong.** A cross-platform replication after a test whose result disagrees with the claim platform beyond the A/A spread. That would show platform-specific behaviour and call for multi-platform reporting.

**5. Prior work.** Pham et al., ASE 2020, "Problems and opportunities in training deep learning software systems: an analysis of variance" (details unchecked). Summers & Dinneen, ICML 2021.

## Decision 6. Login name in committed reports

**1. Recommendation.** Leave `runs/` and history unchanged, and stop new cases.
- Add a test that scans tracked text under `runs/`, `arena/`, `decisions/` and `inbox/` for absolute home paths (`/Users/`, `/home/`). It fails on any file not in an explicit allowlist of the files that already have them: 8 train, 2 phase, and the arena proposals.
- Instruct future arena proposers to use repository-relative paths.
- Tell the owner the scope is larger than eight files (listed at the top), including `.claude/settings.json`, which only the owner can edit.
- **Owner's act, if the owner still wants the name removed:**
  - Do it before step 1, so no new report references old commit hashes.
  - Install git-filter-repo (an installation, so the owner's act).
  - Run `git filter-repo --replace-text <file>` with one line, `literal:==>`.
  - Commit `.git/filter-repo/commit-map` as `provenance/commit-map`, so the old hashes in reports still resolve.
  - Force-push, and re-clone every working copy.
  - Commit author metadata, which already carries a name on every commit, is a separate question.

**2. Against the criteria.**
- Validity: the recorded commits and `provenance/` stay intact.
- Fails loudly: the scan test fails on any new leak.
- Cost: zero.
- Authority: no runs edit and no history rewrite.
- Simplicity: nothing for the collaborator to do.

**3. Strongest objection.** A login name stays in a public repository, against the spirit of the no-names rule. Not fatal: the same identity is already in commit metadata, so a rewrite of paths alone removes little.

**4. What would show me wrong.** The owner judging the exposure unacceptable. Then the rewrite above, done now, is the cheapest point.

**5. Prior work.** GitHub's documentation on removing sensitive data from a repository's history. The git-filter-repo documentation (unchecked).

## Decision 7. README results section

**1. Recommendation.** Replace the subsection "### Trained agents (preliminary)" with the exact text below, which contains no numbers:

> ### Trained agents
>
> `experiments/train_agent.py` trains OC-SAC (entropy over mechanism classes) and standard SAC (entropy over drugs) on paired seeds. Each agent has one scalar temperature, fixed or learned toward a target inside the entropy range its own policy class can reach.
>
> No trained-agent result is reported yet. The reports in `runs/train/` up to `train_20261123_194806_29133.json` come from an earlier design with a state-dependent temperature, since removed; they are diagnostics of that design, and their seeds are not comparable with later runs. Results will appear here after a preregistered comparison has run, with intervals, the interquartile mean and the number of runs that learned.

If Decision 4 is adopted, the Layout row for `experiments/train_agent.py` becomes:

> Trains OC-SAC and standard SAC across seeds and compares them with the baselines; writes `runs/train/`, with every trained agent's checkpoint under `runs/train/checkpoints/<tier>/` (test-tier checkpoints are tracked in git; others are archived with their hashes in the report)

The toy-environment subsection stays, with one source line added under its noise table:

> Source: `runs/vv/vv_20261025_201416_713126_29699.json` (`noise_robustness.replanning_gain`, 20 seeds).

I checked the noise table against that file. Mean and half-width are 0.0077 ± 0.0061 at sigma 0.1, 0.0311 ± 0.0130 at 0.2 and 0.0540 ± 0.0402 at 0.4, which matches the README's rounding. Its 0.3467 matches `tuned_rotation` at sigma 0 and its uniform 0.389 matches `uniform` at sigma 0. Two caveats:
- Those intervals inherit the harness's rounded t table (`decisions/2026-11-23.md`).
- The subsection's other numbers (0.380, 0.532, 0.355, 0.349, and 89–98% / 0.344–0.348) are unchecked.

**Mismatch found:** the "Seed protocol" line says the paired differences have "a standard deviation of about 0.07 at σ = 0.4". The same file gives std 0.086 for `replanning_gain` at 0.4. Suggested replacement: "about 0.09 at σ = 0.4 (`runs/vv/vv_20261025_201416_713126_29699.json`)". Under CLAUDE.md, writing any number goes to run-auditor and adversarial-reviewer first.

**2. Against the criteria.**
- Validity: no claim from a superseded design or from unlogged runs.
- Fails loudly: every remaining number names its file.
- Cost: zero.
- Authority: the README is not locked. The numbers still need the two reviewers.
- Simplicity: short.

**3. Strongest objection.** The README loses its only learned-agent content. Not fatal: that content described a removed design.

**4. What would show me wrong.** An auditor tracing a removed README number to a logged run of the current design. None exists, because every logged train run predates the change (`decisions/2026-11-24.md`).

**5. Prior work.** None needed.

Sources:
- [GitHub: about releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- [GitHub: about large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub: Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [GitHub: removing files from history](https://docs.github.com/de/articles/removing-files-from-a-repositorys-history)
- [CleanRL sac_atari.py](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/sac_atari.py)
