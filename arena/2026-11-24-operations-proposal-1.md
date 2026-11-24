# Proposal 1

Four findings change the premises of the arena:
- **The two temperature clamp checks cannot currently fail.** At alpha_init 0.05 and alpha_lr 3e-4, log alpha can only move within about [-6.3, 0.3], well inside the clamp [-12, 2] (`ocmdp/agents/oc_sac.py`). The adopted check that a learned temperature "stays off the clamp" therefore has the same defect as the isolation rule in decision 3.
- **The login-name path is in more files than eight.** The absolute home-directory path appears in 10 reports under `runs/`: 8 in `runs/train/` and 2 in `runs/phase/`. It also appears in the nine `arena/2026-11-23-*-proposal-*.md` files and in `.claude/settings.json`. I did not check whether git tracks all of these; git status suggests the arena files are committed.
- **Nothing exists yet to evaluate saved checkpoints.** Only the tests call `load_checkpoint`; no script in `experiments/` does. As things stand, intermediate checkpoints cannot be scored, which matters for both the convergence check and val checkpoint choice.
- **No checkpoints exist yet.** `runs/train/checkpoints/` is empty, so any storage choice costs no history rewrite.

---

## Decision 1. Run order before val

**1. Recommendation.**

*Step 0: local checks, no launch.* Run the tests from decisions 2 and 3, plus the pair-job change from decision 5. All must pass before anything is launched.

*Step 1: convergence check, on the second Modal account.*
- Settings: fixed alpha 0, both agents, dev seeds 10-19, `--sigmas 0`, class sizes (2, 2) and (8, 5, 4, 2, 1).
- Launch: `--quarters 48000 --checkpoint-every 12000`, one launch per instance (2 launches). Each agent then has checkpoints at 12k, 24k and 36k decisions, plus the final agent at 48k.
- Why one 48k run works: at 12k decisions it is in the same state as a 12k run. The replay buffer has not wrapped and the sampling stream is the same (`ocmdp/agents/oc_sac.py`), so each change is measured along one trajectory with no fresh noise from a new run.
- Needs new code first: a checkpoint-evaluation script, written and tested before launch. It verifies each file's SHA-256 against the report and evaluates it on the tier's episodes. If the script is not ready, fall back to six separate launches at 12k, 24k and 48k; on Modal each of those is a new random draw, so the pairing is weaker.
- Pass rule, per instance and per agent:
  - Compute the change in expected failure rate from 12k to 48k decisions, paired by seed.
  - Pass if the upper limit of its 95% t-interval is below m, and `runs_learned` at 12k is no lower than at 48k.
  - m = 0.1 × (uniform − every-quarter replanner), both read from the same report's baselines. It is fixed here, before any data exists.
- If it fails or is inconclusive, every later step uses the shortest length in {24k, 48k} that passes the same test against 48k. If 24k also fails, use 48k and record that 48k itself is not shown to be converged.
- The same launch also serves as an A/A check (decision 3d): at alpha 0 the two agents are the same program, so the paired hashes must match at all four checkpoints.

*Step 2: the adopted dev check, on the second Modal account.*
- Its 8 launches as written: dev seeds 0-4, `--sigmas 0`, both class sizes. It uses the length L fixed in step 1 and the temperature settings from decision 2.
- Pass rules adopted earlier: the widening rule; the learned temperature stays off the clamp in every window and at evaluation; |H − target| / (H_max − H_floor) < 0.1 in the last window; the computed floor is within 10% of the range of the lowest entropy reached.
- Added rule: each learned temperature path reverses direction at least once. Otherwise "on target in the last window" could be a temperature still travelling that happens to pass through.
- The chaos-floor rule is replaced by the checks in decision 3.

*Proposed 20-seed per-platform A/A test: not run as a separate step.*
- With pair jobs (decision 5), both agents at alpha 0 on one host are the same arithmetic, so the A/A distribution is a single point at zero. Step 1 already checks this bitwise (20 pairs × 4 checkpoints), and so do the dev check's fixed-0 cells (10 pairs).
- If pair jobs are not adopted: run it on the val/test platform before val, at alpha 0, L decisions, dev seeds 20-39. It must show identical final hashes wherever the two jobs' host records match; the spread of the paired differences is reported as a descriptive chaos floor.

*Order:* step 0, then step 1, then step 2, then val. Convergence comes first because the length fixes the number of temperature updates, and so what the dev check tests.

**2. Criteria.**
- **Validity:** the training length is chosen at alpha 0, where the two agents are identical, so the rule cannot favour either. The margin is fixed now. Every launch leaves a launch record.
- **Fails loudly:** each check has a stated failure and meaning. Convergence failing means agents trained for 12k decisions are undertrained, and a reviewer would attack the baselines. A hash mismatch means some difference between the agents other than the entropy term.
- **Cost:** Modal jobs at 12k took 425-649 s each (`seconds` fields in `runs/train/train_20261123_190717_20498.json`).
  - Step 1 is 2 instances × 10 seeds × 2 agents × 4 units of 12k = 160 units, roughly 19-29 CPU-hours.
  - Step 2 is 80 units.
  - Not running the separate A/A saves about 40 units.
- **Within authority:** dev seeds only. Ten launches in total, under the 40 per day in `budget.yaml`. Launches on another account bypass the hook, so the brief must tell the collaborator to count them by hand.
- **Simplicity:** three blocks of commands in fixed order, each with one pass line.

**3. Strongest objection.** Convergence at alpha 0 may not carry over to alpha 3, where the entropy term may slow learning. Not fatal: the dev check logs learning curves at every candidate, and a slope still falling in the last quarter of training is visible there. The other objection is that step 1 costs twice what one agent would, because both run at alpha 0. Its price is that it doubles as the long-run bitwise A/A.

**4. What would show me wrong.**
- Step 1 hashes match, but the dev check's fixed-0 pairs diverge on one host: the bitwise check is too short.
- Learning curves at alpha 3 still improving in the last quarter of L: alpha 0 was the wrong proxy.

**5. Prior work.**
- Two sets of seeds of the same algorithm can differ significantly, which is the reason for an A/A test (Henderson et al. 2018, arXiv:1709.06560; UNCHECKED, cited from memory).
- Seeding reproduces only on identical hardware (Agarwal et al. 2021, https://arxiv.org/pdf/2108.13264).
- Deterministic implementations do not survive a change of hardware (Nagarajan and Warnell, https://ar5iv.labs.arxiv.org/html/1809.05676).

---

## Decision 2. Learned-temperature reachability

**1. Recommendation.**
- Change both settings now, identically for both agents: `alpha_init` = 0.17, the geometric midpoint of the non-zero fixed grid (√(0.01 × 3)), and `alpha_lr` = 1e-3.
- Rule: from its start, log alpha must be able to reach either end of the non-zero grid (0.01 or 3) within 30% of the updates, and either clamp edge within the run. So lr ≥ max(ln(3/0.17) / (0.3 U), (ln 0.17 + 12) / U).
  - With U = 11 001 updates, that is max(8.7e-4, 9.3e-4), so 1e-3.
  - If decision 1 lengthens training, keep 1e-3; more updates only make both targets easier to reach.
- Add a unit test that asserts lr × U covers both distances for the configured length.
- No reachability candidate at 3e-4: its ceiling of about 1.36 follows from arithmetic, not from an outcome.
- Contingency, stated as the arena asks: whether the target is actually reached is settled by the dev check. If any learned run fails a reach check, classify it from `alpha_path`:
  - pinned at the clamp: the target lies outside what the policy can reach in practice, which the protocol calls a bug;
  - still moving one way: the budget is too short;
  - oscillating: the learning rate is too high.

  Then stop for a new dated decision. No retuning inside the session.

**2. Criteria.**
- **Validity:** the setting is chosen from an arithmetic bound before any learned-temperature data exists under the current design. It is the same for both agents and does not depend on either agent's target.
- **Fails loudly:** at 3e-4 the clamp check cannot fail. At 1e-3 the clamp is reachable from both sides (about 3 750 updates to the top, 10 250 to the floor), so it can.
- **Cost:** zero extra launches.
- **Within authority:** `ocmdp/agents/` is not locked, and this arena covers the question.
- **Simplicity:** the new values become the defaults, so the brief needs no flags.

**3. Strongest objection.** A reviewer may say it departs from the published defaults: CleanRL starts alpha at 1 with Adam at 3e-4. It is not fatal. CleanRL takes on the order of a million alpha updates (5M steps, an update every 4), so its lr × U is hundreds of nats. This rule keeps the property that configuration has, reachability, rather than its number. A higher rate risks oscillation, and the reversal and last-window checks would show it.

**4. What would show me wrong.** In the last quarter of training, an alpha that swings by more than a factor of 2 between windows, or an entropy that never settles within 10% of the range despite reversals.

**5. Prior work.**
- CleanRL `sac_atari.py`: log_alpha starts at 0; Adam at q_lr 3e-4, eps 1e-4; target scale 0.89; 5M steps (https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/sac_atari.py).
- Christodoulou 2019, discrete SAC: https://arxiv.org/pdf/1910.07207
- Xu et al. 2021: a constant low target forces alpha to collapse early (https://arxiv.org/abs/2112.02852v1).
- Zhou et al. 2022: discrete-SAC failures come from policy instability rather than the target (https://arxiv.org/pdf/2209.10081).
- Haarnoja et al. 2018, arXiv:1812.05905 (UNCHECKED).

---

## Decision 3. Isolation check

**1. Recommendation.** Replace the chaos-floor rule with four checks. The first three are unit tests run in float64.
- **(a) Tied classes, (2, 2).** Build both agents from one seed; their initial hashes are equal, as already tested. Give both the same batch at alpha in {0.1, 3}.
  - The actor-loss gradients must match to a relative 1e-10.
  - The critic targets must differ by exactly γ·α·log 2·(1 − done) per sample. Under policies uniform within classes, the drug entropy equals the mechanism entropy plus log 2.
- **(b) Unequal classes, (8, 5, 4, 2, 1).** The gradient difference SAC − OC-SAC must equal α·∇E[Σ_c q_c log n_c], the named extra term, to 1e-10. This pins down that the named difference is the only one.
- **(c) Bitwise at alpha 0.** Both agents, one seed, one process, `total_quarters` 1 400, which is one 400-update window. The final `parameter_sha256` values must be equal.
- **(d) In the runs.** In the convergence check and in the dev check's fixed-0 cells, both agents of every seed must have identical final hashes whenever their host records match; with pair jobs that is always.

Failure meaning:
- (a), (b) or (d) failing: something besides the entropy differs between the agents, for example random-number consumption, masking or normalisation.
- (c) failing alone: a 0 × log 0 NaN path or nondeterminism on the host. I did not check how `categorical_entropy` handles zero probabilities.

The paired difference at matched alpha is still reported, labelled descriptive, never as a check.

**2. Criteria.** Each check fails on a single observable mismatch, with a fixed meaning (fails loudly). The tests take seconds and the run checks cost nothing extra (cost). Tests are within proceed authority. They run with the existing pytest command (simplicity). They measure the agents' identity directly instead of borrowing a noisy floor (validity).

**3. Strongest objection.** One-update identity says nothing about long-run behaviour, where the critic offset under (2, 2) and alpha > 0 makes the runs diverge chaotically. Not fatal: that divergence is chaos the design expects, and the bitwise check (d) covers the long run at alpha 0.

**4. What would show me wrong.** If (a) and (b) pass but agents at alpha 0 still diverge on one host over 48k decisions, the asymmetry sits in a path the one-update test does not reach.

**5. Prior work.** The identity that drug-entropy SAC equals OC-SAC plus a reward bonus α·log n_class(a) was already confirmed in this project (`decisions/2026-11-23.md`). I found no external precedent for gradient-identity A/A tests between RL variants.

---

## Decision 4. Checkpoint storage

**1. Recommendation.** Use external storage with hashes, plus a cap. These steps need the owner, because they reverse the owner's instruction to keep checkpoints in the repository.
- In `.gitignore`, remove the line `!runs/train/checkpoints/**`, so `*.pt` stays ignored everywhere.
- After each launch, pack `runs/train/checkpoints/<report>/` into `<report>-checkpoints.tar` and attach it to a GitHub Release named `checkpoints-<report>`. Uploading is the owner's act.
- Verification is extract-and-compare against the per-file SHA-256 values the report already holds. Nothing under `runs/` is edited.
- For the release rung, deposit a frozen copy in a DOI archive (limits UNCHECKED).
- Keep a second copy on the owner's disk.

Cap on intermediate checkpoints:
- dev check: none;
- convergence check: 3 per agent;
- val: at most 3 per agent, at L/4, L/2 and 3L/4, the same for every candidate and agent;
- test: final only.

Fewer networks per file: not now. It would change the checkpoint format under a tested reload path. Revisit only if storage costs ever matter.

**2. Criteria.**
- **Validity:** the hashes in the reports bind each file, and substitution is detectable.
- **Fails loudly:** a hash mismatch on extraction.
- **Cost:** zero repository growth. GitHub states no limit on a release's total size or bandwidth: each file under 2 GiB, up to 1000 assets per release. GitHub recommends repositories below 1 GB and strongly below 5 GB, which several hundred MB per val grid would approach. Git LFS on a free personal account gives 10 GiB of storage and 10 GiB of bandwidth per month, and the bandwidth is spent by every clone.
- **Within authority:** the `.gitignore` reversal and every upload are owner acts. LFS would add an install on two machines.
- **Simplicity:** one tar and one upload per launch.

**3. Strongest objection.** Link rot: a deleted release loses the weights, and the hashes cannot bring them back. Not fatal: the results live in the reports, and the second copy covers loss.

**4. What would show me wrong.** If checkpoints are reloaded often in analysis, the download step becomes a real cost, and LFS would have been cheaper in practice.

**5. Prior work.**
- GitHub releases: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- Large files on GitHub: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Git LFS billing: https://docs.github.com/en/billing/concepts/product-billing/git-lfs

---

## Decision 5. Platform assignment

**1. Recommendation.**
- **Pair jobs.** Run both agents of one (seed, sigma, candidate) in one job, so they share one container or host and one host record. This is a change to `train_and_evaluate` and `training_jobs` in `experiments/train_agent.py`, with a test that each pair records one host block.
- **Steps 0-2** (local tests, convergence check, dev check): the second Modal account. Dev results serve only mechanical checks and are never paired with val.
- **Val and test of one preregistration: the same platform, required.**
  - The preregistration names that platform's label and backend block.
  - `check_test_launch` refuses a test launch whose `--platform` differs from the val launches that preregistration cites. This is a code change in `experiments/`; the preregistration field is the owner's to freeze.
  - A val grid never spans platforms, because the platform would then be confounded with the candidate.
- **Which platform for val and test.** First run a self-reproduction check on the local host: two identical dev launches, `--quarters 1400 --seeds 0 1 --sigmas 0`, whose final hashes must all match. If it passes, val and test run there, because the run-auditor can then rerun any test seed bit for bit. If it fails, val and test run on the second Modal account with pair jobs.

**2. Criteria.**
- **Validity:** pairing within a host removes CPU-type variation from every paired difference, and one platform per preregistration removes platform from selection. Modal is known not to reproduce itself: identical pinned launches matched on only 7 of 10 jobs.
- **Fails loudly:** a platform-label mismatch refuses the launch; a pair job's single host block shows mismatches directly.
- **Cost:** pair jobs keep the same CPU time and double each container's wall time.
- **Within authority:** test-tier rules stay owner-gated.
- **Simplicity:** one fixed rule plus one self-check.

**3. Strongest objection.** The local host's throughput is unknown (UNCHECKED). A val grid of about 9 candidates × 2 agents × 10 seeds × cells could take many hours there. Not fatal: the fallback is Modal with pair jobs.

**4. What would show me wrong.** A local host that passes the short self-check but fails to reproduce at length L. Rerun one val seed as a spot check.

**5. Prior work.** Reproducibility holds only on fixed hardware (Agarwal et al. 2021, Nagarajan and Warnell 2018, cited above).

---

## Decision 6. Absolute paths in committed reports

**1. Recommendation.** No history rewrite. Contain the problem instead:
- Add a test that scans `runs/` for absolute paths. The 10 existing files are allowlisted by their SHA-256, so any new file, or any change to one of those 10, fails.
- New arena and decision files use repository-relative paths only.
- **Owner act:** replace the absolute path in `.claude/settings.json`, which is locked, with a repository-relative one. I did not check the exact line.

If the owner still chooses a rewrite, do it once, before the collaborator's first clone and before any val report exists:
- Run `git filter-repo --sensitive-data-removal --replace-text <file>`, version 2.47 or later. The file maps the repository's absolute prefix to nothing, as `literal:<prefix>/==>`.
- Commit `.git/filter-repo/commit-map` as `provenance/commit-map-<date>.txt`, then force-push.
- Ask GitHub Support to purge cached views, and have the collaborator re-clone.

**2. Criteria.**
- **Validity:** a rewrite renames every commit, breaking each report's `git.commit` and the base of the provenance patches (`7eb042f`). It would also break the rule that nothing under `runs/` is edited.
- **Fails loudly:** the scan test fails on any new or altered report.
- **Cost:** none.
- **Within authority:** the rewrite, the settings edit and any force-push are owner acts.

**3. Strongest objection.** Privacy, which is not among the arena's criteria. If the repository is public (UNCHECKED), the string has already been exposed. GitHub warns that rewritten commits stay reachable through cached views and forks, so a rewrite would not fully recall it. Whether the login name exposes more than the commit metadata and repository URL already do is the owner's judgement. The criteria and that rule conflict here, and I flag it rather than resolve it.

**4. What would show me wrong.** The login name reveals something the commit metadata does not. That makes a rewrite worth its provenance cost, and only now.

**5. Prior work.** GitHub's guidance on removing sensitive data: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository

---

## Decision 7. README results section

**1. Recommendation.** Replace the whole `### Trained agents (preliminary)` subsection with the text below, which contains no numbers. Writing numbers requires both the run-auditor and the adversarial reviewer to pass first, and neither has run.

> ### Trained agents
>
> No trained-agent result is reported yet.
>
> `experiments/train_agent.py` trains OC-SAC, whose entropy is taken over resistance mechanisms (the pushforward entropy), and standard SAC, whose entropy is taken over drugs. Each agent has one scalar temperature, either fixed or learned toward a target inside the entropy range its own networks can reach (`ocmdp/agents/entropy_range.py`). The temperature is a per-agent hyperparameter: every agent is tuned over the same candidates on the same validation seeds, and a preregistration fixes the selection rule before any test run.
>
> The reports in `runs/train/` up to and including `train_20261123_194806_29133.json` come from earlier temperature designs, including a state-dependent temperature head, which has been removed. These reports are kept as diagnostics. They are not results. Their seeds also do not match later runs, because removing the temperature head changed the networks' initial weights.
>
> Numbers will appear in this section only from test-seed runs of a committed preregistration, with the report file named beside each one.

The numbers in "Results on the toy environments" may trace to `runs/vv/`; I did not check. The run-auditor should trace each one to a named file, or remove it.

**2. Criteria.** No unlogged number remains and nothing is claimed early (validity). It is within authority, since it writes no numbers. It is plain to a reader.

**3. Strongest objection.** The public README loses its only trained-agent content. Not fatal: that content rested on runs that were never logged.

**4. What would show me wrong.** The auditor finds that the previous numbers do trace to logged runs. They would still describe the removed design.

**5. Prior work.** None needed.

Sources:
- [CleanRL sac_atari.py](https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/sac_atari.py)
- [Target Entropy Annealing for Discrete SAC](https://arxiv.org/abs/2112.02852v1)
- [Revisiting Discrete SAC](https://arxiv.org/pdf/2209.10081)
- [SAC for Discrete Action Settings](https://arxiv.org/pdf/1910.07207)
- [Statistical Precipice](https://arxiv.org/pdf/2108.13264)
- [Deterministic Implementations for Reproducibility in Deep RL](https://ar5iv.labs.arxiv.org/html/1809.05676)
- [GitHub: About releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- [GitHub: Large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub: Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
