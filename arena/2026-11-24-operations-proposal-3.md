# Proposal 3

## Findings from the material that change the decisions

- **F1. The dev check's tracking rule can't be computed today.** The rule is "|H − target| / (H_max − H_floor) < 0.1 in the last window". But `train()` in `ocmdp/agents/oc_sac.py` records only `alpha`, `xi`, `episode_failure` and the checkpoints for each window; `_update` returns only alpha. So the training-window entropy is never logged, and `experiments/train_agent.py` writes only `alpha_path` and the evaluation entropy. This has to be added before any launch.
- **F2. At fixed alpha 0 the two agents should be bit-identical on deterministic arithmetic.** `actor_loss` adds `alpha.detach() * entropy`, and the soft target adds `alpha * entropy(next_probs)`. `categorical_entropy` clamps before the log, so the entropy and its gradient stay finite, and 0 × a finite number is exactly 0. Both agents also start from the same weights (already tested).
- **F3. On tied equal classes (2, 2), the actor gradients are not identical inside a real update.** In the action-entropy target, SAC's soft target exceeds OC-SAC's by alpha·Σ_m q_m log n_m, which is alpha·log 2 on (2, 2). That shifts SAC's critic, the actor then uses the freshly updated critic, so the actor gradients differ from the first update. The identity only holds for the actor loss with the critic held fixed.
- **F4. The absolute path is in more places than eight reports.**
  - 10 reports carry it, not 8: the 8 `runs/train` reports from `train_20261116_190634_8365` to `train_20261123_194806_29133`, plus `runs/phase/phase_20261123_184302_13496.json` and `phase_20261123_190013_15752.json`.
  - 9 committed arena proposals contain the `/Users/<login>/` form, 59 times in all: `arena/2026-11-23-{scope,temperature,reproduction}-proposal-{1,2,3}.md`.
  - `.claude/settings.json` contains it too (owner-only file).
  - The untracked `arena/2026-11-24-design-criteria.md` has a second form, `-Users-<login>-`. I didn't scan the rest of the repository for that form (unchecked).
- **F5. A `--seeds-file` outside the repository would still record its absolute path.** `launch_arguments` writes `str(value)` for any path outside the repository.
- **F6. Nothing evaluates a saved checkpoint after the fact.** Only tests call `load_checkpoint`.
- **F7. Modal job time.** Per-job time for 12 000 decisions on (2, 2) was 425–649 s in `runs/train/train_20261123_190717_20498.json` (no gap diagnosis). It was 1713–2240 s in `runs/train/train_20261123_194806_29133.json`, which ran with `--diagnose-gap`. Time on (8, 5, 4, 2, 1) is not measured.

---

## Decision 1. Run order before val

**1. Recommendation.**

*Step 0. No launch; local code and tests, within proceed authority.*
- Log the mean batch entropy for each update window (`entropy_path`) beside `alpha_path` (F1). Test that it draws no random numbers.
- Add the tests from Decision 3.
- Fix F5: record only the file name and its SHA-256 for paths outside the repository.
- Add the checkpoint guard from Decision 4.
- Pass: the full test suite.

*Step 1. Convergence and platform-determinism launch, before the dev check.*
- Fixed alpha 0, dev seeds 0–9, `--quarters 48000`, sigma 0, no intermediate checkpoints. One launch per class size, (2, 2) and (8, 5, 4, 2, 1): 2 launches, 40 jobs, on the claim-candidate platform (Decision 5).
- This replaces the proposed three launches at 12k, 24k and 48k. The replay buffer never wraps before 12 000 decisions, and nothing else depends on `total_quarters`. So, on deterministic arithmetic, the first 12k and 24k decisions of a 48k run are exactly those of the shorter runs, and the `learning_curve` (25-episode blocks) gives all three lengths from one run.
- **Convergence pass, rule fixed now:**
  - Let d(Q) be, for each seed, the mean of the last two `learning_curve` blocks ending at Q minus the same at 48k.
  - The training length L is the smallest of {12k, 24k} for which, on both instances, the upper end of the 95% t-interval of mean d(Q) over 10 seeds is below 0.05 × (uniform − every-quarter replanner), taken from the same report's sigma-0 baselines. Otherwise L = 48k.
  - A failure at 12k means every earlier 12k report was undertrained, and every later step runs at L.
- **Determinism pass:** OC-SAC and SAC final `parameter_sha256` are equal for all 20 seed pairs (F2).
  - A mismatch with the Step 0 bitwise unit test passing means the platform's own arithmetic is not deterministic across jobs. That platform then loses its claim to being the reproducible one (Decision 5).
  - A mismatch on a platform whose jobs report identical `host` fields means a code path leaks entropy at alpha 0.
- **Cost, derived from F7:** about 4 × (425–649 s) × 20 jobs ≈ 9.5–14.5 CPU-hours for the (2, 2) launch. The (8, 5, 4, 2, 1) launch is not measured.

*Step 2. The adopted dev check, run at length L with `alpha_lr` from Decision 2, on the claim platform.*
- 8 launches: candidates {fixed 0, 0.1, 3; learned 0.89} × {(2, 2), (8, 5, 4, 2, 1)}, sigma 0.
- Dev seeds 0–4, except the **(2, 2) fixed-0.1 launch uses dev seeds 0–19**. This is the A/A null (Decision 3), giving 110 jobs in total.
- **Pass criteria:**
  - The adopted widening rule.
  - The adopted floor-versus-lowest-entropy check.
  - Learned temperature: off the clamp in every window and at evaluation; |H − target| / range < 0.1 in the last window, read from `entropy_path`; spread of window log-alpha over the last quarter of training < 1 nat.
  - The alpha-0 cells repeat the hash-equality check.
  - The null check from Decision 3.
- Any failure stops the work: no val launch until a new dated decision exists.

*Dropped: the separate 20-seed alpha-0 A/A.*
- At alpha 0 both agents optimise the same objective and are bitwise identical on deterministic arithmetic (F2). Their paired difference is then 0 by construction, and on Modal it measures only container divergence.
- It says nothing about the spread of the null when the objectives differ numerically. That spread is what power needs, and the (2, 2) fixed-0.1 extension measures it.
- Its integrity role is covered by Step 1's 20 hash pairs.

**2. Against the criteria.**
- *Validity:* all steps are on dev, on one platform, with every rule fixed before data. L is chosen from a curve at alpha 0, not from agent comparisons, and is identical for both agents.
- *Fails loudly:* every check above has a stated failure and meaning.
- *Cost:* 10 launches, within the 40-a-day limit in `budget.yaml`. The convergence check costs about half of the three-launch design, and the A/A adds 30 jobs instead of 40.
- *Authority:* only dev seeds and code within proceed authority.
- *Simplicity:* two commands with `--quarters 48000`, then eight dev-check commands.

**3. Strongest objection.** Convergence at alpha 0 need not carry over to alpha 3 or to the learned temperature. Not fatal: every dev-check job logs its own learning curve. A candidate whose curve still falls in the last quarter of training is flagged in the val preregistration, not re-lengthened after the fact. Training-episode failure is also a proxy for evaluation failure, because it is sampled under the policy on training starting states.

**4. What would show me wrong.**
- At L, the dev-check curves of fixed 3 or of the learned candidate still falling.
- The A/A null SD at 20 seeds being so wide that it cannot inform power.
- Hash mismatches at alpha 0 on a platform with homogeneous hosts.

**5. Prior work (none of these fetched in this session, so citations are unchecked).**
- Colas et al., "How many random seeds?" (arXiv 1806.08295): seed counts for comparing RL agents.
- Agarwal et al. 2021 (arXiv 2108.13264): the IQM and probability-of-improvement reporting CLAUDE.md already uses.
- Summers & Dinneen, ICML 2021 (arXiv 2103.04514): bit-level perturbations cause as much variance as changing the seed. This is the reason the null needs a real numerical difference rather than alpha 0.

---

## Decision 2. Learned-temperature reachability

**1. Recommendation.**
- Keep `alpha_init` 0.05 for both agents, with no extra candidate.
- Set `alpha_lr` by a rule fixed now: the smallest value in {1, 2, 3, 5} × 10^k with `alpha_lr` × N ≥ 3 × d_max.
  - N = L − `learning_starts` + 1.
  - d_max = 9.0 nats, the distance from log 0.05 ≈ −3.0 to the farther clamp edge, −12.
  - This gives 3e-3 at L = 12k, 2e-3 at 24k and 1e-3 at 48k.
- Pass it as `--set alpha_lr=…` to both agents and every learned candidate, on dev and val alike. Record it in `decisions/`.
- Contingent, as the arena allows: if the learned candidate fails the Step 2 tracking or stability criteria on any job, stop and write a new dated decision. Do not tune `alpha_lr` in a loop.

**2. Against the criteria.**
- *Validity:* the setting comes from arithmetic about Adam's maximum travel (about `alpha_lr` per update when the gradient's sign is consistent), not from any outcome. It is identical across agents.
- *Fails loudly:* the tracking and stability criteria.
- *Cost:* no extra launches, whereas a reachability candidate would add 2.
- *Authority:* a `SACConfig` field in `ocmdp/agents/`, which is not locked.
- *Simplicity:* one `--set` flag.

**3. Strongest objection.** A tenfold faster temperature may oscillate against a critic that is still learning. It also departs from the familiar 3e-4.

Not fatal:
- The stability criterion catches oscillation.
- The reference implementations used 3e-4 over runs of hundreds of thousands to millions of updates, so their travel capacity was far larger than here. Matching capacity to run length is the closer analogue to them than matching the raw rate. (CleanRL `sac_atari` uses Adam on log alpha at `q_lr` 3e-4; I did not re-fetch it, so this is unchecked.)

**4. What would show me wrong.**
- The last-quarter log-alpha spread exceeding 1 nat.
- Tracking failing at the new rate while the window entropy is still moving: an undertrained policy, not a reachability problem.

**5. Prior work (unchecked).**
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (arXiv 1812.05905): automatic temperature tuning.
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (arXiv 1910.07207).
- Huang et al., "CleanRL" (JMLR 2022).

---

## Decision 3. Isolation check

**1. Recommendation.** Replace the "within the seed-chaos floor" rule with three checks that can fail.

- **T1, bitwise at alpha 0.** A unit test with a short run (for example 1 500 decisions) of each variant on one seed in one process, for (2, 2) and (8, 5, 4, 2, 1).
  - Pass: equal final `parameter_sha256`.
  - Failure means some entropy path leaks at alpha 0, such as a NaN times 0 or a reward bonus.
- **T2, exact identities at matched alpha > 0.** Use identical networks, a frozen critic and several random batches.
  - (a) The difference between the two agents' actor-loss gradients equals the gradient of alpha·Σ_m q_m log n_m. That is zero for (2, 2) and (3, 3), and nonzero for (8, 5, 4, 2, 1). Tolerance: relative 1e-5.
  - (b) The soft targets differ by exactly γ(1 − done)·alpha·Σ_m q_m(s′) log n_m.
  - Failure means the entropy spaces differ by more than the named difference: a wrong assignment, tied drugs not receiving equal probability, or a mismatched clamp.
  - This corrects the arena's proposed one-update test (F3).
- **T3, the statistical null in the dev check.** The (2, 2) fixed-0.1 cell at 20 dev seeds.
  - Pass: the two-sided 95% t-interval of the mean paired difference (OC-SAC − SAC) contains 0.
  - Report the paired SD, which feeds power.
  - Failure means either a bug that T1 and T2 missed, or that SAC's constant critic shift of alpha·log 2 per step changes learning. Either way, (2, 2) cannot serve as a null. Stop and report.

**2. Against the criteria.**
- T1 and T2 are deterministic: no launch, and each has an exact pass condition.
- T3 is a real test with a 5% false-alarm rate, which I state rather than hide.
- All of it is within proceed authority.

**3. Strongest objection.** T3 tests at only one matched temperature, and the paired SD under the selected temperatures may differ. Not fatal: it is a bug detector and a first spread estimate, not the power calculation itself.

**4. What would show me wrong.** T2 passing while T3's mean is far from 0 across repeated dev draws.

**5. Prior work.** Summers & Dinneen 2021 (above). The tied-network identity itself is already recorded in `decisions/2026-11-23.md` (theory-checker entry).

---

## Decision 4. Checkpoint storage

**1. Recommendation.** Git LFS for `runs/train/checkpoints/**/*.pt`, final checkpoints only by default, and a cap on intermediate checkpoints.

- **Now, within authority:** remove the line `!runs/train/checkpoints/**` from `.gitignore`. Until LFS exists, an `add -A` then cannot put binary weights into plain history, which could only be undone by a rewrite. Each report still holds every checkpoint's SHA-256.
- **Owner acts, exact changes:**
  1. Install git-lfs with the system package manager.
  2. Run `git lfs install` and `git lfs track "runs/train/checkpoints/**/*.pt"`.
  3. Restore the `.gitignore` exception line and commit `.gitattributes` with it.
  - The collaborator also needs git-lfs to hand over files.
- **Clone cost:** the brief tells clones to use `GIT_LFS_SKIP_SMUDGE=1`.
- **Cap, a code change in `experiments/train_agent.py`:**
  - Refuse `--checkpoint-every` on dev.
  - On val, allow it only when `quarters // checkpoint_every <= 4` (at most 3 intermediate checkpoints per agent), and only when the val preregistration declares checkpoint selection.
- **Weights:** keep the full seven-network checkpoint. The final-hash-equals-checkpoint test depends on it.

**2. Against the criteria.**
- *Validity:* seeds are draws and reruns diverge, so the saved weights are the only way to re-evaluate an agent. Keeping every final checkpoint, tied to its report by hash, keeps every launch auditable.
- *Cost:* GitHub's current docs list 10 GiB of LFS storage and 10 GiB of bandwidth a month on Free accounts, so several hundred MB per val grid fits. Plain git would make every clone fetch it all forever.
- *Authority:* the install is named as the owner's act.
- *Simplicity:* one rule, with no external store to manage.

**3. Strongest objection.** It depends on the owner installing software, and on bandwidth if clones fetch the weights. Until then the files exist only on the launching machine. Not fatal: the hashes in each report make a later upload verifiable. If LFS is declined, the fallback is an external archive per launch (for example a release asset or a Zenodo record at rung 6), with the same hashes. Any upload is an owner act.

**4. What would show me wrong.** Monthly LFS bandwidth approaching the quota, or clone time growing. Measure with `git count-objects -vH` and the LFS usage page.

**5. Prior work.** GitHub docs: [Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs), [about storage and bandwidth usage](https://help.github.com/articles/about-storage-and-bandwidth-usage), [about Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). Some older GitHub pages still show the earlier 1 GB limits.

---

## Decision 5. Platform assignment

**1. Recommendation.**

*Choosing the claim platform:*
- The claim platform is the local host if Step 1's 20 alpha-0 pairs are all hash-equal there. Otherwise it is the second Modal account.
- This rule is fixed now and depends on no agent outcome.

*What runs where:*
- The claim platform runs everything from Step 1 onward: the dev check with its A/A, the val grid and the test runs.
- The other platform runs only dev diagnostics, and its results are never pooled with the claim platform's.

*Val and test share a platform:*
- The val selection and the test run of one preregistration must share the platform label and the package versions.
- The preregistration names both. The run-auditor checks the test report's `platform` and `backend.packages` against it.
- Optionally, `check_test_launch` refuses a mismatch. That needs a preregistration field, which is the owner's to add.
- Modal pins packages to the launching machine's local versions, so the collaborator's `.venv` must match the recorded pins.

**2. Against the criteria.**
- *Validity:* pairing and selection happen within one arithmetic. A deterministic platform also makes a test report reproducible by rerunning, which Modal cannot offer (7 of 10 jobs).
- *Fails loudly:* the hash rule.
- *Simplicity:* one platform per preregistration.

**3. Strongest objection.** The local host's core count, and so the val grid's wall time, is unknown (unchecked). A Modal pool may be far faster. Not fatal: the rule falls back to Modal automatically if local isn't deterministic, and local throughput is a cost, not a validity, issue.

**4. What would show me wrong.** Local reruns of an identical launch not reproducing bit for bit even though the pair hashes are equal.

**5. Prior work.** PyTorch's reproducibility notes say results are not guaranteed across platforms or releases (unchecked; not fetched this session).

---

## Decision 6. Login name in committed reports

**1. Recommendation.** No history rewrite, and no edit under `runs/`.

*Fix forward, within authority, in the working tree for the owner to commit:*
- Replace the absolute repository prefix with repository-relative paths in the 9 arena proposals (F4).
- Fix F5.
- Add a test that scans tracked text files outside `runs/` for home-directory paths in both forms, `/Users/<name>/` and `-Users-<name>-`, and fails on a hit.
- Record the 10 affected reports (not 8) in `decisions/`.

*Owner acts:*
- `.claude/settings.json` (locked) also contains the path.
- If removal from history is wanted, the exact act is:
  1. Install git-filter-repo.
  2. Run `git filter-repo --replace-text <file>` with a regex rule for both path forms.
  3. Commit the resulting `.git/filter-repo/commit-map` as `provenance/commit-map`.
  4. Force-push.
  5. Have every clone re-clone, and ask GitHub to purge cached refs.

**2. Against the criteria.**
- *Validity is the deciding criterion.* A rewrite changes every later commit hash, while every report's `git.commit` and the `provenance/` patches (applied onto `7eb042f`) point at the current hashes. A rewrite would orphan all of them unless a commit map is kept.
- *Exposure:* the commit author metadata already identifies the account holder at least as specifically, so rewriting paths alone removes little.

**3. Strongest objection.** The name stays public in history. Not fatal for validity, but it is the owner's privacy call, and that is why the exact rewrite is given above.

**4. What would show me wrong.** If the owner considers the path more identifying than the author metadata.

**5. Prior work.** GitHub's guide "Removing sensitive data from a repository" (unchecked; not fetched).

---

## Decision 7. README results section

**1. Recommendation.** Replace the "Trained agents (preliminary)" subsection with this exact text. It contains no numbers.

> ### Trained agents
>
> No trained-agent result exists yet for the current agent design, in which each agent has one scalar temperature, fixed or learned toward a target inside the entropy range its own policy class can reach. The protocol these runs follow is in `arena/2026-11-23-temperature-outcome.md`.
>
> The reports in `runs/train/` up to and including `runs/train/train_20261123_194806_29133.json` trained a superseded design with a state-dependent temperature. They are diagnostics of that design, not results: the same training seed starts the current design from different weights, and its temperature objective differs. In `runs/train/train_20261123_190717_20498.json`, `runs/train/train_20261123_190750_20538.json` and `runs/train/train_20261123_194806_29133.json`, both agents' median temperature at evaluation sits at the clamp floor in every job, so those runs do not compare entropy spaces.
>
> An earlier version of this section compared three entropy targets on validation runs that were never logged. Those numbers are withdrawn.

Before the owner adopts it:
- The run-auditor confirms the clamp-floor sentence against the three files. It is taken from the arena's facts and I did not recheck it.
- The run-auditor maps every number in "Results on the toy environments" to a file under `runs/`. I found 0.3467, 0.856 and 0.355 in `runs/vv/vv_20261123_183549_650277_10740.json`. The rest are unchecked; any number that isn't found is removed.

**2. Against the criteria.** It states no number and no claim the logs don't support, and it says what was withdrawn. Changing what the public README says is the owner's act, after the adversarial reviewer has run.

**3. Strongest objection.** It leaves the README without any agent result. Not fatal: per CLAUDE.md, "not measured" is the correct state.

**4. What would show me wrong.** The run-auditor finding the clamp-floor sentence false for some job.

**5. Prior work.** None needed.

---

**Contradictions and empirical parts.** The criteria are not contradictory. Two parts are empirical, and my proposal makes each depend on a rule fixed now: whether the learned temperature reaches its target (Decision 2, settled by the dev check) and whether the local host is deterministic (Decision 5, settled by Step 1).

Sources:
- [GitHub Docs: Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- [GitHub Docs: about billing for Git LFS](https://docs.github.com/en/billing/using-the-new-billing-platform/about-billing-for-git-large-file-storage)
- [GitHub Docs: about storage and bandwidth usage](https://help.github.com/articles/about-storage-and-bandwidth-usage)
- [GitHub Docs: about Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
