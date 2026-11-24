# Outcome: operating decisions for the next training runs

Judged blind against `2026-11-24-operations-criteria.md`. The ranking was proposal 2 first, proposal 3 second and proposal 1 third, with 3 narrowly ahead of 1. Proposal 2 is adopted with the judge's amendments, under the owner's instruction that arenas settle owner-reserved decisions. Owner acts are listed at the end.

## Adopted

### 1. Run order before val

**Step 0: code and tests (done on 2026-11-24, in the working tree).**
- Per-window entropy logging (`entropy_path`).
- The isolation tests.
- Pair jobs.
- The checkpoint layout and cap.
- The preregistered-platform check.
- The temperature learning-rate rule.

**Step 1: convergence, bitwise A/A and reproducibility, on the candidate claim platform.**
- Six launches: fixed temperature 0, dev seeds 0–9, noise 0, class sizes (2, 2) and (8, 5, 4, 2, 1), each at 12 000, 24 000 and 48 000 decisions.
- **(a) Bitwise A/A.** OC-SAC and SAC have equal final `parameter_sha256` on all 60 (seed, length, instance) pairs.
- **(b) Reproducibility across launches.** The `initial` and `after_first_window` hashes of every (variant, seed, instance) agree across the three lengths.
- **(c) Convergence.**
  - Per seed, g(L) = (uniform − agent) / (uniform − every-quarter replanner), on the same evaluation episodes.
  - L* is the smaller of 12 000 and 24 000 for which the upper end of the 95% t-interval of the mean of g(2L) − g(L) over seeds is below 0.05 on both instances.
  - In addition, runs learned at L may be at most one fewer than at 2L.
  - If 24 000 → 48 000 also fails, stop for the owner.
- **If (a) or (b) fails:**
  - a statistical A/A: the interval of the mean OC-SAC − SAC difference over the 20 pairs at L* must contain 0;
  - a Fisher test of arm against CPU-feature group (from `host`);
  - pair jobs keep each pair on one host.

**Step 2: the temperature dev check at L*.**
- Six new launches: fixed 0.1, fixed 3, and learned at fraction 0.89, on both instances, dev seeds 0–4, noise 0.
- The fixed-0 cells are reused from step 1 if the code hash is identical; otherwise all eight are run.
- **Pass criteria:**
  - the widening rule;
  - the learned temperature off the clamp in every window and at evaluation;
  - |H − target| / (H_max − H_floor) < 0.1 in the last window, read from `entropy_path`;
  - the computed floor within 10% of the range of the lowest entropy reached;
  - settling: the range of log alpha over the last quarter of windows is below 1 nat;
  - the isolation tests pass at the same commit.
- Any failure stops the work for a new dated decision. There is no rerun at another rate.

The separate 20-seed A/A launch is dropped: step 1 gives 60 bitwise pairs.

### 2. Temperature learning rate (from proposal 3)

- `alpha_lr` is the smallest of {1, 2, 3, 5} × 10^k with `alpha_lr` × updates ≥ 3 × the distance from log `alpha_init` to the farther clamp edge.
- With the defaults (`alpha_init` 0.05, clamp [−12, 2]) that gives 3e-3 at 12 000 decisions, 2e-3 at 24 000 and 1e-3 at 48 000.
- It is the default of `SACConfig.alpha_lr`, and an explicit value still overrides it.
- Any failure of the step-2 criteria stops the work.

### 3. Isolation

- **T1, bitwise at temperature 0.** 2 000 decisions, seeds 0 and 1, both instances. Hashes agree at initialisation, after the first window and at the end.
- **T2, tied classes (2, 2), temperature 0.1 and 3, critic held fixed.** The two agents' actor gradients are equal, and drug entropy minus class entropy is log 2 per sample.
- **T3, unequal classes.** The gradient difference equals the gradient of alpha · Σ_c q_c log n_c, the named bonus.
- **Mutation check.** Scaling the drug entropy by 1.01 breaks T2 and T3. T1 cannot detect it, because at temperature 0 the entropy carries no weight.
- The paired difference at matched temperature is reported descriptively only.

### 4. Checkpoints

**This reverses the owner's 2026-11-24 instruction to keep every checkpoint in the repository, for dev and val. It needs the owner's confirmation; restoring the line `!runs/train/checkpoints/**` in `.gitignore` undoes it.**

- Written to `runs/train/checkpoints/<tier>/<report>/`. Only test-tier checkpoints are tracked in git.
- Dev and val folders are archived as tar files with their SHA-256 recorded, and the owner uploads them as GitHub Release assets.
- `experiments/verify_checkpoints.py` checks any restored folder against its report.
- `--checkpoint-every` is refused except on val, at no less than a quarter of the run (at most three intermediates per agent). It is used only when a preregistration names checkpoint choice, and only after an evaluator for intermediate checkpoints exists.

### 5. Platform

- **Claim platform.** The local host, if it passes step 1 (a) and (b). Otherwise the second Modal account, with pair jobs (amendment from proposal 1), the statistical A/A and the balance check.
- **The second Modal account** otherwise runs exploratory dev work outside the claim path.
- **Val and test of one preregistration share a platform.**
  - The preregistration names it in a `platform:` line, and the scripts refuse a test launch from any other.
  - If the platform is lost between val and test, a new dated preregistration reruns val before any test launch.

### 6. Login name in committed files

- No history rewrite and no edit under `runs/`.
- The live source is fixed: paths outside the repository are recorded by name only.
- The arena records of 2026-11-23 are fixed forward.
- A test scans every tracked or new file for home-directory paths in both forms (`/Users/<name>/` and `-Users-<name>-`), with the 10 reports that already carry one allowlisted by SHA-256 (amendments from proposals 3 and 1).
- Optional owner act, if removal from history is wanted:
  1. Run `git filter-repo --sensitive-data-removal --replace-text <file>`. The file holds one line, `literal:<the repository's absolute prefix>/==>`.
  2. Commit `.git/filter-repo/commit-map` as `provenance/commit-map-<date>.txt`, so the commit hashes recorded in reports still resolve.
  3. Force-push, ask GitHub to purge cached views, and have every clone re-clone.

### 7. README

- "Trained agents" is replaced with text that states no result yet and contains no numbers.
- A sentence withdraws the never-logged validation numbers (amendment from proposal 3).
- The noise table gets its source line.
- The seed-protocol line no longer states a spread. "About 0.07" was wrong; the file holds 0.0860.

The brief tells the collaborator to count launches against the daily budget by hand (amendment from proposal 1).

## Where the proposals disagreed

- **Convergence.** Proposal 1 used one 48 000-decision launch with intermediate checkpoints; it is cheaper but needs a checkpoint evaluator. Proposal 3 used one long launch read from its learning curve. The adopted design uses three lengths, which buys the cross-launch reproducibility check.
- **Learning rate.** Proposal 1 used 1e-3 with `alpha_init` 0.17. Proposal 2 used 28 / updates, with one rerun at half the rate. The adopted rule is proposal 3's, with a stop instead of a rerun.
- **Checkpoints.** Proposal 1 put every checkpoint in external release assets. Proposal 3 used Git LFS. The adopted design splits by tier.

## Not satisfied by any proposal

- **The exact edit to `.claude/settings.json`.** No proposal gave it. On inspection the file holds only the glob `//Users/**/.ssh/**` and no login name, so no edit is needed.
- **Training length at other temperatures.** The length is chosen at temperature 0, and none of the proposals tests before val whether it carries over to temperature 3 or to the learned temperature. The dev check's learning curves are the only evidence.

## Convergence

All three proposals agreed on six points:
- convergence before the dev check;
- dropping the separate 20-seed A/A;
- an arithmetic rule for the learning rate;
- exact tests in place of the seed-chaos rule;
- no history rewrite;
- a README with no new numbers.

Identical briefs produce correlated answers, so this agreement is weak evidence.

## Owner acts

- **Confirm or revert the checkpoint storage split** (item 4).
- **Optionally, the history rewrite** (item 6).
