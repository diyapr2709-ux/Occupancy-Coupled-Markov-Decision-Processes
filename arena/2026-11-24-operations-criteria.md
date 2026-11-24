# Arena: operating decisions for the next training runs

Fixed before any proposal is written. The judge scores against these and nothing else. Training runs are now launched by a collaborator on a second Modal account or on a local machine whose GPU is unused (the code is CPU-only).

## The decisions

Answer each separately and concretely enough to act on.

1. **Run order before val.** The run plan has step 1 adopted: the temperature dev check from `arena/2026-11-23-temperature-outcome.md`. It is 8 launches: fixed alpha 0, 0.1 and 3 and learned at fraction 0.89, both agents, dev seeds 0-4, no noise, class sizes (2, 2) and (8, 5, 4, 2, 1). The standards review proposes two more checks:
   - a per-platform A/A test: both agents at fixed alpha 0, 20 dev seeds;
   - a convergence check: alpha 0, 10 dev seeds, at 12 000, 24 000 and 48 000 training decisions, run before the dev check.

   Decide which run, in what order, with what seeds and budgets, and what each must show to pass.
2. **Learned-temperature reachability.** Adam with `alpha_lr` 3e-4 moves log alpha by at most about 3.3 over a default run of 11 001 updates. From `alpha_init` 0.05, the learned temperature therefore stays inside about [0.002, 1.36] whatever its target, and may never reach the target. Decide whether the dev check keeps `alpha_lr` and `alpha_init` as they are, changes them (to what, and by what rule), or adds a reachability candidate.
3. **Isolation check in the dev check.** The arena's rule ("at matched fixed alpha on (2, 2) the paired difference sits within the seed-chaos floor") cannot fail, because the logged floor varies tenfold. Decide its replacement, for example a one-update test that both agents' actor gradients are identical at matched alpha on tied classes, plus a bitwise check at alpha 0.
4. **Checkpoint storage.** Every trained agent is saved as about 1 MB in `runs/train/checkpoints/`, tracked in git. A default launch adds about 20 MB, and a val grid several hundred MB. Decide among:
   - plain git;
   - Git LFS (installing it is the owner's act);
   - external storage with hashes kept in the reports;
   - fewer saved networks.

   Also decide whether to cap intermediate checkpoints.
5. **Platform assignment.** Decide which of the two platforms runs which steps, and whether the val selection and the test run of one preregistration must share a platform.
6. **Eight committed reports carry an absolute local path** (`args.seeds_file`) that contains a login name. `CLAUDE.md` forbids editing anything under `runs/`; rewriting git history and force-pushing are the owner's acts. Decide what to do.
7. **README results section.** It describes the removed state-dependent temperature head and quotes validation runs that were never logged. Decide what replaces it. If the proposal contains text, give the exact text. It may use only numbers that are in files under `runs/`, each with the file named.

## What it affects downstream

The collaborator's first launches and the handoff brief; the repository's size and the cost of cloning it; whether reports from two platforms can ever be compared; what the public README claims.

## Facts the proposals must take as given

- **Seeds are draws.** Two identical pinned Modal launches (`runs/train/train_20261123_190717_20498.json`, `train_20261123_194806_29133.json`) trained identically on 7 of 10 jobs; 3 diverged within the first 400 updates. Across local and Modal runs, per-seed results differ.
- **Temperatures.** In all 30 jobs of `train_20261123_190717_20498`, `train_20261123_190750_20538` and `train_20261123_194806_29133`, the median evaluation temperature of both agents sits at the clamp floor, e^-12. Those reports are from the superseded state-dependent temperature design.
- **The code now** (`ocmdp/agents/oc_sac.py`, `ocmdp/agents/entropy_range.py`, `experiments/train_agent.py`):
  - one scalar temperature, fixed or learned toward floor + fraction × (maximum − floor) of the entropy range the agent's networks can reach;
  - every report records schema 2: launch id, platform label, per-job host and arithmetic, and weight hashes;
  - every launch writes a launch record before any job starts;
  - the scripts refuse test-tier launches without the owner's approval.
- **The repository** is on GitHub. Its whole history so far is a few tens of MB.
- **Locked.** `CLAUDE.md`, `.claude/`, `seeds.yaml`, `budget.yaml`, `ocmdp/eval/`, `ocmdp/envs/` and `ocmdp/core/` can be changed only by the owner. Nothing under `runs/` is edited.

## Criteria, in priority order

1. **Validity.** Results that later carry a claim stay interpretable: pairing within a platform, every launch accounted for, no setting chosen after seeing its outcome.
2. **Fails loudly.** Each check proposed can fail, and says what its failure means.
3. **Cost.** Launches, CPU time and repository growth stay proportionate to what each step can teach.
4. **Within authority.** Anything that needs an owner act (history rewrite, installing software, locked files) is stated as such, with the exact change.
5. **Simplicity for a collaborator** who runs commands from a written brief, with a Claude Code session that has no memory of the project.

## Hard constraints

- No rule chosen because it makes one agent win.
- Test-tier seeds are never used here.

## What would settle parts of it empirically instead

Whether the learned temperature reaches its target at `alpha_lr` 3e-4 is settled by the dev check itself. A proposal may make the check's design contingent on that, but it must say so.

## Material

`CLAUDE.md`; `experiments/train_agent.py`; `experiments/modal_backend.py`; `ocmdp/agents/oc_sac.py`; `ocmdp/agents/losses.py`; `ocmdp/agents/entropy_range.py`; `arena/2026-11-23-temperature-outcome.md`; `decisions/2026-11-23.md`; `decisions/2026-11-24.md`; `README.md`; `.gitignore`; the three `runs/train` reports named above.
