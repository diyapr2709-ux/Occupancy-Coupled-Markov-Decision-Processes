# Occupancy-Coupled Markov Decision Processes

Environments, planners and policy components for MDPs whose transition kernel depends on the agent's own action history. Antimicrobial stewardship is the running example. The kernel is `P(s' | s, a; mu_t)`, where `mu_t = (1 - lam) sum_k lam^k d_{t-k}` is an exponentially weighted occupancy measure of past actions, and `Phi(mu) = mu @ C` turns it into selection pressure on resistance mechanisms through a cross-resistance matrix `C`.

Two properties of this problem class shape the code:

1. **Deterministic stationary policies are strictly suboptimal.** The occupancy collapses onto the policy's support, and `Phi` drives the kernel to its worst case there (`ocmdp/theory/two_drug.py`).
2. **Entropy over drugs is the wrong exploration measure.** Drugs with the same cross-resistance profile are one selection pressure, so exploration is measured by `H(psi#pi)`, the entropy of the policy pushed through a drug embedding (`ocmdp/objectives/pushforward_entropy.py`).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python experiments/validate.py --quick   # V&V harness, two seeds
.venv/bin/python experiments/validate.py           # full seed protocol (about 10 minutes)
.venv/bin/python experiments/train_agent.py --tier dev   # train and compare agents (about 15 minutes on 10 cores)
.venv/bin/python experiments/planner_restarts.py   # how far the oracle's plan moves with its random starts
.venv/bin/python experiments/sweep_phase_diagram.py --tier dev   # value of feedback over delay, memory and noise (planning only)
.venv/bin/python experiments/plot_phase_diagram.py runs/phase/<report>.json   # small-multiples SVG and CSV in figures/
```

Training and the phase diagram can run on Modal, one container per job, with `--backend modal` (needs `pip install -e ".[modal]"` and a Modal profile):

```bash
.venv/bin/python experiments/train_agent.py --tier dev --backend modal
```

## Layout

| Path | Contents |
|---|---|
| `ocmdp/core/occupancy.py` | Occupancy measure `mu_t` with decay `lam` and delay |
| `ocmdp/core/coupling.py` | `Phi`: occupancy → selection pressure via `C`; classes of interchangeable drugs |
| `ocmdp/core/selection.py` | Linear selection model behind the toy dynamics |
| `ocmdp/envs/toy.py` | Gymnasium environment (`ocmdp/TwoDrug-v0`, `ocmdp/PairedDrugs-v0`) |
| `ocmdp/envs/wrappers.py` | `MechanismObservation`: observe prescribing as selection pressure `x @ C` |
| `ocmdp/components/pairwise.py` | Shared pairwise (state, embedded drug) scorer |
| `ocmdp/components/policy_head.py` | Mixing policy head over the embedded drugs; tied embeddings get equal probability |
| `ocmdp/components/reserve_critic.py` | Efficacy-reserve critic, its per-quarter cost and TD target |
| `ocmdp/components/constraint_filter.py` | Per-patient hard feasibility projection |
| `ocmdp/objectives/pushforward_entropy.py` | `H(psi#pi)`: cluster form and cluster-free similarity-sensitive form `H_Z` |
| `ocmdp/agents/losses.py` | Actor loss, scalar temperature loss reflected at its clamp, entropy target inside the reachable range |
| `ocmdp/agents/entropy_range.py` | Least and greatest entropy an agent's policy class can reach, given the drugs its networks cannot tell apart |
| `ocmdp/agents/lagrangian.py` | Dual variable `xi`, projected ascent on a slow clock |
| `ocmdp/agents/oc_sac.py` | Occupancy-coupled SAC: quarterly decisions, pushforward or action entropy, one fixed or learned temperature, reserve critic, dual variable, checkpoints |
| `ocmdp/agents/distill.py` | Behaviour cloning and DAgger from the oracle into the policy head |
| `ocmdp/theory/two_drug.py` | Closed-form stationary analysis of the two-drug model |
| `ocmdp/theory/mean_field.py` | Differentiable mean-field model that replays the environment exactly |
| `ocmdp/baselines/heuristics.py` | Fixed guideline, cycling, stationary and uniform mixing, fixed schedules, myopic greedy |
| `ocmdp/baselines/oracle.py` | Full-information oracle and the rotation search it starts from |
| `ocmdp/eval/` | Rollout logging, metrics, seed protocol, V&V harness |
| `experiments/validate.py` | Runs the harness and writes a JSON report to `runs/vv/` |
| `experiments/train_agent.py` | Trains OC-SAC and standard SAC across seeds, both agents of a seed in one job, and compares them with the baselines; writes `runs/train/` and a launch record per launch, with every trained agent under `runs/train/checkpoints/<tier>/` (test-tier checkpoints tracked in git, others archived with their hashes in the report) |
| `experiments/verify_checkpoints.py` | Checks a report's checkpoint files against the sizes and SHA-256 hashes it records |
| `experiments/planner_restarts.py` | Spread of the oracle's planned failure rate over planner seeds; writes `runs/planner/` |
| `experiments/sweep_phase_diagram.py` | Heuristics and planners over a (delay, memory, noise) grid, paired per cell; writes `runs/phase/` |
| `experiments/plot_phase_diagram.py` | Small-multiples figure of one phase-diagram contrast, with an optional prediction overlay; writes `figures/` |
| `experiments/modal_backend.py` | Runs the jobs of the experiment scripts on Modal, with package versions and SIMD kernels pinned |
| `provenance/` | Patches that rebuild the uncommitted code behind earlier reports, each checked against its report's recorded hash |
| `experiments/predict_entropy_gap.py` | Failure-rate gap between class-entropy and drug-entropy soft-optimal schedules on the mean-field model; writes `runs/predict/` |
| `experiments/score_phase_predictions.py` | Scores predicted sign changes of a phase-diagram contrast against measured ones, with a joint episode bootstrap |
| `experiments/null_phase_predictions.py` | Theory-free predictions for held-out settings, carried over from the dev map |
| `seeds.yaml` | The dev, val and test seed tiers |
| `budget.yaml` | Daily limit on launched experiment commands |

## Environment contract

- **Steps and quarters.** One `step` is one prescribing decision. A quarter closes every `decisions_per_quarter` steps. The kernel only moves when a quarter closes, so a single action cannot influence it.
- **Delay.** A quarter's empirical action distribution enters `mu` after `delay` quarters.
- **Episode length.** An episode is `horizon_quarters` quarters (default 40) and ends by truncation.
- **`info` contents.** `quarter`, `resistance`, `occupancy`, `fail_prob`, and `action_mask` (the feasible set for the *next* decision). After a step it also carries `action`, `failed` and `violation`.
- **Optional behaviour:**
  - `resistance_noise` adds mean-zero drift: the shock is reflected rather than clipped at 0 and 1.
  - `r0_spread` and `mu0_spread` jitter the starting state per episode.
  - `markov_observation` exposes the rest of the Markov state.

## Verification and validation

`ocmdp/eval/vv.py` holds the harness.

- **Verification** is a registry of checks that each return `pass` or `fail`. pytest runs every one (`tests/test_verification.py`). A check that raises or returns anything else counts as a failure.
- **Validation** produces measurements:
  - the comparator table with paired regret against the oracle;
  - `lam` misspecification;
  - process noise, with paired 95% intervals;
  - distillation of the oracle into the policy head.

Beside the harness:

- `tests/test_system.py` runs the stack end to end:
  - fuzzed configurations, with invariants checked at every step and exact mean-field replay;
  - rejection of invalid input;
  - sync and async vector environments;
  - determinism, deepcopy and pickling;
  - a 20-drug, 5-mechanism run;
  - the CLI.
- `tests/test_pentest_regressions.py` pins every confirmed finding from adversarial testing.

Compare policies with `expected_failure_rate`, the mean failure probability of the drugs actually given. It has the same expectation as the sampled `failure_rate`, without the patient-outcome noise.

### Results on the toy environments (no learned reward)

- **Without noise, the best schedule is a deterministic rotation.**
  - The rotation search finds 13-quarter blocks with a 7-quarter first block.
  - It reaches an expected failure rate of 0.3467 on both toy environments, the same as the oracle (regret 0.1 failures per 1,000-decision episode).
  - For comparison: 4-quarter cycling 0.380, uniform mixing 0.389, fixed guideline 0.532.
  - Property 1 rules out deterministic *stationary* policies only.
- **With noise, feedback is what matters.** The oracle re-plans every 4 quarters, starting each plan from the best rotation for the state it has reached. Over committing to the best rotation in advance, its paired gains on TwoDrug with 20 seeds are:

  | Noise σ | Paired gain (95% interval) |
  |---|---|
  | 0.1 | +0.008 [0.002, 0.014] |
  | 0.2 | +0.031 [0.018, 0.044] |
  | 0.4 | +0.054 [0.014, 0.094] |

  Source: `runs/vv/vv_20261025_201416_713126_29699.json` (`noise_robustness.replanning_gain`, 20 seeds).

  Absolute rates at σ = 0.4 vary widely between seeds, so read the paired gains.
- **Splitting mass inside a cross-resistance class changes nothing.**
  - Within-class mixing has the same action entropy as cross-class mixing (0.856).
  - But it has the fixed guideline's pushforward entropy and failure rate.
  - In the environment, the trajectories agree to 1e-16.
- **Misspecifying `lam` costs little.** With a quarter of the true effective memory, the oracle reaches 0.355; with four times, 0.349; correct, 0.347.
- **The policy head can represent the oracle.** Trained on held-out starting states, it captures 89–98% of the oracle's gain over uniform mixing on PairedDrugs. The rates are 0.344–0.348, against the oracle's 0.343 and uniform's 0.387.
  - The 98% end comes from observing prescribing as selection pressure.
  - These runs use the true `C` for the embeddings and the observation map, and the observation includes exact resistance.

### Trained agents

`experiments/train_agent.py` trains OC-SAC (entropy over mechanism classes) and standard SAC (entropy over drugs) on paired seeds. Each agent has one scalar temperature, fixed or learned toward a target inside the entropy range its own policy class can reach.

No trained-agent result is reported yet. All ten reports in `runs/train/` (the latest is `train_20261123_194806_29133.json`) come from earlier versions of the agent, written before its state-dependent temperature was replaced by the current scalar one. They are diagnostics of those versions, and their seeds are not comparable with later runs. Results will appear here after a preregistered comparison has run, with intervals, the interquartile mean and the number of runs that learned.

An earlier version of this section compared three entropy targets on validation runs that were never logged. Those numbers are withdrawn.

## Seed protocol

`ocmdp.eval.seeds.SEEDS = (0, 1, 2, 3, 4)` is the harness protocol. The noise study uses 20 seeds, because its paired differences spread widely at σ = 0.4 (`runs/vv/vv_20261025_201416_713126_29699.json`).
- **Tiers.** `seeds.yaml` splits training seeds into three disjoint ranges: dev for exploration, val for choosing hyperparameters, test for the single confirmatory run of a preregistration. `experiments/train_agent.py --tier` rejects seeds outside the tier.
- **Evaluation episodes follow the tier.** A run is evaluated on episodes `10 000 + first seed of the tier + e`, so val and test episodes are never seen by a dev run. A test-tier run also needs a committed preregistration in `preregistrations/` that names its platform, the owner's approval file, a clean tree, and no earlier test launch of the same preregistration; the scripts refuse it otherwise. They do not stop a different preregistration from reusing test seeds.
- **Run reports.** Training and phase reports written by the current scripts record the tier, seeds, evaluation episodes, git state, platform label and instance, and training reports also record each agent's reachable entropy range and temperature. Older training reports lack some of these fields, and none records a platform label. The validation reports under `runs/vv/` record their seeds, but not the tier, git state or evaluation episodes.
- **Independent streams.** The environment and the policy draw from independent random streams, so every policy faces the same patients and shocks for a given seed.
