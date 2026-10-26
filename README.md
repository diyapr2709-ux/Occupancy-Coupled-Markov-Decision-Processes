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
.venv/bin/python experiments/train_agent.py        # train and compare agents (about 15 minutes on 10 cores)
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
| `ocmdp/components/policy_head.py` | Mixing policy head with state-dependent temperature `alpha(s)` |
| `ocmdp/components/reserve_critic.py` | Efficacy-reserve critic, its per-quarter cost and TD target |
| `ocmdp/components/constraint_filter.py` | Per-patient hard feasibility projection |
| `ocmdp/objectives/pushforward_entropy.py` | `H(psi#pi)`: cluster form and cluster-free similarity-sensitive form `H_Z` |
| `ocmdp/agents/losses.py` | Actor loss, temperature loss, state-dependent entropy target |
| `ocmdp/agents/lagrangian.py` | Dual variable `xi`, projected ascent on a slow clock |
| `ocmdp/agents/oc_sac.py` | Occupancy-coupled SAC: quarterly decisions, pushforward or action entropy, `alpha(s)`, reserve critic, dual variable |
| `ocmdp/agents/distill.py` | Behaviour cloning and DAgger from the oracle into the policy head |
| `ocmdp/theory/two_drug.py` | Closed-form stationary analysis of the two-drug model |
| `ocmdp/theory/mean_field.py` | Differentiable mean-field model that replays the environment exactly |
| `ocmdp/baselines/heuristics.py` | Fixed guideline, cycling, stationary and uniform mixing, fixed schedules, myopic greedy |
| `ocmdp/baselines/oracle.py` | Full-information oracle and the rotation search it starts from |
| `ocmdp/eval/` | Rollout logging, metrics, seed protocol, V&V harness |
| `experiments/validate.py` | Runs the harness and writes a JSON report to `runs/vv/` |
| `experiments/train_agent.py` | Trains OC-SAC and standard SAC across seeds and compares them with the baselines; writes `runs/train/` |

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

  Absolute rates at σ = 0.4 vary widely between seeds, so read the paired gains.
- **Splitting mass inside a cross-resistance class changes nothing.**
  - Within-class mixing has the same action entropy as cross-class mixing (0.856).
  - But it has the fixed guideline's pushforward entropy and failure rate.
  - In the environment, the trajectories agree to 1e-16.
- **Misspecifying `lam` costs little.** With a quarter of the true effective memory, the oracle reaches 0.355; with four times, 0.349; correct, 0.347.
- **The policy head can represent the oracle.** Trained on held-out starting states, it captures 89–98% of the oracle's gain over uniform mixing on PairedDrugs. The rates are 0.344–0.348, against the oracle's 0.343 and uniform's 0.387.
  - The 98% end comes from observing prescribing as selection pressure.
  - These runs use the true `C` for the embeddings and the observation map, and the observation includes exact resistance.

### Trained agents (preliminary)

`experiments/train_agent.py` trains OC-SAC (pushforward entropy) and standard SAC (action entropy) on PairedDrugs. Settings:
- 12 000 quarterly decisions per run;
- 5 training seeds per variant;
- each trained agent evaluated on 10 held-out episodes with jittered starting states;
- the same episodes for every policy, so comparisons are paired.

The entropy target was chosen on separate validation episodes, from three candidates: rising (0.05→0.5 of the maximum entropy), lower (0.02→0.1) and flat (0.05). The rule was the best mean OC-SAC validation score across both noise levels, and the flat target won.

Expected failure rate (mean ± 95% interval over training seeds):

| | σ = 0 | σ = 0.2 |
|---|---|---|
| OC-SAC | 0.351 ± 0.002 (5/5 learn) | 0.341 ± 0.003 (5/5) |
| SAC | 0.351 ± 0.003 (5/5) | 0.344 ± 0.013 (5/5) |
| myopic greedy | 0.372 | 0.341 |
| tuned rotation | 0.347 | 0.363 |
| oracle | 0.345 | 0.357 |
| uniform | 0.389 | 0.421 |

- **Without noise,** both agents capture 87% of the oracle's gain over uniform. They beat greedy, but trail the tuned rotation and the oracle by 0.004–0.006.
- **With noise,** both match greedy and beat the oracle and the rotation. The oracle plans on the noise-free model, so it is not an upper bound here.
- **Entropy space makes no measurable difference once the target is flat.** The paired OC-SAC − SAC difference is 0.000 ± 0.003 at σ = 0 and −0.002 ± 0.011 at σ = 0.2.
- **The rising target separates the variants, in opposite directions.**
  - At σ = 0, OC-SAC learned on 8/8 runs (validation and evaluation combined) against 6/8 for SAC.
  - At σ = 0.2, OC-SAC learned on 2/8 runs against 8/8 for SAC.
  - The rising target forces OC-SAC to mix mechanisms near fixation. The same validation-run policies score 0.318 when sampling and 0.267 when taking their most probable drug. Standard SAC meets the same target by splitting mass between interchangeable drugs, so it never has to mix mechanisms.

## Seed protocol

`ocmdp.eval.seeds.SEEDS = (0, 1, 2, 3, 4)`.
- **Training and evaluation seeds.** Validation episodes run on evaluation seeds offset by 10 000. The noise study uses 20 seeds, because its paired differences have a standard deviation of about 0.07 at σ = 0.4.
- **Independent streams.** The environment and the policy draw from independent random streams, so every policy faces the same patients and shocks for a given seed.
