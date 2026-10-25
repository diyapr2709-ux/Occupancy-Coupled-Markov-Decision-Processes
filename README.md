# Occupancy-Coupled Markov Decision Processes

Reinforcement learning when the agent's own policy history rewrites the transition kernel, applied to hospital antimicrobial stewardship. The transition kernel is `P(s' | s, a; mu_t)`, where `mu_t = (1 - lam) sum_k lam^k d_{t-k}` is an exponentially weighted occupancy measure of the agent's past actions and `Phi` maps it into kernel parameters.

Two claims organise the project:

1. **Deterministic stationary policies are strictly suboptimal.** The occupancy collapses onto the policy's support, and `Phi` drives the kernel to its worst case there.
2. **Max-entropy RL regularises in the wrong space.** The right objective is `H(psi#pi)`, the entropy of the policy pushed forward through a cross-resistance embedding, not `H(pi)`.

Both are already visible in the toy environment, with no learning involved (`experiments/validate.py`).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python experiments/validate.py --quick   # V&V harness, two seeds (~30 s)
.venv/bin/python experiments/validate.py           # full seed protocol
```

Extras: `graph` (PyTorch Geometric, for C2), `train` (tyro, W&B, TensorBoard, for the CleanRL fork), `baselines` (Stable-Baselines3).

## Layout and status

| Path | What | Owner | Status |
|---|---|---|---|
| `ocmdp/core/occupancy.py` | Occupancy measure `mu_t` with decay `lam` and delay | Diya | done |
| `ocmdp/core/coupling.py` | `Phi`: occupancy → selection pressure via cross-resistance | Nikhila | done |
| `ocmdp/core/selection.py` | Linear selection model (toy dynamics) | Nikhila | done |
| `ocmdp/envs/toy.py` | Reduced OC-MDP, Gymnasium API; `TwoDrug-v0`, `PairedDrugs-v0`; per-mechanism rates, `mu0`, opt-in Markov observation, privileged `state()` | Nikhila | done |
| `ocmdp/envs/stewardship.py` | Full env: fitness landscape + ward ABM + antibiogram | Nikhila | stub |
| `ocmdp/envs/{evolution,transmission,observation}.py` | Its three layers | Nikhila | stub |
| `ocmdp/components/belief_filter.py` | C1 population belief filter | Nikhila | stub |
| `ocmdp/components/action_encoder.py` | C2 graph encoder `psi` | Soumitra | stub |
| `ocmdp/components/policy_head.py` | C3 mixing policy head, `alpha(s)` | Diya | stub |
| `ocmdp/components/reserve_critic.py` | C4 efficacy-reserve critic | Diya | stub |
| `ocmdp/components/constraint_filter.py` | C5 per-patient hard feasibility projection | Nikhila | done |
| `ocmdp/objectives/pushforward_entropy.py` | `H(psi#pi)`: cluster form, and cluster-free similarity-sensitive form `H_Z` | Diya | done |
| `ocmdp/agents/losses.py` | Actor loss, `alpha(s)` loss, entropy target | Diya | done |
| `ocmdp/agents/lagrangian.py` | Dual variable `xi`, slow-clock projected ascent | Diya | done |
| `ocmdp/agents/oc_sac.py` | CleanRL `sac_atari.py` fork, training loop | Diya | stub (args only) |
| `ocmdp/theory/two_drug.py` | Mean-field two-drug result, suboptimality gap vs `rho` | Soumitra | done |
| `ocmdp/theory/mean_field.py` | Differentiable mean-field model of the toy env (exact replay of env trajectories) | Soumitra | done |
| `ocmdp/baselines/heuristics.py` | Fixed guideline, cycling, stationary / uniform mixing | Soumitra | done |
| `ocmdp/baselines/oracle.py` | Full-information oracle: replanning on the mean-field model; runs under misspecified `lam` | Soumitra | done |
| `ocmdp/baselines/{deep_rl,sw_ucb,offline}.py` | PPO/DQN/SAC, SW-UCB, CQL/IQL | Soumitra | stub |
| `ocmdp/eval/` | Rollout logging, Section 5 metrics, fixed seed protocol | Soumitra | done |
| `ocmdp/eval/vv.py`, `experiments/validate.py` | V&V harness: verification registry, comparator table with regret, `lam` misspecification | Soumitra | done |
| `ocmdp/transfer/safety_filter.py` | Safety filters vs adapting attack populations | Diya | stub |
| `docs/cleanrl_diff.md` | Documented diff against CleanRL | Diya | outline |
| `data/` | Fitness landscape data | Nikhila | empty |

## Environment contract

The full environment must keep this contract, so every comparator runs unmodified against both environments.

- One `step` is one empiric prescribing decision. A quarter closes every `decisions_per_quarter` steps. The kernel only moves when a quarter closes, so a single action cannot influence it.
- A quarter's empirical action distribution enters `mu` after `delay` quarters.
- An episode is `horizon_quarters` quarters (default 40, i.e. ten years) and ends by truncation.
- `info` carries `quarter`, `resistance`, `occupancy`, `fail_prob`, `action_mask` (feasible set for the *next* decision), and after a step also `action`, `failed`, `violation`.

## Verification and validation

`ocmdp/eval/vv.py` holds the Section 5 harness.

- **Verification** is a registry of checks. Each returns `pass`, `fail`, or `pending` (the component it needs doesn't exist yet), and pytest runs every one (`tests/test_verification.py`). Register a new check with the `@verification` decorator.
- **Validation** produces measurements, not verdicts: the comparator table, regret against the oracle, and degradation under `lam` misspecification.

`experiments/validate.py` runs both, prints them, and writes a JSON report to `runs/vv/`. It exits non-zero if any verification check fails.

Use `expected_failure_rate` to compare policies. It is the mean failure probability of the drugs actually given. It has the same expectation as the sampled `failure_rate` without the patient-outcome noise.

### Findings so far (toy environments, 5 seeds, no learning)

- **The full-information oracle is a deterministic rotation, not a stochastic mix.** It switches mechanism in blocks of about 7–8 quarters and reaches expected failure 0.347, against 0.380 for 4-quarter cycling and 0.389 for uniform mixing on TwoDrug. Claim 1 rules out deterministic *stationary* policies only, and nothing so far shows a stochastic policy beating a well-timed rotation. The comparison that decides the project should include the oracle's rotation, not only uniform mixing.
- **Claim 2 holds as stated in the env.** Within-class mixing has the same action entropy as cross-class mixing (0.856) but the fixed guideline's pushforward entropy and failure rate. Redistributing mass inside a cross-resistance class leaves the trajectory unchanged to 1e-16.
- **Misspecifying `lam` costs little, and underestimating memory costs more.** With a ×¼ effective horizon the oracle reaches 0.371, with ×4 it reaches 0.352, against 0.347 when `lam` is correct.

## Seed protocol

`ocmdp.eval.seeds.SEEDS = (0, 1, 2, 3, 4)`. Every reported number is a mean over these. The environment and the policy draw from independent streams derived from the same seed. Evaluation seeds are offset by 10 000.
