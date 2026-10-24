# Occupancy-Coupled Markov Decision Processes

Reinforcement learning when the agent's own policy history rewrites the transition kernel, applied to hospital antimicrobial stewardship. The transition kernel is `P(s' | s, a; mu_t)`, where `mu_t = (1 - lam) sum_k lam^k d_{t-k}` is an exponentially weighted occupancy measure of the agent's past actions and `Phi` maps it into kernel parameters.

Two claims organise the project:

1. **Deterministic stationary policies are strictly suboptimal.** The occupancy collapses onto the policy's support, and `Phi` drives the kernel to its worst case there.
2. **Max-entropy RL regularises in the wrong space.** The right objective is `H(psi#pi)`, the entropy of the policy pushed forward through a cross-resistance embedding, not `H(pi)`.

Both are already visible in the toy environment, with no learning involved (`experiments/run_heuristics.py`).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python experiments/run_heuristics.py
```

Extras: `graph` (PyTorch Geometric, for C2), `train` (tyro, W&B, TensorBoard, for the CleanRL fork), `baselines` (Stable-Baselines3).

## Layout and status

| Path | What | Owner | Status |
|---|---|---|---|
| `ocmdp/core/occupancy.py` | Occupancy measure `mu_t` with decay `lam` and delay | Diya | done |
| `ocmdp/core/coupling.py` | `Phi`: occupancy → selection pressure via cross-resistance | Nikhila | done |
| `ocmdp/core/selection.py` | Linear selection model (toy dynamics) | Nikhila | done |
| `ocmdp/envs/toy.py` | Reduced OC-MDP, Gymnasium API; `TwoDrug-v0`, `PairedDrugs-v0` | Nikhila | done |
| `ocmdp/envs/stewardship.py` | Full env: fitness landscape + ward ABM + antibiogram | Nikhila | stub |
| `ocmdp/envs/{evolution,transmission,observation}.py` | Its three layers | Nikhila | stub |
| `ocmdp/components/belief_filter.py` | C1 population belief filter | Nikhila | stub |
| `ocmdp/components/action_encoder.py` | C2 graph encoder `psi` | Soumitra | stub |
| `ocmdp/components/policy_head.py` | C3 mixing policy head, `alpha(s)` | Diya | stub |
| `ocmdp/components/reserve_critic.py` | C4 efficacy-reserve critic | Diya | stub |
| `ocmdp/components/constraint_filter.py` | C5 per-patient hard feasibility projection | Nikhila | done |
| `ocmdp/objectives/pushforward_entropy.py` | `H(psi#pi)` with soft mechanism assignment | Diya | done |
| `ocmdp/agents/losses.py` | Actor loss, `alpha(s)` loss, entropy target | Diya | done |
| `ocmdp/agents/lagrangian.py` | Dual variable `xi`, slow-clock projected ascent | Diya | done |
| `ocmdp/agents/oc_sac.py` | CleanRL `sac_atari.py` fork, training loop | Diya | stub (args only) |
| `ocmdp/theory/two_drug.py` | Mean-field two-drug result, suboptimality gap vs `rho` | Soumitra | done |
| `ocmdp/baselines/heuristics.py` | Fixed guideline, cycling, stationary / uniform mixing | Soumitra | done |
| `ocmdp/baselines/{deep_rl,sw_ucb,offline,oracle}.py` | PPO/DQN/SAC, SW-UCB, CQL/IQL, oracle | Soumitra | stub |
| `ocmdp/eval/` | Rollout logging, Section 5 metrics, fixed seed protocol | Soumitra | done |
| `ocmdp/transfer/safety_filter.py` | Safety filters vs adapting attack populations | Diya | stub |
| `docs/cleanrl_diff.md` | Documented diff against CleanRL | Diya | outline |
| `data/` | Fitness landscape data | Nikhila | empty |

## Environment contract

The full environment must keep this contract, so every comparator runs unmodified against both environments.

- One `step` is one empiric prescribing decision. A quarter closes every `decisions_per_quarter` steps. The kernel only moves when a quarter closes, so a single action cannot influence it.
- A quarter's empirical action distribution enters `mu` after `delay` quarters.
- An episode is `horizon_quarters` quarters (default 40, i.e. ten years) and ends by truncation.
- `info` carries `quarter`, `resistance`, `occupancy`, `fail_prob`, `action_mask` (feasible set for the *next* decision), and after a step also `action`, `failed`, `violation`.

## Seed protocol

`ocmdp.eval.seeds.SEEDS = (0, 1, 2, 3, 4)`. Every reported number is a mean over these. The environment and the policy draw from independent streams derived from the same seed. Evaluation seeds are offset by 10 000.
