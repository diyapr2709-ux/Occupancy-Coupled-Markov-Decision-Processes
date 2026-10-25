"""Run one episode and keep what the metrics need, per decision and per quarter."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from ocmdp.eval.seeds import policy_rng


@dataclass
class EpisodeLog:
    actions: np.ndarray  # (T,)
    decision_quarter: np.ndarray  # (T,) quarter in which each decision was made
    failures: np.ndarray  # (T,) bool
    violations: np.ndarray  # (T,) bool
    resistance: np.ndarray  # (Q + 1, M), row q is the state at the start of quarter q
    occupancy: np.ndarray  # (Q + 1, A)
    fail_prob: np.ndarray  # (Q + 1, A)


def rollout(env: gym.Env, policy, seed: int) -> EpisodeLog:
    obs, info = env.reset(seed=seed)
    policy.reset()
    rng = policy_rng(seed)
    actions, decision_quarter, failures, violations = [], [], [], []
    quarters = [info]
    while True:
        decision_quarter.append(info["quarter"])
        obs, _, terminated, truncated, info = env.step(policy.act(obs, info, rng))
        actions.append(info["action"])
        failures.append(info["failed"])
        violations.append(info["violation"])
        if info["quarter"] == len(quarters):
            quarters.append(info)
        if terminated or truncated:
            break
    return EpisodeLog(
        actions=np.array(actions),
        decision_quarter=np.array(decision_quarter),
        failures=np.array(failures),
        violations=np.array(violations),
        resistance=np.stack([q["resistance"] for q in quarters]),
        occupancy=np.stack([q["occupancy"] for q in quarters]),
        fail_prob=np.stack([q["fail_prob"] for q in quarters]),
    )
