"""Validation and verification metrics (Section 5 of the proposal)."""

import numpy as np

from ocmdp.eval.rollout import EpisodeLog


def failure_rate(failures: np.ndarray) -> float:
    """Primary metric: cumulative treatment failure rate over the horizon."""
    return float(np.mean(failures))


def expected_failure_rate(log: EpisodeLog) -> float:
    """Mean failure probability of the drugs actually given. Same expectation as failure_rate
    with the patient-outcome noise removed, so policy comparisons need far fewer seeds."""
    return float(np.mean(log.fail_prob[log.decision_quarter, log.actions]))


def time_to_fixation(resistance: np.ndarray, threshold: float) -> np.ndarray:
    """(Q, M) frequencies -> (M,) first quarter each mechanism reaches threshold; inf if never."""
    if np.isnan(resistance).any():
        raise ValueError("resistance contains NaN")
    reached = resistance >= threshold
    return np.where(reached.any(axis=0), reached.argmax(axis=0), np.inf)


def efficacy_reserve(fail_prob: np.ndarray, threshold: float) -> int:
    """Drugs still effective (failure probability below threshold)."""
    return int(np.sum(fail_prob < threshold))


def regret(failures: np.ndarray, oracle_failures: np.ndarray) -> float:
    return float(np.sum(failures) - np.sum(oracle_failures))


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=-1)


def occupancy_entropy(occupancy: np.ndarray) -> float:
    """Mean over quarters of H(mu_t): action-space entropy of what the kernel sees."""
    return float(np.mean(entropy(occupancy)))


def pushforward_occupancy_entropy(occupancy: np.ndarray, assignment: np.ndarray) -> float:
    """Mean over quarters of H(psi#mu_t). The gap to occupancy_entropy is the mechanism being claimed."""
    return float(np.mean(entropy(occupancy @ assignment)))


def effective_dimensionality(embeddings: np.ndarray) -> float:
    """Participation ratio of the embedding covariance spectrum; near 1 means psi has collapsed."""
    x = np.asarray(embeddings, dtype=np.float64)
    eig = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    if eig.sum() == 0:
        return 0.0
    return float(eig.sum() ** 2 / np.sum(eig**2))


def summarize(log: EpisodeLog, assignment: np.ndarray, fixation_threshold: float = 0.5, efficacy_threshold: float = 0.5) -> dict:
    fixation = time_to_fixation(log.resistance, fixation_threshold)
    return {
        "failure_rate": failure_rate(log.failures),
        "expected_failure_rate": expected_failure_rate(log),
        "first_fixation_q": float(fixation.min()),
        "efficacy_reserve": efficacy_reserve(log.fail_prob[-1], efficacy_threshold),
        "H_action": occupancy_entropy(log.occupancy),
        "H_pushforward": pushforward_occupancy_entropy(log.occupancy, assignment),
        "violation_rate": float(np.mean(log.violations)),
    }
