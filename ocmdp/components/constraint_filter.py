"""C5: hard projection onto each patient's feasible set.

Infeasible actions get probability zero before sampling, so the violation rate is zero
by construction rather than a penalty the policy could trade away. The feasible set
comes from the environment (info["action_mask"]); an empty one means no empiric choice
is safe and the case goes to a clinician, so it raises instead of guessing.
"""

import numpy as np
import torch


class NoFeasibleAction(ValueError):
    pass


def project_probs(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero infeasible actions and renormalise; uniform over the feasible set if the policy put no mass there."""
    probs, mask = np.asarray(probs, dtype=np.float64), np.asarray(mask, dtype=bool)
    if probs.shape[-1:] != mask.shape[-1:]:
        raise ValueError(f"probs over {probs.shape[-1:]} drugs but mask over {mask.shape[-1:]}")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probs must be finite and non-negative")
    if not mask.any(axis=-1).all():
        raise NoFeasibleAction("a patient has no feasible action")
    masked = np.where(mask, probs, 0.0)
    total = masked.sum(axis=-1, keepdims=True)
    uniform = mask / mask.sum(axis=-1, keepdims=True)
    return np.where(total > 0, masked / np.where(total > 0, total, 1.0), uniform)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Training-time form: infeasible logits to -inf, so softmax puts exactly zero mass there."""
    if not mask.any(dim=-1).all():
        raise NoFeasibleAction("a patient has no feasible action")
    return logits.masked_fill(~mask, float("-inf"))


def sample_feasible(probs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    p = project_probs(probs, mask)
    return int(rng.choice(len(p), p=p))
