"""Actor and temperature losses for occupancy-coupled SAC over discrete actions.

CleanRL's discrete SAC (sac_atari.py) minimises

    (pi * (alpha * log pi - min_q)).sum(-1).mean()

with a scalar alpha. Here entropy may be taken on psi#pi instead of pi, and the reserve
critic enters with Lagrange multiplier xi, so the actor ascends
E_pi[Q] + alpha H(psi#pi) - xi E_pi[Q_reserve].

The entropy is computed by the caller from the same probs, with either estimator in
ocmdp.objectives.pushforward_entropy (cluster or similarity form), so the actor loss
does not commit to one.
"""

import torch


def _require_shape(shape: torch.Size, **tensors: torch.Tensor) -> None:
    for name, t in tensors.items():
        if t.shape != shape:
            raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {tuple(shape)}")


def actor_loss(
    probs: torch.Tensor,
    q_values: torch.Tensor,
    reserve_q: torch.Tensor,
    alpha: torch.Tensor,
    xi: float,
    entropy: torch.Tensor,
) -> torch.Tensor:
    """
    probs:     (B, A) policy after the constraint filter
    q_values:  (B, A) min of the twin reward critics
    reserve_q: (B, A) reserve-critic estimate of discounted option-value loss (higher is worse)
    alpha:     (B,) temperature per sample (the agent's one scalar, expanded); treated as a constant here
    xi:        dual variable; treated as a constant here
    entropy:   (B,) H(psi#pi), computed from `probs` so that it carries their gradient
    """
    # a (B, 1) input would silently broadcast against (B,) into a (B, B) loss
    _require_shape(probs.shape, q_values=q_values, reserve_q=reserve_q)
    _require_shape(probs.shape[:-1], entropy=entropy, alpha=alpha)
    expected_q = (probs * q_values).sum(-1)
    expected_reserve = (probs * reserve_q).sum(-1)
    return (-(expected_q + alpha.detach() * entropy) + xi * expected_reserve).mean()


def temperature_loss(log_alpha: torch.Tensor, entropy: torch.Tensor, target: float, log_alpha_range: tuple[float, float]) -> torch.Tensor:
    """Loss on the raw scalar log alpha: descending it raises alpha when the batch entropy is below target.

    Reflecting at the clamp range: outside it, only a gradient that points back inside is
    kept. A clamp applied before the loss would stop the gradient altogether, and a
    temperature that crossed the lower bound would then stay there.
    """
    if log_alpha.ndim != 0 or entropy.ndim != 0:
        raise ValueError(f"log alpha and the batch entropy must be scalars, got shapes {tuple(log_alpha.shape)} and {tuple(entropy.shape)}")
    gap = (entropy - target).detach()
    low, high = log_alpha_range
    inside = (log_alpha >= low) & (log_alpha <= high)
    inward = ((log_alpha < low) & (gap < 0)) | ((log_alpha > high) & (gap > 0))
    return log_alpha * gap * (inside | inward).to(log_alpha.dtype)


def entropy_target(floor: float, maximum: float, fraction: float) -> float:
    """A target strictly inside the range an agent's policy class can reach: floor + fraction (maximum - floor)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"the target fraction must lie strictly between 0 and 1, got {fraction}")
    if not maximum > floor:
        raise ValueError(f"no entropy range to regulate: floor {floor}, maximum {maximum}")
    return floor + fraction * (maximum - floor)
