"""Actor and temperature losses for occupancy-coupled SAC over discrete actions.

CleanRL's discrete SAC (sac_atari.py) minimises

    (pi * (alpha * log pi - min_q)).sum(-1).mean()

with a scalar alpha. Here entropy is taken on psi#pi instead of pi, alpha is a function
of state, and the reserve critic enters with Lagrange multiplier xi, so the actor
ascends E_pi[Q] + alpha(s) H(psi#pi) - xi E_pi[Q_reserve].

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
    alpha:     (B,) state-dependent temperature; treated as a constant here
    xi:        dual variable; treated as a constant here
    entropy:   (B,) H(psi#pi), computed from `probs` so that it carries their gradient
    """
    # a (B, 1) input would silently broadcast against (B,) into a (B, B) loss
    _require_shape(probs.shape, q_values=q_values, reserve_q=reserve_q)
    _require_shape(probs.shape[:-1], entropy=entropy, alpha=alpha)
    expected_q = (probs * q_values).sum(-1)
    expected_reserve = (probs * reserve_q).sum(-1)
    return (-(expected_q + alpha.detach() * entropy) + xi * expected_reserve).mean()


def temperature_loss(log_alpha: torch.Tensor, entropy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Raises alpha(s) where H(psi#pi) is below its state-dependent target and lowers it where above."""
    _require_shape(log_alpha.shape, entropy=entropy, target=target)
    return (log_alpha * (entropy - target).detach()).mean()


def entropy_target(
    resistance: torch.Tensor, h_low: float, h_high: float, threshold: float = 0.5, sharpness: float = 10.0
) -> torch.Tensor:
    """Target for H(psi#pi) that rises from h_low to h_high as the most prevalent mechanism nears fixation.

    resistance: (B, M) estimated mechanism frequencies.
    """
    closeness = torch.sigmoid(sharpness * (resistance.max(dim=-1).values - threshold))
    return h_low + (h_high - h_low) * closeness
