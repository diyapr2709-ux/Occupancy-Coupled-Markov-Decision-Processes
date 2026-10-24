"""Actor and temperature losses for occupancy-coupled SAC over discrete actions.

CleanRL's discrete SAC (sac_atari.py) minimises

    (pi * (alpha * log pi - min_q)).sum(-1).mean()

with a scalar alpha. Here entropy is taken on psi#pi instead of pi, alpha is a function
of state, and the reserve critic C4 enters with Lagrange multiplier xi, so the actor
ascends E_pi[Q] + alpha(s) H(psi#pi) - xi E_pi[Q_reserve].
"""

import torch

from ocmdp.objectives.pushforward_entropy import pushforward_entropy


def actor_loss(
    probs: torch.Tensor,
    q_values: torch.Tensor,
    reserve_q: torch.Tensor,
    alpha: torch.Tensor,
    xi: float,
    assignment: torch.Tensor,
) -> torch.Tensor:
    """
    probs:      (B, A) policy after the C5 constraint filter
    q_values:   (B, A) min of the twin reward critics
    reserve_q:  (B, A) C4 estimate of discounted option-value loss (higher is worse)
    alpha:      (B,) state-dependent temperature; treated as a constant here
    xi:         dual variable; treated as a constant here
    assignment: (A, K) soft assignment of embedded actions to mechanism clusters
    """
    expected_q = (probs * q_values).sum(-1)
    expected_reserve = (probs * reserve_q).sum(-1)
    entropy = pushforward_entropy(probs, assignment)
    return (-(expected_q + alpha.detach() * entropy) + xi * expected_reserve).mean()


def temperature_loss(log_alpha: torch.Tensor, entropy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Raises alpha(s) where H(psi#pi) is below its state-dependent target and lowers it where above."""
    return (log_alpha * (entropy - target).detach()).mean()


def entropy_target(
    resistance: torch.Tensor, h_low: float, h_high: float, threshold: float = 0.5, sharpness: float = 10.0
) -> torch.Tensor:
    """Target for H(psi#pi) that rises from h_low to h_high as the most prevalent mechanism nears fixation.

    resistance: (B, M) estimated mechanism frequencies (C1 belief mean).
    """
    closeness = torch.sigmoid(sharpness * (resistance.max(dim=-1).values - threshold))
    return h_low + (h_high - h_low) * closeness
