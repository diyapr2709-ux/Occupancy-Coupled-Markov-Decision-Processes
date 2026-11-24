"""The entropy range an agent's own policy class can reach, in its own entropy space.

The actor scores each drug through its embedding (ocmdp.components.pairwise), so drugs
with identical embedding rows always receive equal probability: the policies it can
express are those uniform within each group of tied drugs. Over that set an entropy that
is concave in the policy, as the action entropy and the entropy of a linear pushforward
both are, takes its least value at a vertex, all mass spread uniformly over one group,
and its greatest by concave maximisation over the group masses. A temperature target is
only meaningful strictly inside this range: below the floor it can never be met and the
temperature only falls, which is what a fraction of log K does to drug-entropy SAC when
classes of drugs are tied.
"""

from collections.abc import Callable

import torch


def tied_groups(embeddings: torch.Tensor) -> list[list[int]]:
    """Indices of drugs with identical embedding rows, which the actor cannot tell apart."""
    _, group = torch.unique(embeddings, dim=0, return_inverse=True)
    return [torch.nonzero(group == g).flatten().tolist() for g in range(int(group.max()) + 1)]


def reachable_entropy_range(
    entropy: Callable[[torch.Tensor], torch.Tensor], embeddings: torch.Tensor, steps: int = 3000, lr: float = 0.05
) -> tuple[float, float]:
    """(floor, maximum) of a concave `entropy` of (..., A) float64 policies, over policies uniform within tied groups."""
    groups = tied_groups(embeddings)
    n_actions = embeddings.shape[0]
    spread = torch.zeros(len(groups), n_actions, dtype=torch.float64)
    for g, members in enumerate(groups):
        spread[g, members] = 1.0 / len(members)
    floor = float(entropy(spread).min())
    sizes = spread.gt(0).sum(-1).double()
    starts = torch.stack([torch.zeros(len(groups), dtype=torch.float64), sizes.log()])  # uniform over groups, over drugs
    logits = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    best = float(entropy(torch.softmax(starts, -1) @ spread).max())
    for _ in range(steps):
        value = entropy(torch.softmax(logits, -1) @ spread)
        best = max(best, float(value.detach().max()))
        opt.zero_grad()
        (-value.sum()).backward()
        opt.step()
    return floor, best
