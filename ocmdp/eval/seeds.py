"""Seed protocol, fixed in advance: every reported number is a mean over SEEDS.

The environment and the policy draw from independent streams derived from the same
seed, so changing a policy never changes the patients the environment generates.
"""

import random

import numpy as np

SEEDS = (0, 1, 2, 3, 4)
EVAL_SEED_OFFSET = 10_000  # evaluation episodes never reuse a training seed


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def policy_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 1])


def eval_seed(seed: int) -> int:
    return seed + EVAL_SEED_OFFSET
