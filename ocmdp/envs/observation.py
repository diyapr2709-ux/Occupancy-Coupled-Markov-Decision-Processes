"""Antibiogram emission model: what the agent sees of the latent resistance state.

Owner: Nikhila. Not implemented yet.

An antibiogram reports susceptibility among cultured isolates, aggregated over a
quarter and released with a lag. Patients are cultured more often after a treatment
failure, which inflates apparent resistance. That bias is a parameter of this model
rather than something a recurrent encoder absorbs silently, and C1 inverts it.
"""

import numpy as np


class AntibiogramEmission:
    def __init__(self, culture_rate_success: float, culture_rate_failure: float, lag_quarters: int):
        raise NotImplementedError

    def sample(self, resistance: np.ndarray, failures: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one quarter's antibiogram: (drugs,) fraction of cultured isolates resistant."""
        raise NotImplementedError

    def log_likelihood(self, antibiogram: np.ndarray, resistance: np.ndarray) -> float:
        raise NotImplementedError
