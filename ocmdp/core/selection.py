"""Linear selection model: the reduced dynamics behind the toy environment and the two-drug result.

Each resistance mechanism j has carriage frequency r_j. Once per quarter,

    r_j <- r_j + kappa_j * pi_j * (1 - r_j) - delta_j * r_j

where pi = Phi(mu) is selection pressure, kappa_j the acquisition rate under full
pressure, and delta_j the reversion rate set by the fitness cost of carriage. The fixed
point is r_j* = kappa_j pi_j / (kappa_j pi_j + delta_j). Drug a fails on a patient when
any mechanism it is susceptible to is present:

    P(fail | a) = 1 - prod_j (1 - C[a, j] * r_j)

The model is kept linear so that its stationary behaviour has a closed form.
"""

import numpy as np


def check_rates(kappa, delta) -> None:
    """kappa + delta <= 1 keeps every frequency in [0, 1] under resistance_step."""
    kappa, delta = np.asarray(kappa, dtype=np.float64), np.asarray(delta, dtype=np.float64)
    if not (np.isfinite(kappa).all() and np.isfinite(delta).all()):
        raise ValueError("kappa and delta must be finite")
    if np.any(kappa < 0) or np.any(delta <= 0):
        raise ValueError("need kappa >= 0 and delta > 0")
    if np.any(kappa + delta > 1):
        raise ValueError("need kappa + delta <= 1 for frequencies to stay in [0, 1]")


def resistance_step(r: np.ndarray, pressure: np.ndarray, kappa, delta) -> np.ndarray:
    return r + kappa * pressure * (1.0 - r) - delta * r


def stationary_resistance(pressure: np.ndarray, kappa, delta) -> np.ndarray:
    return kappa * pressure / (kappa * pressure + delta)


def failure_probability(r, cross_resistance):
    """Per-drug failure probability given mechanism frequencies r of shape (..., M). Returns (..., A).

    Written against the shared numpy/torch array API, so the differentiable mean-field
    model (ocmdp.theory.mean_field) runs this exact function. Pass both arguments as
    the same kind of array.
    """
    return 1.0 - (1.0 - cross_resistance * r[..., None, :]).prod(-1)
