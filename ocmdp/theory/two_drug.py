"""Mean-field analysis of the linear selection model (ocmdp.core.selection).

Under a stationary policy p, and in the limit of many decisions per quarter, the
occupancy measure converges to p whatever lam and the delay are. Selection pressure then
converges to C^T p and resistance to its fixed point, so the long-run failure rate is a
closed-form function of p:

    F(p) = sum_a p_a * P(fail | a; r*(C^T p))

Two independent drugs with equal rates give F(e_A) = kappa / (kappa + delta) and
F(1/2, 1/2) = kappa / (kappa + 2 delta), so mixing strictly beats either pure policy.
That gap is the first claim. Cross-resistance rho shrinks it. At rho = 1 both drugs
select for both mechanisms equally, F is constant in p, and spreading mass over the two
drugs buys nothing. That is the second claim: action entropy is maximal at p = 1/2 while
the entropy that matters is zero for every p.
"""

from dataclasses import dataclass

import numpy as np

from ocmdp.core.coupling import CrossResistanceCoupling
from ocmdp.core.selection import failure_probability, stationary_resistance


def stationary_failure_rate(policy: np.ndarray, coupling: CrossResistanceCoupling, kappa, delta) -> np.ndarray:
    """Long-run per-decision failure rate of stationary policy p. Batches over leading axes of p."""
    p = np.asarray(policy, dtype=np.float64)
    r = stationary_resistance(coupling(p), kappa, delta)
    return (p * failure_probability(r, coupling.matrix)).sum(axis=-1)


@dataclass(frozen=True)
class TwoDrugSolution:
    rho: float
    p_opt: float  # probability of drug A under the best stationary policy
    f_opt: float
    f_deterministic: float  # the better of the two pure policies

    @property
    def gap(self) -> float:
        return self.f_deterministic - self.f_opt


def solve_two_drug(rho: float = 0.0, kappa: float = 0.10, delta: float = 0.05, resolution: int = 10_001) -> TwoDrugSolution:
    coupling = CrossResistanceCoupling(((1.0, rho), (rho, 1.0)))
    ps = np.linspace(0.0, 1.0, resolution)
    f = stationary_failure_rate(np.stack([ps, 1.0 - ps], axis=-1), coupling, kappa, delta)
    i = int(np.argmin(f))
    return TwoDrugSolution(rho=rho, p_opt=float(ps[i]), f_opt=float(f[i]), f_deterministic=float(min(f[0], f[-1])))
