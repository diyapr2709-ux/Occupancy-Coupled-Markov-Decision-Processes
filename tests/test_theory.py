import numpy as np
import pytest

from ocmdp.theory.two_drug import solve_two_drug


def test_closed_form_for_independent_drugs():
    kappa, delta = 0.10, 0.05
    sol = solve_two_drug(rho=0.0, kappa=kappa, delta=delta)
    assert sol.p_opt == pytest.approx(0.5)
    assert sol.f_opt == pytest.approx(kappa / (kappa + 2 * delta))
    assert sol.f_deterministic == pytest.approx(kappa / (kappa + delta))


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 0.9])
def test_deterministic_stationary_policies_are_strictly_suboptimal(rho):
    sol = solve_two_drug(rho=rho)
    assert sol.gap > 1e-4
    assert 0.0 < sol.p_opt < 1.0


def test_gap_closes_under_full_cross_resistance():
    assert solve_two_drug(rho=1.0).gap == pytest.approx(0.0, abs=1e-12)


def test_gap_shrinks_as_cross_resistance_grows():
    gaps = [solve_two_drug(rho=rho).gap for rho in np.linspace(0.0, 1.0, 11)]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))
