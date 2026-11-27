"""Reduced two-class model of the toy environment (class sizes [2, 2], rho = 0), noise-free.

State: r1, r2 (mechanism frequencies), m (class-1 occupancy; class 2 has 1 - m), and the
queue of pending class-1 shares. Timing follows the mean-field model: the cost of quarter q
uses r at its start; the share joins the queue; once the queue is longer than the delay the
oldest share enters the occupancy; then r takes one linear-selection step.
Only numpy: batched forward pass, exact adjoint gradient, projected Adam for the plan.
"""
import sys
import numpy as np


def simulate(P, r0, m0, kappa, delta, lam, delay):
    """P: (B, H) class-1 shares. Returns mean cost (B,), and the path (r1, r2, m) each (B, H+1)."""
    P = np.atleast_2d(P)
    B, H = P.shape
    r1 = np.full(B, r0[0]); r2 = np.full(B, r0[1]); m = np.full(B, m0)
    R1 = np.empty((B, H + 1)); R2 = np.empty((B, H + 1)); M = np.empty((B, H))
    cost = np.zeros(B)
    for q in range(H):
        R1[:, q], R2[:, q] = r1, r2
        cost += P[:, q] * r1 + (1 - P[:, q]) * r2
        if q >= delay:
            m = lam * m + (1 - lam) * P[:, q - delay]
        M[:, q] = m
        r1 = r1 + kappa * m * (1 - r1) - delta * r1
        r2 = r2 + kappa * (1 - m) * (1 - r2) - delta * r2
    R1[:, H], R2[:, H] = r1, r2
    return cost / H, (R1, R2, M)


def gradient(P, r0, m0, kappa, delta, lam, delay):
    P = np.atleast_2d(P)
    B, H = P.shape
    J, (R1, R2, M) = simulate(P, r0, m0, kappa, delta, lam, delay)
    g = (R1[:, :H] - R2[:, :H]) / H
    a1 = np.zeros(B); a2 = np.zeros(B); gm_next = np.zeros(B)
    for q in range(H - 1, -1, -1):
        # m_q affects r_{q+1}
        gm = a1 * kappa * (1 - R1[:, q]) - a2 * kappa * (1 - R2[:, q])
        gm = gm + (lam if q + 1 >= delay else 1.0) * gm_next if q + 1 < H else gm
        if q >= delay:
            g[:, q - delay] += (1 - lam) * gm
        gm_next = gm
        a1_new = P[:, q] / H + a1 * (1 - delta - kappa * M[:, q])
        a2_new = (1 - P[:, q]) / H + a2 * (1 - delta - kappa * (1 - M[:, q]))
        a1, a2 = a1_new, a2_new
    return J, g


def greedy(H, r0, m0, kappa, delta, lam, delay):
    r1, r2 = r0; m = m0; sched = []; total = 0.0
    for q in range(H):
        pq = 0.5 if np.isclose(r1, r2) else (1.0 if r1 < r2 else 0.0)
        sched.append(pq)
        total += pq * r1 + (1 - pq) * r2
        if q >= delay:
            m = lam * m + (1 - lam) * sched[q - delay]
        r1 = r1 + kappa * m * (1 - r1) - delta * r1
        r2 = r2 + kappa * (1 - m) * (1 - r2) - delta * r2
    return total / H, np.array(sched)


def rotation_bank(H):
    q = np.arange(H); out = []
    for first_class in (0, 1):
        for period in range(1, H + 1):
            for first in range(1, period + 1):
                block = np.where(q < first, 0, 1 + (q - first) // period)
                out.append(np.where((block + first_class) % 2 == 0, 1.0, 0.0))
    return np.array(out)


def plan(H, r0, m0, kappa, delta, lam, delay, extra=(), iters=3000, lr=0.02):
    bank = rotation_bank(H)
    vals, _ = simulate(bank, r0, m0, kappa, delta, lam, delay)
    best_rot = bank[np.argmin(vals)]
    starts = np.array([best_rot, np.full(H, 0.5), *extra])
    P = np.clip(starts, 0.0, 1.0).astype(float)
    mom = np.zeros_like(P); v = np.zeros_like(P)
    best_val = np.full(len(P), np.inf); best_P = P.copy()
    for t in range(1, iters + 1):
        J, g = gradient(P, r0, m0, kappa, delta, lam, delay)
        imp = J < best_val
        best_val[imp] = J[imp]; best_P[imp] = P[imp]
        mom = 0.9 * mom + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        step = lr * (mom / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
        P = np.clip(P - step * (0.3 + 0.7 * (1 - t / iters)), 0.0, 1.0)
    J, _ = simulate(P, r0, m0, kappa, delta, lam, delay)
    imp = J < best_val
    best_val[imp] = J[imp]; best_P[imp] = P[imp]
    i = int(np.argmin(best_val))
    return best_val[i], best_P[i], vals.min()


def check_gradient():
    rng = np.random.default_rng(0)
    for delay, lam in ((0, 0.0), (2, 0.8), (8, 0.95)):
        P = rng.uniform(0.1, 0.9, (1, 40))
        _, g = gradient(P, (0.04, 0.06), 0.5, 0.1, 0.05, lam, delay)
        fd = np.zeros(40)
        for k in range(40):
            e = np.zeros((1, 40)); e[0, k] = 1e-6
            fd[k] = (simulate(P + e, (0.04, 0.06), 0.5, 0.1, 0.05, lam, delay)[0][0]
                     - simulate(P - e, (0.04, 0.06), 0.5, 0.1, 0.05, lam, delay)[0][0]) / 2e-6
        print("gradient check", delay, lam, float(np.abs(fd - g[0]).max()))


if __name__ == "__main__":
    check_gradient()
    H = 40
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    for kappa, delta in [(0.1, 0.05), (0.2, 0.05), (0.1, 0.1)]:
        print(f"kappa {kappa} delta {delta}  start r0 = 0.05 -/+ {eps}, m0 = 0.5")
        print("  delay  lam   F_greedy0  F_plan   F_rot    F_unif   G0=Fg-Fp  switches")
        for delay in (0, 1, 2, 4, 8):
            for lam in (0.0, 0.5, 0.8, 0.95):
                r0 = (0.05 - eps, 0.05 + eps)
                fg, gs = greedy(H, r0, 0.5, kappa, delta, lam, delay)
                fp, ps, frot = plan(H, r0, 0.5, kappa, delta, lam, delay, extra=(gs,))
                fu = simulate(np.full(H, 0.5), r0, 0.5, kappa, delta, lam, delay)[0][0]
                sw = int(np.sum(np.abs(np.diff(gs)) > 0.5))
                print(f"  {delay:5d} {lam:5.2f}  {fg:.5f}   {fp:.5f}  {frot:.5f}  {fu:.5f}  {fg-fp:+.5f}  {sw}")
