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


import math
import sys

SQ2 = math.sqrt(2.0)
Phi = lambda u: 0.5 * (1.0 + math.erf(u / SQ2))
phi = lambda u: math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def folded_var(s, b):
    """Variance of N(0, s^2) folded into [-b, b] by reflection (period 4b), by quadrature."""
    if s <= 0 or b <= 0:
        return 0.0
    if s < 0.05 * b:
        return s * s
    x = np.linspace(-10 * s, 10 * s, 20001)
    w = np.exp(-0.5 * (x / s) ** 2); w /= w.sum()
    y = np.mod(x + b, 4 * b)
    f = np.where(y > 2 * b, 4 * b - y, y) - b
    return float((w * f * f).sum())


def greedy_closure(H, r0, m0, kappa, delta, lam, delay, sigma, fold=True, r0_var=(0.0, 0.0)):
    mu = np.array([r0[0], r0[1], m0]); S = np.diag([r0_var[0], r0_var[1], 0.0])
    costs = []
    for q in range(H):
        n = len(mu)
        cx = S[0] - S[1]
        mx, vx = mu[0] - mu[1], cx[0] - cx[1]
        if vx < 1e-16:
            p_mean = 0.5 if abs(mx) < 1e-12 else (1.0 if mx < 0 else 0.0)
            costs.append(mu[1] + p_mean * mx)
            p_cov = np.zeros(n); p_var = 0.0
        else:
            sx = math.sqrt(vx); u = mx / sx
            p_mean = Phi(-u)
            costs.append(mu[1] + mx * Phi(-u) - sx * phi(u))
            p_cov = -(phi(u) / sx) * cx
            p_var = p_mean * (1 - p_mean)
        mu = np.append(mu, p_mean)
        S2 = np.zeros((n + 1, n + 1)); S2[:n, :n] = S; S2[n, :n] = S2[:n, n] = p_cov; S2[n, n] = p_var
        S = S2
        if q >= delay:  # oldest pending (index 3) enters m, then is dropped
            n = len(mu)
            L = np.eye(n); L[2, 2] = lam; L[2, 3] = 1 - lam
            L = np.delete(L, 3, axis=0)
            mu = L @ mu; S = L @ S @ L.T
        # selection step: r1' = r1 + k m - k m r1 - d r1 ; r2' = r2 + k - k r2 - k m + k m r2 - d r2
        n = len(mu)
        b1 = np.zeros(n); b1[0] = 1 - delta; b1[2] = kappa
        Q1 = np.zeros((n, n)); Q1[0, 2] = Q1[2, 0] = -kappa / 2
        b2 = np.zeros(n); b2[1] = 1 - delta - kappa; b2[2] = -kappa
        Q2 = np.zeros((n, n)); Q2[1, 2] = Q2[2, 1] = kappa / 2
        c = [0.0, kappa]
        rows_b = [b1, b2]; rows_Q = [Q1, Q2]
        new_mu = mu.copy()
        G = []  # gradient vectors at mu for the two new r's
        for i in range(2):
            new_mu[i] = c[i] + rows_b[i] @ mu + mu @ rows_Q[i] @ mu + np.trace(rows_Q[i] @ S)
            G.append(rows_b[i] + 2 * rows_Q[i] @ mu)
        # linear map for all components (identity for m and pending), plus quadratic corrections
        A = np.eye(n); A[0] = G[0]; A[1] = G[1]
        newS = A @ S @ A.T
        for i in range(2):
            for j in range(2):
                newS[i, j] += 2 * np.trace(rows_Q[i] @ S @ rows_Q[j] @ S)
        mu, S = new_mu, newS
        if sigma > 0:
            for j in range(2):
                er = mu[j] - mu[j] ** 2 - S[j, j]
                er = max(er, 0.0)
                if fold:
                    mj = min(max(mu[j], 1e-9), 1 - 1e-9)
                    s0 = sigma * math.sqrt(mj * (1 - mj)); b = min(mj, 1 - mj)
                    ratio = folded_var(s0, b) / (s0 * s0) if s0 > 0 else 1.0
                else:
                    ratio = 1.0
                S[j, j] += sigma * sigma * er * ratio
    return float(np.mean(costs))


def greedy_det(H, r0, m0, kappa, delta, lam, delay):
    r1, r2 = r0; m = m0; sched = []; total = 0.0
    for q in range(H):
        pq = 0.5 if np.isclose(r1, r2) else (1.0 if r1 < r2 else 0.0)
        sched.append(pq); total += pq * r1 + (1 - pq) * r2
        if q >= delay:
            m = lam * m + (1 - lam) * sched[q - delay]
        r1 = r1 + kappa * m * (1 - r1) - delta * r1
        r2 = r2 + kappa * (1 - m) * (1 - r2) - delta * r2
    return total / H



def crossing(sigmas, means, beyond=1):
    nonzero = [(s, m) for s, m in zip(sigmas, means) if m != 0]
    if not nonzero:
        return None, 0
    last = int(np.sign(nonzero[-1][1]))
    before = [i for i, (_, m) in enumerate(nonzero) if np.sign(m) != beyond]
    if last != beyond or not before:
        return None, last
    (s0, m0), (s1, m1) = nonzero[before[-1]], nonzero[before[-1] + 1]
    return float(s0 + (s1 - s0) * m0 / (m0 - m1)), last


def place(c, sign, low=0.1, high=0.4):
    if c is None:
        return low if sign == 1 else high
    return min(max(c, low), high)


if __name__ == "__main__":
    H = 40
    grid = (0.0, 0.1, 0.2, 0.4)
    starts = {"eps0.002": [((0.048, 0.052), 0.5)], "eps0.01": [((0.04, 0.06), 0.5)],
              "eps0.03": [((0.02, 0.08), 0.5)], "eps0.01_mu0.45": [((0.04, 0.06), 0.45)]}
    for kappa, delta in [(0.1, 0.05), (0.2, 0.05), (0.1, 0.1)]:
        print(f"kappa {kappa} delta {delta}: placed crossing (censored to [0.1, 0.4]) per start; then G0 and C(grid) for eps0.01")
        for delay in (0, 1, 2, 4, 8):
            for lam in (0.0, 0.5, 0.8, 0.95):
                out = []
                for name, sl in starts.items():
                    C = np.zeros(len(grid)); G0s = []
                    for r0, m0 in sl:
                        fg = greedy_det(H, r0, m0, kappa, delta, lam, delay)
                        _, gs = greedy(H, r0, m0, kappa, delta, lam, delay)
                        fp, _, _ = plan(H, r0, m0, kappa, delta, lam, delay, extra=(gs,), iters=1000)
                        G0s.append(fg - fp)
                        C += np.array([fp - (fg if s == 0 else greedy_closure(H, r0, m0, kappa, delta, lam, delay, s)) for s in grid]) / len(sl)
                    C[0] = 0.0 if abs(C[0]) < 1e-9 else C[0]
                    c, sg = crossing(grid, C)
                    out.append((name, place(c, sg), c, G0s[0], C))
                row = " ".join(f"{o[0]}:{o[1]:.3f}" for o in out)
                base = out[1]
                print(f"  {delay} {lam:4.2f}  {row} | G0 {base[3]:+.4f} C " + " ".join(f"{v:+.4f}" for v in base[4]))
