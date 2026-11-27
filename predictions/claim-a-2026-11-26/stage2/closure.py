"""Gaussian moment closure of the myopic greedy rule under resistance noise (no sampling).

The closed loop state z = (r1, r2, m, pending shares) is carried as a Gaussian (mean, cov).
Each quarter: the greedy share p = 1[x < 0], x = r1 - r2, is replaced by its statistically
linearised Gaussian (exact mean, exact variance, Stein cross-covariances); the occupancy update
is linear; the selection step is quadratic in (m, r), whose mean and covariance are exact for a
Gaussian input; the shock adds variance sigma^2 E[r(1-r)] reduced by the reflection fold.
The per-quarter expected cost E[r2 + x 1[x < 0]] is exact for Gaussian x.
Self-contained: no imports beyond numpy and the standard library.
"""
import math
import sys
import numpy as np

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


if __name__ == "__main__":
    H = 40; eps = 0.01
    sig = (0.0, 0.05, 0.1, 0.2, 0.4)
    for kappa, delta in [(0.1, 0.05), (0.2, 0.05), (0.1, 0.1)]:
        print(f"kappa {kappa} delta {delta}: greedy expected cost by sigma {sig} (closure), det check")
        for delay in (0, 1, 2, 4, 8):
            for lam in (0.0, 0.5, 0.8, 0.95):
                r0 = (0.05 - eps, 0.05 + eps)
                vals = [greedy_closure(H, r0, 0.5, kappa, delta, lam, delay, s) for s in sig]
                nf = [greedy_closure(H, r0, 0.5, kappa, delta, lam, delay, s, fold=False) for s in sig[3:]]
                d = greedy_det(H, r0, 0.5, kappa, delta, lam, delay)
                print(f"  {delay} {lam:4.2f} det {d:.5f} | " + " ".join(f"{v:.5f}" for v in vals)
                      + " | gain " + " ".join(f"{vals[0]-v:+.4f}" for v in vals[1:]) + " | nofold " + " ".join(f"{vals[0]-v:+.4f}" for v in nf))
