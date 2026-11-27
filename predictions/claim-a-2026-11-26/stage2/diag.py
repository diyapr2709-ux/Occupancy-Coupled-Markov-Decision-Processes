"""Stage 2: expected planned-once minus greedy contrast on the sigma grid, averaged over the start-state jitter.

Start state (toy.py reset with r0_spread 0.04, mu0_spread 0.5, four drugs in classes [2, 2]):
  r0_j = 0.05 + U(-0.04, 0.04) independently per mechanism (no clipping is reached);
  mu0 = 0.5 * uniform + 0.5 * Dirichlet(1, 1, 1, 1), so the class-1 occupancy is m0 = 0.25 + 0.5 b, b ~ Beta(2, 2).
The jitter expectation is a deterministic equal-weight midpoint rule over 4 x 5 x 5 start states.
Planned-once: best class rotation per start, polished by projected Adam on the exact gradient (noise-free optimum);
its expected value does not depend on sigma (affine dynamics in r under a fixed schedule, odd reflected shock).
Greedy: exact noise-free rollout at sigma = 0; Gaussian moment closure for sigma > 0.
Self-contained: numpy and the standard library only. No random draws anywhere.
"""
import json
import math
import sys
import numpy as np

SQ2 = math.sqrt(2.0)
H = 40


def Phi(u):
    return 0.5 * (1.0 + math.erf(u / SQ2))


def phi(u):
    return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


# folded-variance ratio h(t) = Var(fold(N(0, s^2), b)) / s^2 depends only on t = s / b
_T = np.exp(np.linspace(np.log(0.02), np.log(200.0), 400))
_Hr = []
for t in _T:
    s, b = t, 1.0
    x = np.linspace(-10 * s, 10 * s, 40001)
    w = np.exp(-0.5 * (x / s) ** 2); w /= w.sum()
    y = np.mod(x + b, 4 * b)
    f = np.where(y > 2 * b, 4 * b - y, y) - b
    _Hr.append(float((w * f * f).sum()) / (s * s))
_Hr = np.array(_Hr)


def fold_ratio(s, b):
    if s <= 0 or b <= 0:
        return 1.0
    t = s / b
    if t <= _T[0]:
        return 1.0
    if t >= _T[-1]:
        return float(_Hr[-1] * (_T[-1] / t) ** 2)
    return float(np.interp(np.log(t), np.log(_T), _Hr))


# ---------- noise-free model, vectorised over rows (each row has its own start) ----------
def simulate(P, r1, r2, m0, kappa, delta, lam, delay):
    P = np.atleast_2d(P); B = P.shape[0]
    r1 = np.broadcast_to(r1, (B,)).astype(float).copy(); r2 = np.broadcast_to(r2, (B,)).astype(float).copy()
    m = np.broadcast_to(m0, (B,)).astype(float).copy()
    R1 = np.empty((B, H + 1)); R2 = np.empty((B, H + 1)); M = np.empty((B, H)); cost = np.zeros(B)
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


def gradient(P, r1, r2, m0, kappa, delta, lam, delay):
    J, (R1, R2, M) = simulate(P, r1, r2, m0, kappa, delta, lam, delay)
    B = P.shape[0]
    g = (R1[:, :H] - R2[:, :H]) / H
    a1 = np.zeros(B); a2 = np.zeros(B); gm_next = np.zeros(B)
    for q in range(H - 1, -1, -1):
        gm = a1 * kappa * (1 - R1[:, q]) - a2 * kappa * (1 - R2[:, q])
        if q + 1 < H:
            gm = gm + (lam if q + 1 >= delay else 1.0) * gm_next
        if q >= delay:
            g[:, q - delay] += (1 - lam) * gm
        gm_next = gm
        a1, a2 = (P[:, q] / H + a1 * (1 - delta - kappa * M[:, q]),
                  (1 - P[:, q]) / H + a2 * (1 - delta - kappa * (1 - M[:, q])))
    return J, g


def greedy_det(r1, r2, m0, kappa, delta, lam, delay):
    B = len(r1); r1 = r1.copy(); r2 = r2.copy(); m = m0.copy()
    S = np.empty((B, H)); cost = np.zeros(B)
    for q in range(H):
        tie = np.isclose(r1, r2)
        p = np.where(tie, 0.5, np.where(r1 < r2, 1.0, 0.0))
        S[:, q] = p
        cost += p * r1 + (1 - p) * r2
        if q >= delay:
            m = lam * m + (1 - lam) * S[:, q - delay]
        r1 = r1 + kappa * m * (1 - r1) - delta * r1
        r2 = r2 + kappa * (1 - m) * (1 - r2) - delta * r2
    return cost / H, S


def rotation_bank():
    q = np.arange(H); out = []
    for first_class in (0, 1):
        for period in range(1, H + 1):
            for first in range(1, period + 1):
                block = np.where(q < first, 0, 1 + (q - first) // period)
                out.append(np.where((block + first_class) % 2 == 0, 1.0, 0.0))
    return np.array(out)


BANK = rotation_bank()


def plan(r1, r2, m0, kappa, delta, lam, delay, greedy_sched, iters=400, lr=0.02):
    B, K = len(r1), len(BANK)
    vals, _ = simulate(np.tile(BANK, (B, 1)), np.repeat(r1, K), np.repeat(r2, K), np.repeat(m0, K), kappa, delta, lam, delay)
    vals = vals.reshape(B, K)
    best_rot = BANK[vals.argmin(axis=1)]
    rot_val = vals.min(axis=1)
    P = np.vstack([best_rot, greedy_sched]); R1 = np.concatenate([r1, r1]); R2 = np.concatenate([r2, r2]); M0 = np.concatenate([m0, m0])
    mom = np.zeros_like(P); v = np.zeros_like(P)
    best_val = np.full(len(P), np.inf); best_P = P.copy()
    for t in range(1, iters + 1):
        J, g = gradient(P, R1, R2, M0, kappa, delta, lam, delay)
        imp = J < best_val; best_val[imp] = J[imp]; best_P[imp] = P[imp]
        mom = 0.9 * mom + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        step = lr * (mom / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
        P = np.clip(P - step * (0.3 + 0.7 * (1 - t / iters)), 0.0, 1.0)
    J, _ = simulate(P, R1, R2, M0, kappa, delta, lam, delay)
    best_val = np.minimum(best_val, J)
    return np.minimum(np.minimum(best_val[:B], best_val[B:]), rot_val), rot_val


# ---------- Gaussian moment closure of greedy ----------
def greedy_closure(r0, m0, kappa, delta, lam, delay, sigma, fold=True):
    mu = np.array([r0[0], r0[1], m0]); S = np.zeros((3, 3)); total = 0.0; TRACE = []
    for q in range(H):
        n = len(mu)
        cx = S[0] - S[1]; mx = mu[0] - mu[1]; vx = cx[0] - cx[1]
        if vx < 1e-16:
            pm = 0.5 if abs(mx) < 1e-12 else (1.0 if mx < 0 else 0.0)
            total += mu[1] + pm * mx; pc = np.zeros(n); pv = 0.0
        else:
            sx = math.sqrt(vx); u = mx / sx
            pm = Phi(-u); total += mu[1] + mx * Phi(-u) - sx * phi(u)
            pc = -(phi(u) / sx) * cx; pv = pm * (1 - pm)
        mu = np.append(mu, pm)
        S2 = np.zeros((n + 1, n + 1)); S2[:n, :n] = S; S2[n, :n] = S2[:n, n] = pc; S2[n, n] = pv; S = S2
        if q >= delay:
            n = len(mu); L = np.eye(n); L[2, 2] = lam; L[2, 3] = 1 - lam; L = np.delete(L, 3, axis=0)
            mu = L @ mu; S = L @ S @ L.T
        n = len(mu)
        # r1' = (1-d) r1 + k m - k m r1 ; r2' = k + (1-d-k) r2 - k m + k m r2
        s02, s12 = S[0, 2], S[1, 2]
        new0 = (1 - delta) * mu[0] + kappa * mu[2] - kappa * (mu[0] * mu[2] + s02)
        new1 = kappa + (1 - delta - kappa) * mu[1] - kappa * mu[2] + kappa * (mu[1] * mu[2] + s12)
        G0 = np.zeros(n); G0[0] = 1 - delta - kappa * mu[2]; G0[2] = kappa - kappa * mu[0]
        G1 = np.zeros(n); G1[1] = 1 - delta - kappa + kappa * mu[2]; G1[2] = -kappa + kappa * mu[1]
        A = np.eye(n); A[0] = G0; A[1] = G1
        newS = A @ S @ A.T
        # exact Gaussian quadratic corrections: Q0 = -k/2 (e0 e2^T + e2 e0^T), Q1 = +k/2 (e1 e2^T + e2 e1^T)
        def trQSQS(i, a, j, b, ci, cj):
            # 2 tr(Qi S Qj S) for Qi = ci/2 (e_i e_a^T + e_a e_i^T)
            return 2 * (ci / 2) * (cj / 2) * 2 * (S[a, j] * S[b, i] + S[a, b] * S[i, j])
        newS[0, 0] += trQSQS(0, 2, 0, 2, -kappa, -kappa)
        newS[1, 1] += trQSQS(1, 2, 1, 2, kappa, kappa)
        c01 = trQSQS(0, 2, 1, 2, -kappa, kappa); newS[0, 1] += c01; newS[1, 0] += c01
        mu = mu.copy(); mu[0], mu[1] = new0, new1; S = newS
        TRACE.append((mu[0], mu[1], S[0, 0], S[1, 1], S[0, 0] + S[1, 1] - 2 * S[0, 1]))
        if sigma > 0:
            for j in range(2):
                er = max(mu[j] - mu[j] ** 2 - S[j, j], 0.0)
                ratio = 1.0
                if fold:
                    mj = min(max(mu[j], 1e-9), 1 - 1e-9)
                    ratio = fold_ratio(sigma * math.sqrt(mj * (1 - mj)), min(mj, 1 - mj))
                S[j, j] += sigma * sigma * er * ratio
    return total / H, TRACE


def starts():
    u1 = np.array([-0.03, -0.01, 0.01, 0.03])
    u2 = np.array([-0.032, -0.016, 0.0, 0.016, 0.032])
    # Beta(2, 2) quantile midpoints: solve 3b^2 - 2b^3 = p
    bs = []
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if 3 * mid ** 2 - 2 * mid ** 3 < p else (lo, mid)
        bs.append(0.5 * (lo + hi))
    g = np.array([(0.05 + a, 0.05 + c, 0.25 + 0.5 * b) for a in u1 for c in u2 for b in bs])
    return g[:, 0], g[:, 1], g[:, 2]


def check_closure_quadratic():
    """The hand-written quadratic corrections must match the generic formula 2 tr(Qi S Qj S)."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); S = A @ A.T
    k = 0.1
    Q0 = np.zeros((5, 5)); Q0[0, 2] = Q0[2, 0] = -k / 2
    Q1 = np.zeros((5, 5)); Q1[1, 2] = Q1[2, 1] = k / 2
    def trQSQS(i, a, j, b, ci, cj):
        return 2 * (ci / 2) * (cj / 2) * 2 * (S[a, j] * S[b, i] + S[a, b] * S[i, j])
    errs = [abs(trQSQS(0, 2, 0, 2, -k, -k) - 2 * np.trace(Q0 @ S @ Q0 @ S)),
            abs(trQSQS(1, 2, 1, 2, k, k) - 2 * np.trace(Q1 @ S @ Q1 @ S)),
            abs(trQSQS(0, 2, 1, 2, -k, k) - 2 * np.trace(Q0 @ S @ Q1 @ S))]
    return max(errs)



if __name__ == "__main__":
    # Diagnostics of the closure's Gaussian: mass outside [0, 1] and spread, start r0 = (0.04, 0.06), m0 = 0.5
    for kappa, delta in [(0.1, 0.05), (0.2, 0.05), (0.1, 0.1)]:
        for delay, lam in [(0, 0.0), (2, 0.5), (8, 0.0), (0, 0.95)]:
            for s in (0.1, 0.2, 0.4):
                _, tr = greedy_closure((0.04, 0.06), 0.5, kappa, delta, lam, delay, s)
                tr = np.array(tr)
                out_mass = []
                for m1, m2, v1, v2, vx in tr:
                    for m, v in ((m1, v1), (m2, v2)):
                        sd = math.sqrt(max(v, 1e-30))
                        out_mass.append(Phi(-m / sd) + 1 - Phi((1 - m) / sd))
                print(f"k {kappa} d {delta} delay {delay} lam {lam:4.2f} sigma {s}: max P(r outside [0,1]) {max(out_mass):.3f}, "
                      f"mean P {np.mean(out_mass):.3f}, mean sd(r) {np.mean(np.sqrt(tr[:, 2:4])):.3f}, mean sd(x) {np.mean(np.sqrt(np.maximum(tr[:, 4], 0))):.3f}, mean r {tr[:, :2].mean():.3f}")
