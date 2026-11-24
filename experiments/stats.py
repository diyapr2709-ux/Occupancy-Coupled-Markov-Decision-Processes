"""Student-t quantiles and intervals, standard library only, so every experiment script can use them without scipy.

The evaluation harness's intervals read a 3-decimal t table and fall back to 1.96 above
30 degrees of freedom; these are exact to about 1e-9.
"""

import math


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b), by Lentz's continued fraction."""
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _incomplete_beta(1.0 - x, b, a)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)) / a
    tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    f = d
    for m in range(1, 300):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            f *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return front * f


def t_quantile(probability: float, df: int) -> float:
    """The `probability` quantile of Student's t with df degrees of freedom, for 1/2 < probability < 1, by bisection on its CDF."""
    if not 0.5 < probability < 1.0 or df < 1:
        raise ValueError(f"need 1/2 < probability < 1 and df >= 1, got probability {probability} and df {df}")

    def upper_tail(t: float) -> float:
        return 0.5 * _incomplete_beta(df / (df + t * t), df / 2.0, 0.5)

    lo, hi = 0.0, 1.0
    while upper_tail(hi) > 1.0 - probability:  # widen until the quantile is bracketed: heavy tails at small df
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if upper_tail(mid) > 1.0 - probability else (lo, mid)
    return (lo + hi) / 2.0


def t_interval(values, level: float = 0.95) -> dict:
    """Mean and the half-width of its two-sided Student-t interval at `level`."""
    values = list(values)
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"mean": mean, "half_width": t_quantile(0.5 + level / 2, n - 1) * sd / math.sqrt(n), "n": n}


def equivalent(values, margin: float) -> dict:
    """Two one-sided tests at the 5% level: equivalent when the 90% t-interval of the mean lies inside (-margin, margin)."""
    ci = t_interval(values, level=0.90)
    return {"margin": margin, "ci90": [ci["mean"] - ci["half_width"], ci["mean"] + ci["half_width"]],
            "equivalent": -margin < ci["mean"] - ci["half_width"] and ci["mean"] + ci["half_width"] < margin}  # fmt: skip
