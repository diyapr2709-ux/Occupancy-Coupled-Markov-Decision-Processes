"""Turn stage2_results.json into predictions.json with the scorer's own crossing rule, plus sensitivity tables."""
import json
import math
import sys
import numpy as np

GRID = (0.0, 0.1, 0.2, 0.4)
LOW, HIGH = 0.1, 0.4


def crossing(sigmas, means, beyond=1):
    """Copied rule from score_phase_predictions.crossing."""
    nonzero = [(s, m) for s, m in zip(sigmas, means) if m != 0]
    if not nonzero:
        return None, 0
    last = int(np.sign(nonzero[-1][1]))
    before = [i for i, (_, m) in enumerate(nonzero) if np.sign(m) != beyond]
    if last != beyond or not before:
        return None, last
    (s0, m0), (s1, m1) = nonzero[before[-1]], nonzero[before[-1] + 1]
    zeros = [s for s, m in zip(sigmas, means) if m == 0 and s0 < s < s1]
    return float(zeros[0] if zeros else s0 + (s1 - s0) * m0 / (m0 - m1)), last


def place(c, sign):
    if c is None:
        return LOW if sign == 1 else HIGH
    return min(max(c, LOW), HIGH)


def contrast(rec, scale=1.0, nofold=False, grid=GRID):
    out = []
    for s in grid:
        if s == 0:
            out.append(-rec["G0"])
            continue
        fg = rec["F_greedy_nofold"][str(s)] if nofold and str(s) in rec["F_greedy_nofold"] else rec["F_greedy"][str(s)]
        gain = rec["F_greedy0"] - fg
        out.append(-rec["G0"] + scale * gain)
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    recs = json.load(open(src, encoding="utf-8"))
    lines = []
    fine = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
    for kd in sorted({(r["kappa"], r["delta"]) for r in recs}, key=lambda t: (t[0], t[1])):
        print(f"\nkappa {kd[0]} delta {kd[1]}")
        print(" dly  lam     G0      C(0.1)   C(0.2)   C(0.4)  | sig*grid placed | fine-grid sig* | x0.6  x0.75 x1.25 x1.5 nofold")
        for r in [r for r in recs if (r["kappa"], r["delta"]) == kd]:
            C = contrast(r)
            c, sg = crossing(GRID, C)
            cf, sgf = crossing(fine, contrast(r, grid=fine))
            sens = [place(*crossing(GRID, contrast(r, scale=f))) for f in (0.6, 0.75, 1.25, 1.5)]
            nf = place(*crossing(GRID, contrast(r, nofold=True)))
            lines.append({"kappa": r["kappa"], "delta": r["delta"], "horizon": 40, "delay": r["delay"], "lam": r["lam"],
                          "sigma_star": None if c is None else round(c, 4), "sign_at_max_sigma": sg})
            cs = "None" if c is None else f"{c:.3f}"
            cfs = "None" if cf is None else f"{cf:.3f}"
            print(f" {r['delay']:3d} {r['lam']:4.2f} {r['G0']:+.4f}  {C[1]:+.4f}  {C[2]:+.4f}  {C[3]:+.4f} | {cs:>6} {place(c, sg):.3f} | {cfs:>6} | "
                  + " ".join(f"{v:.3f}" for v in sens) + f" {nf:.3f}")
    pred = {"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": 1, "lines": lines,
            "directions": [{"from": {"kappa": 0.1, "delta": 0.05, "horizon": 40}, "to": {"kappa": 0.2, "delta": 0.05, "horizon": 40}, "sign": 1}]}
    # directional shifts between settings on lines where both predicted crossings are numeric
    by = {(l["kappa"], l["delta"], l["delay"], l["lam"]): l["sigma_star"] for l in lines}
    for to in ((0.2, 0.05), (0.1, 0.1)):
        shifts = [math.log(by[(*to, d, lam)] / by[(0.1, 0.05, d, lam)]) for (k, de, d, lam) in by
                  if (k, de) == (0.1, 0.05) and by[(k, de, d, lam)] and by[(*to, d, lam)]]
        print(f"\nlog shift base -> {to}: n {len(shifts)}, positive {sum(s > 0 for s in shifts)}, median {np.median(shifts):+.3f}")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(pred, fh, indent=1)
    print("wrote", dst, len(lines), "lines")
