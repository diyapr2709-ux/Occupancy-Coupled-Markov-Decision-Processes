"""Draw one contrast of a phase-diagram report as small multiples, with an optional prediction overlay.

    python experiments/plot_phase_diagram.py runs/phase/phase_<stamp>.json
    python experiments/plot_phase_diagram.py runs/phase/phase_<stamp>.json --contrast greedy_over_rotation
    python experiments/plot_phase_diagram.py runs/phase/phase_<stamp>.json --predictions predictions.json

One panel per occupancy memory lam, delay on the x axis, one line per resistance noise
level with its 95% interval at every point. Noise levels are ordered, so they take one
hue from light to dark; sigma = 0 is grey, the floor the noisy lines are read against.
A prediction file (a JSON list of {"delay", "lam", "sigma", <contrast>: value}) is drawn
dashed with hollow markers in the same colours. Means and exact Student-t 95% intervals
are recomputed from the per-episode values in the report, not read from its summaries.
Writes an SVG and a CSV of every plotted value to figures/, named after the report.
Standard library only.
"""

import argparse
import math
import csv
import json
import sys
from pathlib import Path

from stats import t_interval

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
FLOOR = "#8a8984"  # sigma = 0
RAMP = ("#86b6ef", "#2a78d6", "#104281")  # validated ordinal blue steps 250 / 450 / 650, light to dark
PANEL_W, PANEL_H, GAP_X, GAP_Y, LEFT, TOP, BOTTOM, COLUMNS = 260, 190, 40, 70, 70, 84, 48, 2


def interval(values: list[float]) -> dict:
    """Mean and exact 95% Student-t half-width."""
    ci = t_interval(values)
    return {"mean": ci["mean"], "ci95": ci["half_width"]}


def series_colours(sigmas: list[float]) -> dict[float, str]:
    noisy = sorted(s for s in sigmas if s > 0)
    if len(noisy) > len(RAMP):
        raise ValueError(f"at most {len(RAMP)} noisy levels fit the ramp, got {noisy}")
    steps = {1: RAMP[1:2], 2: RAMP[::2], 3: RAMP}.get(len(noisy), ())
    return {**{s: c for s, c in zip(noisy, steps)}, **({0.0: FLOOR} if 0.0 in sigmas else {})}


def draw(report: dict, contrast: str, predictions: list[dict] | None) -> tuple[str, list[dict]]:
    cells = report["cells"]
    delays = sorted({c["delay"] for c in cells})
    lams = sorted({c["lam"] for c in cells})
    sigmas = sorted({c["sigma"] for c in cells})
    colour = series_colours(sigmas)
    minuend, subtrahend = report["contrasts"][contrast]
    raw = {(r["delay"], r["lam"], r["sigma"], r["policy"]): r["expected_failure_rate"] for r in report["results"]}
    lookup = {
        (c["delay"], c["lam"], c["sigma"]): interval([a - b for a, b in zip(raw[(c["delay"], c["lam"], c["sigma"], minuend)], raw[(c["delay"], c["lam"], c["sigma"], subtrahend)])])
        for c in cells
    }
    predicted = {(p["delay"], p["lam"], p["sigma"]): p[contrast] for p in predictions or [] if contrast in p}

    lows = [a["mean"] - a["ci95"] for a in lookup.values()] + list(predicted.values()) + [0.0]
    highs = [a["mean"] + a["ci95"] for a in lookup.values()] + list(predicted.values()) + [0.0]
    lo, hi = min(lows), max(highs)
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad

    def origin(panel: int) -> tuple[float, float]:
        row, col = divmod(panel, COLUMNS)
        return LEFT + col * (PANEL_W + GAP_X), TOP + row * (PANEL_H + GAP_Y)

    def y(panel: int, v: float) -> float:
        return origin(panel)[1] + PANEL_H * (hi - v) / (hi - lo)

    def x(panel: int, i: int, s: float) -> float:
        offset = (s - (len(sigmas) - 1) / 2) * 5  # keep the intervals of different sigmas apart
        return origin(panel)[0] + 18 + i * (PANEL_W - 36) / max(len(delays) - 1, 1) + offset

    rows_of_panels = -(-len(lams) // COLUMNS)
    width = LEFT + COLUMNS * (PANEL_W + GAP_X) + 30
    height = TOP + rows_of_panels * (PANEL_H + GAP_Y) - GAP_Y + BOTTOM
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="22" font-size="14" font-weight="bold" fill="{INK}">{contrast.replace("_", " ")}</text>',
        f'<text x="{LEFT}" y="40" fill="{INK_2}">difference in expected failure rate, mean and 95% interval over '
        f'{len(report["evaluation_episode_seeds"])} {report["args"]["tier"]}-tier episodes{"; dashed: prediction" if predicted else ""}</text>',
    ]
    # legend, one entry per noise level, under the subtitle
    for k, s in enumerate(sigmas):
        out.append(f'<circle cx="{LEFT + 4 + k * 76}" cy="56" r="4" fill="{colour[s]}"/>')
        out.append(f'<text x="{LEFT + 12 + k * 76}" y="60" fill="{INK_2}">σ = {s:g}</text>')

    rows = []
    for panel, lam in enumerate(lams):
        px, py = origin(panel)
        out.append(f'<text x="{px + PANEL_W / 2}" y="{py - 8}" text-anchor="middle" fill="{INK}">λ = {lam:g}</text>')
        for tick in _ticks(lo, hi):
            out.append(f'<line x1="{px}" x2="{px + PANEL_W}" y1="{y(panel, tick):.1f}" y2="{y(panel, tick):.1f}" stroke="{GRID}" stroke-width="1"/>')
            if panel % COLUMNS == 0:
                out.append(f'<text x="{px - 6}" y="{y(panel, tick) + 4:.1f}" text-anchor="end" fill="{INK_2}">{tick:+.3f}</text>')
        out.append(f'<line x1="{px}" x2="{px + PANEL_W}" y1="{y(panel, 0):.1f}" y2="{y(panel, 0):.1f}" stroke="{INK_2}" stroke-width="1"/>')
        for i, d in enumerate(delays):
            out.append(f'<text x="{x(panel, i, (len(sigmas) - 1) / 2):.1f}" y="{py + PANEL_H + 18}" text-anchor="middle" fill="{INK_2}">{d}</text>')
        out.append(f'<text x="{px + PANEL_W / 2}" y="{py + PANEL_H + 36}" text-anchor="middle" fill="{INK_2}">delay (quarters)</text>')
        for s_index, s in enumerate(sigmas):
            measured = [(i, lookup[(d, lam, s)]) for i, d in enumerate(delays) if (d, lam, s) in lookup]
            points = " ".join(f"{x(panel, i, s_index):.1f},{y(panel, a['mean']):.1f}" for i, a in measured)
            out.append(f'<polyline points="{points}" fill="none" stroke="{colour[s]}" stroke-width="2"/>')
            for i, a in measured:
                cx = x(panel, i, s_index)
                out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(panel, a["mean"] - a["ci95"]):.1f}" y2="{y(panel, a["mean"] + a["ci95"]):.1f}" stroke="{colour[s]}" stroke-width="1.5"/>')
                out.append(f'<circle cx="{cx:.1f}" cy="{y(panel, a["mean"]):.1f}" r="4" fill="{colour[s]}" stroke="{SURFACE}" stroke-width="2"/>')
                rows.append({"delay": delays[i], "lam": lam, "sigma": s, "mean": a["mean"], "ci95": a["ci95"],
                             "predicted": predicted.get((delays[i], lam, s), "")})  # fmt: skip
            guessed = [(i, predicted[(d, lam, s)]) for i, d in enumerate(delays) if (d, lam, s) in predicted]
            if guessed:
                points = " ".join(f"{x(panel, i, s_index):.1f},{y(panel, v):.1f}" for i, v in guessed)
                out.append(f'<polyline points="{points}" fill="none" stroke="{colour[s]}" stroke-width="2" stroke-dasharray="5 4"/>')
                for i, v in guessed:
                    out.append(f'<circle cx="{x(panel, i, s_index):.1f}" cy="{y(panel, v):.1f}" r="4" fill="{SURFACE}" stroke="{colour[s]}" stroke-width="2"/>')
        if panel == len(lams) - 1:  # direct labels at the line ends of the last panel, pushed apart to stay legible
            ends = sorted((y(panel, lookup[(delays[-1], lam, s)]["mean"]), s) for s in sigmas if (delays[-1], lam, s) in lookup)
            placed = []
            for end, s in ends:
                placed.append((max(end, placed[-1][0] + 13) if placed else end, s))
            for label_y, s in placed:
                out.append(f'<text x="{x(panel, len(delays) - 1, len(sigmas)) + 6:.1f}" y="{label_y + 4:.1f}" fill="{INK_2}">σ {s:g}</text>')
    out.append("</svg>")
    return "\n".join(out), rows


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step) * step
    return [first + k * step for k in range(int((hi - first) / step) + 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="a runs/phase/phase_*.json report")
    parser.add_argument("--contrast", default="value_of_feedback")
    parser.add_argument("--predictions", type=Path, help="JSON list of predicted cell values to overlay")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.contrast not in report["contrasts"]:
        parser.error(f"--contrast must be one of {sorted(report['contrasts'])}")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8")) if args.predictions else None
    if isinstance(predictions, dict):  # also accept {"cells": [...]}
        predictions = predictions.get("cells", [])
    svg, rows = draw(report, args.contrast, predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.report.stem}_{args.contrast}"
    (args.out / f"{stem}.svg").write_text(svg, encoding="utf-8")
    with open(args.out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["delay", "lam", "sigma", "mean", "ci95", "predicted"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out / stem}.svg and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
