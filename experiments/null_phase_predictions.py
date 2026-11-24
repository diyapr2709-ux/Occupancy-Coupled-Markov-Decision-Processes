"""Theory-free predictions for held-out settings: each (delay, lam) line's crossing on the dev map, carried over unchanged.

    python experiments/null_phase_predictions.py runs/phase/phase_<dev>.json settings.json \\
        --contrast planned_once_minus_greedy --sign-beyond-crossing 1 > null_predictions.json

settings.json is a JSON list of {"kappa", "delta", "horizon"}. For every held-out setting
and every (delay, lam) line of the dev report, the predicted crossing is the one measured
on dev for that line, in the format experiments/score_phase_predictions.py reads. A
theory earns the word "predicts" only by placing crossings better than this.
"""

import argparse
import json
import sys
from pathlib import Path

from score_phase_predictions import crossing, measured_lines
from sweep_phase_diagram import CONTRASTS


def null_predictions(dev_report: dict, contrast: str, beyond: int, settings: list[dict]) -> dict:
    lines, _ = measured_lines([dev_report], contrast)
    rows = []
    for (_, _, _, delay, lam), (sigmas, values) in sorted(lines.items()):
        sigma_star, sign = crossing(sigmas, values.mean(axis=1))
        for setting in settings:
            rows.append({**{k: setting[k] for k in ("kappa", "delta", "horizon")}, "delay": delay, "lam": lam,
                         "sigma_star": sigma_star, "sign_at_max_sigma": sign})  # fmt: skip
    return {"contrast": contrast, "sign_beyond_crossing": beyond, "lines": rows, "fitted_on": dev_report.get("evaluation_episode_seeds")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_report", type=Path)
    parser.add_argument("settings", type=Path)
    parser.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    parser.add_argument("--sign-beyond-crossing", type=int, required=True, choices=(1, -1))
    args = parser.parse_args()
    report = json.loads(args.dev_report.read_text())
    if report["args"]["tier"] != "dev":
        parser.error("the null is fitted on the dev map only")
    print(json.dumps(null_predictions(report, args.contrast, args.sign_beyond_crossing, json.loads(args.settings.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
