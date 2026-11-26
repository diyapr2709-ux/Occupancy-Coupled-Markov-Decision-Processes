"""Theory-free predictions for held-out settings: each (delay, lam) line's crossing on the dev map, carried over unchanged.

    python experiments/null_phase_predictions.py runs/phase/phase_<dev>.json settings.json \\
        --contrast planned_once_minus_greedy --sign-beyond-crossing 1 > null_predictions.json

settings.json is a JSON list of {"kappa", "delta", "horizon"}, or an object holding that
list under "settings" (as experiments/claim_a_heldout_settings.json does). For every held-out setting
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
        sigma_star, sign = crossing(sigmas, values.mean(axis=1), beyond)
        for setting in settings:
            rows.append({**{k: setting[k] for k in ("kappa", "delta", "horizon")}, "delay": delay, "lam": lam,
                         "sigma_star": sigma_star, "sign_at_max_sigma": sign})  # fmt: skip
    return {"contrast": contrast, "sign_beyond_crossing": beyond, "lines": rows, "fitted_on": dev_report.get("evaluation_episode_seeds")}


def load_settings(path: Path) -> list[dict]:
    """The held-out settings: a bare list, or the list under "settings" in a held-out file that also records its rule."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    settings = spec["settings"] if isinstance(spec, dict) else spec
    if not isinstance(settings, list) or not all(isinstance(s, dict) and {"kappa", "delta", "horizon"} <= set(s) for s in settings):
        raise ValueError(f"{path} must hold a list of {{kappa, delta, horizon}} settings")
    return settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_report", type=Path)
    parser.add_argument("settings", type=Path)
    parser.add_argument("--contrast", required=True, choices=sorted(CONTRASTS))
    parser.add_argument("--sign-beyond-crossing", type=int, required=True, choices=(1, -1))
    args = parser.parse_args()
    report = json.loads(args.dev_report.read_text(encoding="utf-8"))
    if report["args"]["tier"] != "dev":
        parser.error("the null is fitted on the dev map only")
    try:
        settings = load_settings(args.settings)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(null_predictions(report, args.contrast, args.sign_beyond_crossing, settings), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
