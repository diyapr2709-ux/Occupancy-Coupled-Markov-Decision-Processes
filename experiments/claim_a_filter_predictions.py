"""Keep only the held-out lines of a predictions file, so the scorer sees exactly the lines the test measures.

    python experiments/claim_a_filter_predictions.py predictions/claim-a-2026-11-26/stage2/predictions.json \\
        experiments/claim_a_heldout_settings.json --out runs/claim_a/predictions_heldout.json

The blind derivation predicts every (kappa, delta, horizon) setting it was given, the dev
map's own setting included. experiments/score_phase_predictions.py refuses a predictions
file that names a line no report measures, and the dev map cannot be scored beside the
held-out reports because it was evaluated on other episodes. This script therefore drops
every line whose setting is not in the held-out file, and every direction check whose two
settings are not both held out, and changes nothing else: no value is read, rounded or
reordered beyond sorting the lines by their key. The output is written with sorted keys
and LF line endings so that its SHA-256 is the same on every platform, and that hash is
printed, so a preregistration can pin the filtered file before any held-out data exist.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from null_phase_predictions import load_settings

SETTING_KEYS = ("kappa", "delta", "horizon")


def filter_predictions(predictions: dict, settings: list[dict]) -> dict:
    held_out = {tuple(s[k] for k in SETTING_KEYS) for s in settings}

    def setting(row: dict) -> tuple:
        return tuple(row[k] for k in SETTING_KEYS)

    lines = sorted((row for row in predictions["lines"] if setting(row) in held_out),
                   key=lambda row: (setting(row), row["delay"], row["lam"]))  # fmt: skip
    directions = [d for d in predictions.get("directions", []) if setting(d["from"]) in held_out and setting(d["to"]) in held_out]
    missing = held_out - {setting(row) for row in lines}
    if missing:
        raise ValueError(f"the predictions file has no line at held-out setting(s) {sorted(missing)}")
    return {"contrast": predictions["contrast"], "sign_beyond_crossing": predictions["sign_beyond_crossing"],
            "lines": lines, "directions": directions, "filtered_from": {"lines": len(predictions["lines"]),
                                                                        "directions": len(predictions.get("directions", []))}}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("settings", type=Path, help="the held-out settings file (experiments/claim_a_heldout_settings.json)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        filtered = filter_predictions(json.loads(args.predictions.read_text(encoding="utf-8")), load_settings(args.settings))
    except (KeyError, ValueError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    text = json.dumps(filtered, indent=1, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {args.out.as_posix()}  "
          f"({len(filtered['lines'])} lines, {len(filtered['directions'])} directions kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
