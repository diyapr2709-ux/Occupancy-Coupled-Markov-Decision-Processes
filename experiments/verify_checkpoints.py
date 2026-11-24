"""Check a training report's checkpoints against the sizes and SHA-256 hashes the report records.

    python experiments/verify_checkpoints.py runs/train/train_<stamp>_<pid>.json
    python experiments/verify_checkpoints.py runs/train/train_<stamp>_<pid>.json --load

Checkpoint paths in a report are relative to the report's folder, so an archived
checkpoint folder is checked by restoring it next to its report. With --load, each
agent's final checkpoint is also rebuilt and its weight hash compared with the run's
recorded parameter_sha256.final. Prints every problem and exits 1 if there is any.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def verify(report_path: Path, load: bool = False) -> list[str]:
    """Every missing file, size or hash mismatch, and (with load) weight-hash mismatch, as one line each."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems = []
    for run in report["runs"]:
        entries = run.get("checkpoints", [])
        final = max((entry["decisions"] for entry in entries), default=None)
        for entry in entries:
            path = report_path.parent / entry["path"]
            if not path.is_file():
                problems.append(f"missing: {entry['path']}")
                continue
            blob = path.read_bytes()
            if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                problems.append(f"size or SHA-256 differs from the report: {entry['path']}")
                continue
            if load and entry["decisions"] == final and "parameter_sha256" in run:
                from ocmdp.agents.oc_sac import load_checkpoint, parameter_sha256

                agent, _ = load_checkpoint(blob)
                if parameter_sha256(agent) != run["parameter_sha256"]["final"]:
                    problems.append(f"weights differ from the run's final hash: {entry['path']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--load", action="store_true", help="also rebuild each final agent and compare its weight hash")
    args = parser.parse_args()
    problems = verify(args.report, args.load)
    for problem in problems:
        print(problem)
    count = sum(len(run.get("checkpoints", [])) for run in json.loads(args.report.read_text(encoding="utf-8"))["runs"])
    print(f"{count} checkpoints, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
