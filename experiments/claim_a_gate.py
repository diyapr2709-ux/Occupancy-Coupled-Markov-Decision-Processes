"""Claim A's launch gate and scoring, driven by the pins in its preregistration. Prints PASS or FAIL lines and the verdict, never a map value.

    python experiments/claim_a_gate.py pins --prereg preregistrations/<file>.yaml
    python experiments/claim_a_gate.py after-run1 runs/phase/<run 1 report>.json --prereg preregistrations/<file>.yaml
    python experiments/claim_a_gate.py score runs/phase/<kappa report>.json runs/phase/<delta report>.json --prereg preregistrations/<file>.yaml

`pins` recomputes every quantity the preregistration pins from the committed files (the dev
map's SHA-256, the content hashes of the theory-free null and of the held-out predictions,
the episode count) and prints them, so the preregistration can be checked line by line
before it is frozen and again afterwards. `after-run1` is the gate the held-out launches
wait for: the dev-map rerun must reproduce the pinned map bitwise under
tools/compare_phase_reports.py with only `workers` ignored, the null rebuilt from it must
have the pinned content hash, and the episode count recomputed on it must be the pinned n.
`score` runs the preregistered scorer command on the two held-out reports, with the null
built from the pinned map, writes the scorer's output and the secondary null-as-theory and
per-setting scores under runs/claim_a/, and prints the primary verdict. Every file it writes
is new; nothing under runs/ is rewritten.

Pins are read from `key: value` lines of the preregistration. The keys used are dev_map,
dev_map_sha256, heldout_settings, heldout_settings_sha256, predictions, predictions_sha256,
predictions_heldout_sha256, null_sha256, episodes, resamples, seed and sign_beyond_crossing.
A content hash is the SHA-256 of json.dumps(object, indent=1, sort_keys=True) plus a newline,
so it does not depend on the platform's line endings.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from claim_a_episode_count import episode_count
from claim_a_filter_predictions import filter_predictions
from null_phase_predictions import load_settings, null_predictions

ROOT = Path(__file__).resolve().parents[1]
PINS = ("dev_map", "dev_map_sha256", "heldout_settings", "heldout_settings_sha256", "predictions", "predictions_sha256",
        "predictions_heldout_sha256", "null_sha256", "episodes", "resamples", "seed", "sign_beyond_crossing")  # fmt: skip
EXPECTED_COMPARE = "PASS: 480 results, 96000 values compared, 0 mismatched, 0 missing or malformed fields"


def content_hash(obj) -> str:
    return hashlib.sha256((json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pins(prereg: Path) -> dict[str, str]:
    pins = {}
    for line in prereg.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([a-z0-9_]+):\s*(\S+)\s*$", line)
        if match and match.group(1) in PINS:
            if match.group(1) in pins:
                raise ValueError(f"{prereg.name} pins {match.group(1)} twice")
            pins[match.group(1)] = match.group(2)
    missing = [k for k in PINS if k not in pins]
    if missing:
        raise ValueError(f"{prereg.name} lacks the pins {missing}")
    return pins


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def computed_pins(dev_map: Path, settings: Path, predictions: Path, beyond: int) -> dict:
    """Everything the preregistration pins, recomputed from the files it names."""
    report = _load(dev_map)
    held = load_settings(settings)
    theory = _load(predictions)
    null = null_predictions(report, theory["contrast"], beyond, held)
    return {
        "dev_map_sha256": file_hash(dev_map),
        "heldout_settings_sha256": file_hash(settings),
        "predictions_sha256": file_hash(predictions),
        "predictions_heldout_sha256": content_hash(filter_predictions(theory, held)),
        "null_sha256": content_hash(null),
        "episodes": str(episode_count(report)),
        "sign_beyond_crossing": str(theory["sign_beyond_crossing"]),
    }


def _check(name: str, got: str, want: str) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}: {name} {'matches' if ok else 'differs from'} the preregistration")
    return ok


def cmd_pins(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    computed = computed_pins(ROOT / pins["dev_map"], ROOT / pins["heldout_settings"], ROOT / pins["predictions"], beyond)
    ok = True
    for key, value in computed.items():
        print(f"{key}: {value}")
        ok &= _check(key, value, pins[key])
    return 0 if ok else 1


def cmd_after_run1(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    ok = _check("dev_map_sha256 of the pinned map", file_hash(ROOT / pins["dev_map"]), pins["dev_map_sha256"])
    compare = subprocess.run([sys.executable, str(ROOT / "tools" / "compare_phase_reports.py"), str(ROOT / pins["dev_map"]), str(args.run1),
                              "--ignore-arg", "workers"], capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    first = (compare.stdout.strip().splitlines() or [""])[0]
    bitwise = compare.returncode == 0 and first == EXPECTED_COMPARE
    print(f"{'PASS' if bitwise else 'FAIL'}: run 1 against the pinned map: {first or compare.stderr.strip().splitlines()[-1:]}")
    ok &= bitwise
    rerun = _load(args.run1)
    held = load_settings(ROOT / pins["heldout_settings"])
    theory = _load(ROOT / pins["predictions"])
    ok &= _check("null_sha256 rebuilt from run 1", content_hash(null_predictions(rerun, theory["contrast"], beyond, held)), pins["null_sha256"])
    ok &= _check("episodes recomputed on run 1", str(episode_count(rerun)), pins["episodes"])
    print("GATE PASS: the held-out launches may go ahead" if ok else "GATE FAIL: stop; nothing launches")
    return 0 if ok else 1


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; nothing under runs/ is rewritten")
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def _score(predictions: Path, reports: list[Path], null: Path | None, resamples: str, seed: str) -> dict:
    cmd = [sys.executable, str(ROOT / "experiments" / "score_phase_predictions.py"), str(predictions), *map(str, reports),
           "--resamples", resamples, "--seed", seed]  # fmt: skip
    if null is not None:
        cmd += ["--null", str(null)]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if done.returncode != 0:
        raise ValueError(f"the scorer refused: {done.stderr.strip().splitlines()[-1] if done.stderr.strip() else done.returncode}")
    return json.loads(done.stdout)


def cmd_score(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    slug = args.prereg.stem
    out = ROOT / "runs" / "claim_a" / slug
    held = load_settings(ROOT / pins["heldout_settings"])
    theory = _load(ROOT / pins["predictions"])
    filtered = filter_predictions(theory, held)
    null = null_predictions(_load(ROOT / pins["dev_map"]), theory["contrast"], beyond, held)
    ok = _check("predictions_heldout_sha256", content_hash(filtered), pins["predictions_heldout_sha256"])
    ok &= _check("null_sha256", content_hash(null), pins["null_sha256"])
    ok &= _check("predictions_sha256", file_hash(ROOT / pins["predictions"]), pins["predictions_sha256"])
    for report in args.reports:
        got = str(len(_load(report)["evaluation_episode_seeds"]))
        ok &= _check(f"episodes of {report.name}", got, pins["episodes"])
    if not ok:
        print("SCORE FAIL: the inputs are not the preregistered ones")
        return 1
    theory_path = _write(out / "predictions_heldout.json", filtered)
    null_path = _write(out / "null.json", null)
    primary = _score(theory_path, args.reports, null_path, pins["resamples"], pins["seed"])
    _write(out / "primary.json", primary)
    secondary = {"null_as_theory": _score(null_path, args.reports, null_path, pins["resamples"], pins["seed"])}
    for setting, report in zip(held, args.reports):
        one = _write(out / f"predictions_{setting['kappa']}_{setting['delta']}.json", filter_predictions(theory, [setting]))
        one_null = _write(out / f"null_{setting['kappa']}_{setting['delta']}.json",
                          null_predictions(_load(ROOT / pins["dev_map"]), theory["contrast"], beyond, [setting]))  # fmt: skip
        secondary[f"setting_{setting['kappa']}_{setting['delta']}"] = _score(one, [report], one_null, pins["resamples"], pins["seed"])
    _write(out / "secondary.json", secondary)
    print(f"primary verdict_vs_null: {primary['verdict_vs_null']}  E_minus_E_null_ci95: {primary['E_minus_E_null_ci95']}  "
          f"E: {primary['E']}  E_ci95: {primary['E_ci95']}  E_null: {primary['E_null']}  not_predicted: {len(primary['not_predicted'])}")
    print(f"written under {out.relative_to(ROOT).as_posix()}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pins = sub.add_parser("pins")
    pins.add_argument("--prereg", type=Path, required=True)
    after = sub.add_parser("after-run1")
    after.add_argument("run1", type=Path)
    after.add_argument("--prereg", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("reports", type=Path, nargs=2, help="the kappa-0.2 report, then the delta-0.1 report")
    score.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args()
    try:
        return {"pins": cmd_pins, "after-run1": cmd_after_run1, "score": cmd_score}[args.command](args)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
