"""Claim A's launch gate and scoring, driven by the pins in its preregistration. Prints PASS or FAIL lines and the verdict, never a map value.

    python experiments/claim_a_gate.py pins --prereg preregistrations/<file>.yaml
    python experiments/claim_a_gate.py after-run1 runs/phase/<run 1 report>.json --prereg preregistrations/<file>.yaml
    python experiments/claim_a_gate.py score runs/phase/<report>.json runs/phase/<report>.json --prereg preregistrations/<file>.yaml

`pins` recomputes every quantity the preregistration pins from the files it names (the dev
map's SHA-256, the content hashes of the theory-free null and of the held-out predictions,
the episode count, and the SHA-256 of every script in its code_sha256 block) and prints them,
so the preregistration can be checked line by line before it is frozen and again afterwards.
`after-run1` is the gate the held-out launches wait for: the preregistration must be
committed and unmodified; run 1 must be a dev-tier report from a clean commit whose tree
holds this preregistration byte for byte (so neither 26 November map can stand in for it);
it must reproduce the pinned map bitwise under tools/compare_phase_reports.py with only the
launch arguments `out` and `workers` skipped; the null rebuilt from it must have the pinned
content hash; and the episode count recomputed on it must be the pinned n. `score` runs the
preregistered scorer command on the two held-out reports, which are matched to the held-out
settings by their recorded instance, not by position, and must each be a complete, finite,
test-tier report launched under the committed preregistration that names its setting (its
recorded prereg_sha256 equals that file's committed blob). It writes the scorer's output and
the secondary null-as-theory and per-setting scores under runs/claim_a/<slug>/ and prints
the primary verdict; if any report value, E, E_null or interval bound is not finite, it
prints NOT COMPLETED and no verdict, except that lines the predictions leave out make E
infinite and the verdict fail, as the preregistration's severity clause says. Every file it
writes is new; nothing under runs/ is rewritten.

Pins are read from `key: value` lines of the preregistration: dev_map, dev_map_sha256,
heldout_settings, heldout_settings_sha256, predictions, predictions_sha256,
predictions_heldout_sha256, null_sha256, episodes, resamples, seed, sign_beyond_crossing and
setting (kappa/delta of the launch the file licenses); the code hashes come from the folded
`code_sha256: >` block, one `<path> <sha256>;` per line. A content hash is the SHA-256 of
json.dumps(object, indent=1, sort_keys=True) plus a newline, so it does not depend on the
platform's line endings.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from claim_a_episode_count import episode_count
from claim_a_filter_predictions import filter_predictions
from null_phase_predictions import load_settings, null_predictions

ROOT = Path(__file__).resolve().parents[1]
PINS = ("dev_map", "dev_map_sha256", "heldout_settings", "heldout_settings_sha256", "predictions", "predictions_sha256",
        "predictions_heldout_sha256", "null_sha256", "episodes", "resamples", "seed", "sign_beyond_crossing", "setting")  # fmt: skip
EXPECTED_COMPARE = "PASS: 480 results, 96000 values compared, 0 mismatched, 0 missing or malformed fields"
CELLS = 480  # 80 cells x 6 policies in one held-out report


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


def read_code_pins(prereg: Path) -> dict[str, str]:
    """path -> SHA-256 from the folded code_sha256 block."""
    text = prereg.read_text(encoding="utf-8")
    match = re.search(r"^code_sha256: >\n((?:[ \t]+\S.*\n)+)", text, re.M)
    if match is None:
        raise ValueError(f"{prereg.name} has no code_sha256 block")
    pins = dict(re.findall(r"^\s+(\S+) ([0-9a-f]{64});?\s*$", match.group(1), re.M))
    if not pins:
        raise ValueError(f"{prereg.name} has an empty code_sha256 block")
    return pins


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def committed_unmodified(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    return _git("ls-files", "--error-unmatch", rel).returncode == 0 and _git("diff", "--quiet", "HEAD", "--", rel).returncode == 0


def blob_sha256(commit: str, path: Path) -> str | None:
    """SHA-256 of `path` as committed in `commit`, or None if that commit does not hold it."""
    rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    shown = _git("show", f"{commit}:{rel}")
    return hashlib.sha256(shown.stdout).hexdigest() if shown.returncode == 0 else None


def computed_pins(dev_map: Path, settings: Path, predictions: Path, beyond: int) -> dict:
    """Everything the preregistration pins by value, recomputed from the files it names."""
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


def _check(name: str, got, want) -> bool:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}: {name} {'matches' if ok else 'differs from'} the preregistration")
    return ok


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f": {detail}" if detail else ""))
    return ok


def cmd_pins(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    computed = computed_pins(ROOT / pins["dev_map"], ROOT / pins["heldout_settings"], ROOT / pins["predictions"], beyond)
    ok = True
    for key, value in computed.items():
        print(f"{key}: {value}")
        ok &= _check(key, value, pins[key])
    for path, want in read_code_pins(args.prereg).items():
        ok &= _check(f"code_sha256 of {path}", file_hash(ROOT / path), want)
    return 0 if ok else 1


def cmd_after_run1(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    ok = _report("the preregistration is committed and unmodified", committed_unmodified(args.prereg))
    ok &= _check("dev_map_sha256 of the pinned map", file_hash(ROOT / pins["dev_map"]), pins["dev_map_sha256"])
    rerun = _load(args.run1)
    git = rerun.get("git") or {}
    ok &= _report("run 1 is a dev-tier report from a clean commit", rerun.get("args", {}).get("tier") == "dev" and git.get("dirty") is False,
                  f"tier {rerun.get('args', {}).get('tier')}, dirty {git.get('dirty')}")  # fmt: skip
    held_in_commit = blob_sha256(str(git.get("commit")), args.prereg) == file_hash(args.prereg) if git.get("commit") else False
    ok &= _report("run 1's commit holds this preregistration byte for byte", held_in_commit, f"commit {git.get('commit')}")
    ok &= _report("run 1 is not the pinned map itself", rerun.get("launch_id") != _load(ROOT / pins["dev_map"]).get("launch_id"))
    compare = subprocess.run([sys.executable, str(ROOT / "tools" / "compare_phase_reports.py"), str(ROOT / pins["dev_map"]), str(args.run1),
                              "--ignore-arg", "workers"], capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    first = (compare.stdout.strip().splitlines() or [""])[0]
    # the comparator appends "(launch arguments not compared: workers)" when an argument is ignored
    bitwise = compare.returncode == 0 and first.startswith(EXPECTED_COMPARE)
    ok &= _report("run 1 against the pinned map", bitwise, first or " ".join(compare.stderr.strip().splitlines()[-1:]))
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


def _setting_of(report: dict) -> str:
    inst = report["instance"]
    return f"{inst['kappa']}/{inst['delta']}"


def check_held_out_report(report: dict, path: Path, pins: dict[str, str]) -> list[str]:
    """Why a report is not a complete, finite, test-tier report of the preregistration that names its setting; empty if it is."""
    problems = []
    if report.get("args", {}).get("tier") != "test":
        problems.append("not a test-tier report")
    if len(report.get("evaluation_episode_seeds") or []) != int(pins["episodes"]):
        problems.append(f"not {pins['episodes']} evaluation episodes")
    results = report.get("results") or []
    if len(results) != CELLS:
        problems.append(f"{len(results)} results, not {CELLS}")
    values = [v for r in results for v in r.get("expected_failure_rate", [])]
    if len(values) != CELLS * int(pins["episodes"]) or not all(math.isfinite(v) for v in values):
        problems.append("a missing or non-finite value")
    named = report.get("args", {}).get("prereg")
    prereg = ROOT / named if named else None
    if prereg is None or prereg.resolve().parent != (ROOT / "preregistrations").resolve() or not committed_unmodified(prereg):
        problems.append(f"launched under {named!r}, not a committed preregistration")
    else:
        own = read_pins(prereg)
        if {k: own[k] for k in PINS if k != "setting"} != {k: pins[k] for k in PINS if k != "setting"}:
            problems.append(f"{prereg.name} pins a different rule")
        if own["setting"] != _setting_of(report):
            problems.append(f"{prereg.name} licenses setting {own['setting']}, the report is at {_setting_of(report)}")
        if report.get("prereg_sha256") != blob_sha256("HEAD", prereg):
            problems.append(f"the report's prereg_sha256 is not the committed {prereg.name}")
    return problems


def cmd_score(args) -> int:
    pins = read_pins(args.prereg)
    beyond = int(pins["sign_beyond_crossing"])
    slug = args.prereg.stem
    out = ROOT / "runs" / "claim_a" / slug
    held = load_settings(ROOT / pins["heldout_settings"])
    theory = _load(ROOT / pins["predictions"])
    filtered = filter_predictions(theory, held)
    null = null_predictions(_load(ROOT / pins["dev_map"]), theory["contrast"], beyond, held)
    ok = _report("the preregistration is committed and unmodified", committed_unmodified(args.prereg))
    ok &= _check("predictions_heldout_sha256", content_hash(filtered), pins["predictions_heldout_sha256"])
    ok &= _check("null_sha256", content_hash(null), pins["null_sha256"])
    ok &= _check("predictions_sha256", file_hash(ROOT / pins["predictions"]), pins["predictions_sha256"])
    reports = {}
    for path in args.reports:
        report = _load(path)
        problems = check_held_out_report(report, path, pins)
        ok &= _report(f"{path.name} is the held-out report at {_setting_of(report)}", not problems, "; ".join(problems))
        reports.setdefault(_setting_of(report), []).append(path)
    wanted = [f"{s['kappa']}/{s['delta']}" for s in held]
    ok &= _report("one report per held-out setting", sorted(reports) == sorted(wanted) and all(len(v) == 1 for v in reports.values()),
                  f"got {sorted(reports)}, need {sorted(wanted)}")  # fmt: skip
    if not ok:
        print("NOT COMPLETED: the inputs are not the preregistered, complete reports; no verdict")
        return 1
    ordered = [reports[w][0] for w in wanted]  # matched by instance, in the held-out file's order
    theory_path = _write(out / "predictions_heldout.json", filtered)
    null_path = _write(out / "null.json", null)
    primary = _score(theory_path, ordered, null_path, pins["resamples"], pins["seed"])
    _write(out / "primary.json", primary)
    secondary = {"null_as_theory": _score(null_path, ordered, null_path, pins["resamples"], pins["seed"])}
    for setting, report in zip(held, ordered):
        one = _write(out / f"predictions_{setting['kappa']}_{setting['delta']}.json", filter_predictions(theory, [setting]))
        one_null = _write(out / f"null_{setting['kappa']}_{setting['delta']}.json",
                          null_predictions(_load(ROOT / pins["dev_map"]), theory["contrast"], beyond, [setting]))  # fmt: skip
        secondary[f"setting_{setting['kappa']}_{setting['delta']}"] = _score(one, [report], one_null, pins["resamples"], pins["seed"])
    _write(out / "secondary.json", secondary)
    finite = all(np.isfinite([primary["E"], primary["E_null"], *primary["E_ci95"], *primary["E_minus_E_null_ci95"]]))
    if not finite and not primary["not_predicted"]:
        print(f"NOT COMPLETED: a non-finite E, E_null or bound; no verdict. written under {out.relative_to(ROOT).as_posix()}/")
        return 1
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
    score.add_argument("reports", type=Path, nargs=2, help="the two held-out reports, in either order")
    score.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args()
    try:
        return {"pins": cmd_pins, "after-run1": cmd_after_run1, "score": cmd_score}[args.command](args)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
