"""The pass/fail checks of the dev protocol before val (arena/2026-11-24-operations-outcome.md), computed from training reports.

    python experiments/check_dev_protocol.py step1 runs/train/<report>.json ...   # the six step-1 reports
    python experiments/check_dev_protocol.py step2 runs/train/<report>.json ...   # every dev-check report, step-1 temperature-0 ones included

Step 1 takes the launches at fixed temperature 0, one per instance and length (12 000,
24 000 and 48 000 decisions), and checks:
  (a) both agents of every seed end with identical weights (equal final hashes);
  (b) each (variant, seed, instance) has identical weights at initialisation and after
      the first update window in all three lengths;
  (c) convergence: per seed, g(L) = (uniform - agent) / (uniform - every-quarter
      replanner) on the run's own evaluation episodes, using OC-SAC if (a) holds and the
      mean of both agents otherwise. L* is the smaller of 12 000 and 24 000 for which, on
      every instance, the upper end of the 95% t-interval of the mean of g(2L) - g(L) over
      seeds is below 0.05 and no more than one fewer run learned at L than at 2L.
Step 2 checks each report of the dev check:
  fixed temperatures hold exactly; a learned temperature stays strictly inside its clamp in
  every window and at evaluation, ends within 0.1 of the reachable range from its target
  (last entropy window) and settles (log temperature varies by less than 1 nat over the
  last quarter of windows); and each agent's computed entropy floor lies within 10% of the
  range of the lowest entropy any of its runs reached on that instance. It also reports,
  without failing, whether the fixed grid must widen upward (mean class entropy at
  evaluation below 0.9 log M at temperature 3), which the owner applies.
Prints every check and exits 1 if any fails; numbers come only from the reports given.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from stats import t_quantile

REPLANNER = "oracle[replans every quarter]"


def _load(paths: list[Path]) -> list[dict]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path)
        reports.append(report)
    return reports


def _instance(report: dict) -> tuple[int, ...]:
    return tuple(report["class_structure"]["class_sizes"])


def _quarters(report: dict) -> int:
    return report["sac_config"]["oc-sac"]["total_quarters"]


def _runs(report: dict, variant: str) -> dict[int, dict]:
    return {r["seed"]: r for r in report["runs"] if r["variant"] == variant and r["sigma"] == 0.0 and "diverged" not in r}


def _gain_share(run: dict, report: dict) -> float:
    base = report["baselines"]["0.0"]
    uniform, replanner = np.mean(base["uniform"]["expected_failure_rate"]), np.mean(base[REPLANNER]["expected_failure_rate"])
    return float((uniform - np.mean(run["expected_failure_rate"])) / (uniform - replanner))


def step1(reports: list[dict]) -> tuple[list[str], dict]:
    lines, failed = [], False
    for report in reports:
        if any(report["sac_config"][v]["temperature"] != "fixed" or report["sac_config"][v]["alpha"] != 0.0 for v in ("oc-sac", "sac")):
            raise ValueError(f"{report['_path']} is not a fixed-temperature-0 launch")
    pairs = [(report, seed) for report in reports for seed in _runs(report, "oc-sac")]
    equal = [seed for report, seed in pairs if _runs(report, "sac").get(seed, {}).get("parameter_sha256", {}).get("final")
             == _runs(report, "oc-sac")[seed]["parameter_sha256"]["final"]]  # fmt: skip
    bitwise = len(equal) == len(pairs)
    lines.append(f"(a) bitwise A/A: {len(equal)}/{len(pairs)} pairs end with identical weights -> {'PASS' if bitwise else 'FAIL'}")
    failed |= not bitwise
    by_instance: dict[tuple, dict[int, dict]] = {}
    for report in reports:
        by_instance.setdefault(_instance(report), {})[_quarters(report)] = report
    for sizes, by_length in sorted(by_instance.items()):
        if sorted(by_length) != [12_000, 24_000, 48_000]:
            raise ValueError(f"instance {list(sizes)} needs reports at 12000, 24000 and 48000 decisions, got {sorted(by_length)}")
        mismatched = 0
        for variant in ("oc-sac", "sac"):
            for seed in _runs(by_length[12_000], variant):
                hashes = {(r["parameter_sha256"]["initial"], r["parameter_sha256"]["after_first_window"])
                          for r in (_runs(by_length[q], variant).get(seed) for q in by_length) if r}  # fmt: skip
                mismatched += len(hashes) != 1
        lines.append(f"(b) instance {list(sizes)}: {mismatched} (variant, seed) differ across lengths at initialisation or first window -> {'PASS' if not mismatched else 'FAIL'}")
        failed |= mismatched > 0
    chosen = None
    for length in (12_000, 24_000):
        ok = True
        for sizes, by_length in sorted(by_instance.items()):
            short, long = by_length[length], by_length[2 * length]
            seeds = sorted(set(_runs(short, "oc-sac")) & set(_runs(long, "oc-sac")))
            variants = ("oc-sac",) if bitwise else ("oc-sac", "sac")
            d = [np.mean([_gain_share(_runs(long, v)[s], long) - _gain_share(_runs(short, v)[s], short) for v in variants]) for s in seeds]
            upper = float(np.mean(d) + t_quantile(0.975, len(d) - 1) * np.std(d, ddof=1) / math.sqrt(len(d)))
            learned_short = short["summary"]["0.0"]["oc-sac"]["runs_learned"]
            learned_long = long["summary"]["0.0"]["oc-sac"]["runs_learned"]
            passes = upper < 0.05 and learned_short >= learned_long - 1
            ok &= passes
            lines.append(f"(c) instance {list(sizes)}, {length} -> {2 * length}: upper 95% bound of the gain-share change {upper:.4f} (limit 0.05), "
                         f"runs learned {learned_short} vs {learned_long} -> {'PASS' if passes else 'FAIL'}")  # fmt: skip
        if ok:
            chosen = length
            break
    if chosen is None:
        lines.append("(c) no length up to 24000 passes on every instance: training is not shown to converge by 48000. STOP for the owner.")
        failed = True
    else:
        lines.append(f"(c) training length L* = {chosen}")
    return lines, {"failed": failed, "bitwise": bitwise, "length": chosen}


def _own_entropy(run: dict, variant: str) -> float:
    spent = run["policy_entropy"]
    return spent["action"] if variant == "sac" else spent["action"] - spent["within_class"]  # class entropy with tied drugs


def step2(reports: list[dict]) -> tuple[list[str], dict]:
    lines, failed, widen = [], False, False
    lowest: dict[tuple, float] = {}
    for report in reports:
        for variant in ("oc-sac", "sac"):
            cfg = report["sac_config"][variant]
            low, high = cfg["log_alpha_range"]
            for seed, run in sorted(_runs(report, variant).items()):
                key = (_instance(report), variant)
                lowest[key] = min(lowest.get(key, math.inf), _own_entropy(run, variant))
                label = f"{Path(report['_path']).name} {variant} seed {seed}"
                if cfg["temperature"] == "fixed":
                    ok = all(a == cfg["alpha"] for a in run["alpha_path"]) and run["alpha_at_evaluation"] == cfg["alpha"]
                    lines.append(f"{label}: fixed temperature {cfg['alpha']} held -> {'PASS' if ok else 'FAIL'}")
                    failed |= not ok
                    continue
                logs = [math.log(a) for a in run["alpha_path"]] + [math.log(run["alpha_at_evaluation"])]
                inside = all(low + 1e-6 < x < high - 1e-6 for x in logs)
                span = run["entropy_range"]["maximum"] - run["entropy_range"]["floor"]
                tracking = abs(run["entropy_path"][-1] - run["entropy_range"]["target"]) / span
                tail = [math.log(a) for a in run["alpha_path"][-max(1, len(run["alpha_path"]) // 4):]]
                settling = max(tail) - min(tail)
                ok = inside and tracking < 0.1 and settling < 1.0
                lines.append(f"{label}: off the clamp {inside}, distance from target {tracking:.3f} of the range (limit 0.1), "
                             f"log-temperature range over the last quarter {settling:.3f} nats (limit 1) -> {'PASS' if ok else 'FAIL'}")  # fmt: skip
                failed |= not ok
            if cfg["temperature"] == "fixed" and cfg["alpha"] == 3.0:
                class_entropy = np.mean([r["policy_entropy"]["action"] - r["policy_entropy"]["within_class"] for r in _runs(report, variant).values()])
                below = class_entropy < 0.9 * math.log(len(report["class_structure"]["class_sizes"]))
                widen |= below
                lines.append(f"widening: {variant} on {list(_instance(report))} at alpha 3, mean class entropy {class_entropy:.3f} "
                             f"vs 0.9 log M -> {'widen the grid upward (owner)' if below else 'no widening'}")  # fmt: skip
    for (sizes, variant), reached in sorted(lowest.items()):
        report = next(r for r in reports if _instance(r) == sizes)
        rng = next(iter(_runs(report, variant).values()))["entropy_range"]
        gap = (reached - rng["floor"]) / (rng["maximum"] - rng["floor"])
        ok = gap <= 0.1
        lines.append(f"floor: {variant} on {list(sizes)}, lowest entropy reached sits {gap:.3f} of the range above the computed floor (limit 0.1) -> {'PASS' if ok else 'FAIL'}")
        failed |= not ok
    return lines, {"failed": failed, "widen_upward": widen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("step1", "step2"))
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        lines, result = (step1 if args.step == "step1" else step2)(_load(args.reports))
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print("\n".join(lines))
    print(json.dumps(result))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
