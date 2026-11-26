"""The pass/fail checks of the dev protocol before val, computed from training reports.

    python experiments/check_dev_protocol.py step1 --planner runs/planner/<report>.json runs/train/<report>.json ...   # the six step-1 reports
    python experiments/check_dev_protocol.py step2 runs/train/<report>.json ...   # every dev-check report, step-1 temperature-0 ones included

Step 1 (arena/2026-11-25-step1-outcome.md §3-4, with the owner's six-launch design of
2026-11-26: c-given only, dev seeds 0-4, one report per instance and length, no second
seed stage and no 96 000 extension) takes six launches from one clean commit at fixed
temperature 0, noise 0, default delay and lam and one agent configuration: class sizes
(2, 2) and (8, 5, 4, 2, 1), each at 12 000, 24 000 and 48 000 decisions. It decides, in
this order:
  1. Reference: the planner-restart report at (8, 5, 4, 2, 1), from the same code, at
     the planner's own steps and random starts and at least 5 planner seeds, must plan a
     gain over uniform and put the spread over planner seeds below 0.02 of it; otherwise
     REFERENCE NOT ESTABLISHED. It measures the planned-once prediction from the nominal
     start, a proxy for the replanner's restart error on the evaluation episodes.
  2. DEFECT, a stop with no fallback, on any of: (a) the two agents of a seed differ in a
     weight hash (the first differing stage is named) or in whether, where and how they
     diverged; (b) a (variant, seed, instance) differs across lengths at initialisation or
     after the first window, or diverges in one length without diverging at the same
     decision in every length that reaches it; a pair not trained in one process one job
     after the other; a replanner that does not fail less than uniform mixing; jobs with
     different platform fingerprints, a fingerprint not matching its launch, or a canary
     that is not the full one; baselines or evaluation episodes that differ between the
     reports of an instance; different seed sets; different commits or code diffs.
  3. Convergence. Per seed, g = (U - A) / (U - R), with U uniform mixing and R the
     every-quarter replanner on the report's own episodes, from OC-SAC (equal to SAC once
     (a) holds). A diverged run gets g = min(0, the lowest g of the other runs of its
     report) and has not learned; learned means g > 0.5. With n seeds, a(n) = max(1,
     floor(n / 10)) and half = ceil(n / 2). For a step L -> L', d = g(L') - g(L) per seed
     and I its two-sided 95% t-interval: EQUIV if I lies inside (-0.05, 0.05), RISING if
     its lower end is at least 0.05, DECLINING if its upper end is at most -0.05, and
     INCONCLUSIVE otherwise. A step passes when it is EQUIV, the learned counts differ by
     at most a(n), at least half the runs learned at L, and at most a(n) runs diverged at
     each length. An instance converges at 12 000 when 12->24, 24->48 and 12->48 all
     pass, otherwise at 24 000 when 24->48 passes, and otherwise not at all. L* is the
     larger of the two instances' lengths.
An instance that does not converge stops the work for the owner, with the reason read
from its 24->48 step, first that applies: DIVERGING (more than a(n) diverged runs),
DECLINING (an interval or a learned count that falls by more than a(n)),
FLAT_BELOW_MAJORITY (equivalent, fewer than half learned, no count rise), RISING (an
interval or a learned count that rises), UNDECIDED (inconclusive). No default length is
ever returned. L* is qualified only for the condition the six launches ran in.

Step 2 checks each report of the dev check:
  fixed temperatures hold exactly; a learned temperature stays strictly inside its clamp in
  every window and at evaluation, ends within 0.1 of the reachable range from its target
  (last entropy window) and settles (log temperature varies by less than 1 nat over the
  last quarter of windows); and each agent's computed entropy floor lies within 10% of the
  range of the lowest entropy any of its runs reached on that instance and condition. It
  also reports, without failing, whether the fixed grid must widen upward (mean class
  entropy at evaluation below 0.9 log M at temperature 3), which the owner applies.
Prints every check and exits 0 if all pass and 1 otherwise, malformed input included;
numbers come only from the reports given.
"""

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

from stats import t_quantile
from train_agent import CONDITIONS, REPLANNER, REPORT_SCHEMA, VARIANTS, pairing_defects

INSTANCES = ((2, 2), (8, 5, 4, 2, 1))
LENGTHS = (12_000, 24_000, 48_000)
DELTA = 0.05  # equivalence margin on the change in g, from the operations outcome
REFERENCE_INSTANCE = (8, 5, 4, 2, 1)
SEEDS = [0, 1, 2, 3, 4]  # the owner's step-1 seeds
CONDITION = "c-given"  # the owner's step-1 condition
MIN_PLANNER_SEEDS = 5
SPREAD_LIMIT = 0.02  # planner-restart spread, as a share of the gain over uniform
BASELINE_TOLERANCE = 1e-12
HASH_STAGES = ("initial", "after_first_window", "final")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(1)


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


def _name(report: dict) -> str:
    return Path(report["_path"]).name


def _runs(report: dict, variant: str) -> dict[int, dict]:
    """The completed (not diverged) runs of one variant without noise, by seed."""
    return {r["seed"]: r for r in report["runs"] if r["variant"] == variant and r["sigma"] == 0.0 and "diverged" not in r}


def _all_runs(report: dict, variant: str) -> dict[int, dict]:
    """Every run of one variant without noise, diverged ones included, by seed."""
    return {r["seed"]: r for r in report["runs"] if r["variant"] == variant and r["sigma"] == 0.0}


def _step1_cells(reports: list[dict]) -> dict[tuple, dict]:
    """(instance, length) -> its one report, after checking the six reports are the step-1 design."""
    cells = {}
    for report in reports:
        if report.get("report_schema") != REPORT_SCHEMA:
            raise ValueError(f"{_name(report)} has report schema {report.get('report_schema')}, not {REPORT_SCHEMA}")
        if any(report["sac_config"][v]["temperature"] != "fixed" or report["sac_config"][v]["alpha"] != 0.0 for v in VARIANTS):
            raise ValueError(f"{_name(report)} is not a fixed-temperature-0 launch")
        if report["args"]["tier"] != "dev" or report["args"]["sigmas"] != [0.0]:
            raise ValueError(f"{_name(report)} must be a dev-tier launch at noise 0 only")
        if report["args"]["seeds"] != SEEDS or report["condition"] != CONDITION:
            raise ValueError(f"{_name(report)} must be a {CONDITION} launch on dev seeds {SEEDS}")
        if report["args"]["delay"] is not None or report["args"]["lam"] is not None:
            raise ValueError(f"{_name(report)} must use the default delay and lam")
        if report["git"]["dirty"] is not False:
            raise ValueError(f"{_name(report)} ran from uncommitted code")
        key = (_instance(report), _quarters(report))
        if key in cells:
            raise ValueError(f"two reports for instance {list(key[0])} at {key[1]} decisions: {_name(cells[key])} and {_name(report)}")
        cells[key] = report
    expected = {(sizes, length) for sizes in INSTANCES for length in LENGTHS}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(f"step 1 needs exactly one report per instance in {[list(s) for s in INSTANCES]} and length in {list(LENGTHS)}; "
                         f"missing {missing}, unexpected {extra}")  # fmt: skip
    def config(report: dict) -> list:
        return [{k: v for k, v in report["sac_config"][variant].items() if k not in ("total_quarters", "alpha_lr", "entropy")} for variant in VARIANTS]

    first = reports[0]
    for report in reports[1:]:
        if config(report) != config(first) or report["args"]["set"] != first["args"]["set"]:
            raise ValueError(f"{_name(report)} and {_name(first)} differ in agent configuration beyond the run length")
    for report in reports:
        if report["sac_config"]["sac"]["total_quarters"] != _quarters(report):
            raise ValueError(f"{_name(report)}: the two agents trained for different lengths")
    return cells


def _reference(planner: dict, git: dict) -> tuple[bool, str]:
    """Whether the planner-restart report establishes the replanner as the reference at REFERENCE_INSTANCE.

    The report must come from the step-1 code, compare at least MIN_PLANNER_SEEDS planner
    seeds and plan a gain over uniform. It measures the spread of the planned-once
    prediction from the nominal start, a proxy for the replanner's restart error on the
    evaluation episodes, which no report measures.
    """
    from ocmdp.baselines.oracle import FullInformationOracle

    defaults = inspect.signature(FullInformationOracle.__init__).parameters
    args = planner["args"]
    if args.get("class_sizes") != list(REFERENCE_INSTANCE):
        raise ValueError(f"the planner report must be at class sizes {list(REFERENCE_INSTANCE)}, got {args.get('class_sizes')}")
    if args["steps"] != defaults["steps"].default or args["random_starts"] != defaults["n_random"].default:
        raise ValueError(f"the planner report must use the planner's own {defaults['steps'].default} steps and "
                         f"{defaults['n_random'].default} random starts")  # fmt: skip
    if args["planner_seeds"] < MIN_PLANNER_SEEDS:
        raise ValueError(f"the planner report must compare at least {MIN_PLANNER_SEEDS} planner seeds, got {args['planner_seeds']}")
    if (planner["git"]["commit"], planner["git"]["code_diff_sha256"]) != (git["commit"], git["code_diff_sha256"]):
        raise ValueError("the planner report must come from the commit and code the step-1 reports ran")
    if not planner["uniform"] > min(planner["planned_failure_rate"]):
        raise ValueError("the planner report plans no gain over uniform mixing")
    share = planner["spread_share_of_gain_over_uniform"]
    ok = 0.0 <= share < SPREAD_LIMIT
    return ok, f"reference: planner-restart spread at {list(REFERENCE_INSTANCE)} is {share:.4f} of the gain over uniform (limit {SPREAD_LIMIT}) -> {'PASS' if ok else 'REFERENCE NOT ESTABLISHED'}"


def _first_difference(a: dict, b: dict) -> str | None:
    """The first weight-hash stage at which two runs differ, or how their divergence differs; None if they agree."""
    if "diverged" in a or "diverged" in b:
        same = (a.get("diverged"), a.get("diverged_at_decision")) == (b.get("diverged"), b.get("diverged_at_decision"))
        return None if same else "divergence (one run diverged, or at another decision or with another message)"
    return next((stage for stage in HASH_STAGES if a["parameter_sha256"].get(stage) != b["parameter_sha256"].get(stage)), None)


def _defects(cells: dict[tuple, dict]) -> list[str]:
    defects = []
    reports = list(cells.values())
    seeds = {f"{_name(r)} {v}": sorted(_all_runs(r, v)) for r in reports for v in VARIANTS}
    if len({tuple(s) for s in seeds.values()}) != 1:
        defects.append(f"seed sets differ between reports or variants: {seeds}")
    for (sizes, length), report in sorted(cells.items()):
        base = report["baselines"]["0.0"]
        if not np.mean(base["uniform"]["expected_failure_rate"]) > np.mean(base[REPLANNER]["expected_failure_rate"]):
            defects.append(f"{_name(report)}: the replanner does not fail less than uniform mixing, so g has no scale")
        oc, sac = _all_runs(report, "oc-sac"), _all_runs(report, "sac")
        for seed in sorted(set(oc) & set(sac)):
            stage = _first_difference(oc[seed], sac[seed])
            if stage:
                defects.append(f"(a) {list(sizes)} at {length}, seed {seed}: OC-SAC and SAC first differ at {stage}")
        defects += [f"pairing in {_name(report)}: {d}" for d in sorted(set(pairing_defects(report["runs"])) | set(report["pairing_defects"]))]
    for sizes in INSTANCES:
        for variant in VARIANTS:
            # at a fixed temperature a shorter run is a prefix of a longer one: a divergence before L appears, at the same
            # decision, in every longer run, and none appears in a run of length L at a decision beyond L
            for seed in SEEDS:
                runs = {length: _all_runs(cells[(sizes, length)], variant).get(seed) for length in LENGTHS}
                diverged = {length: run["diverged_at_decision"] for length, run in runs.items() if run and "diverged" in run}
                for length in LENGTHS:
                    earlier = [d for d in diverged.values() if d <= length]
                    if earlier and diverged.get(length) != min(earlier):
                        defects.append(f"(b) {list(sizes)} {variant} seed {seed}: diverged at decision {min(earlier)} in one length "
                                       f"but not at the same decision in the {length} run")  # fmt: skip
            by_seed: dict[int, set] = {}
            for length in LENGTHS:
                for seed, run in _runs(cells[(sizes, length)], variant).items():
                    by_seed.setdefault(seed, set()).add((run["parameter_sha256"]["initial"], run["parameter_sha256"]["after_first_window"]))
            for seed, hashes in sorted(by_seed.items()):
                if len(hashes) != 1:
                    defects.append(f"(b) {list(sizes)} {variant} seed {seed}: initial or first-window weights differ across lengths")
        first = cells[(sizes, LENGTHS[0])]
        for length in LENGTHS[1:]:
            other = cells[(sizes, length)]
            for name in ("uniform", REPLANNER):
                a = np.asarray(first["baselines"]["0.0"][name]["expected_failure_rate"])
                b = np.asarray(other["baselines"]["0.0"][name]["expected_failure_rate"])
                if a.shape != b.shape or np.max(np.abs(a - b)) > BASELINE_TOLERANCE:
                    defects.append(f"baseline {name} on {list(sizes)} differs between {_name(first)} and {_name(other)}")
    if len({tuple(r["evaluation_episode_seeds"]) for r in reports}) != 1:
        defects.append("evaluation episodes differ between reports")
    if len({(r["git"]["commit"], r["git"]["code_diff_sha256"]) for r in reports}) != 1:
        defects.append("the reports ran from different commits or code diffs")
    fingerprints = {r["host"]["fingerprint"]["platform_fingerprint"] for report in reports for r in report["runs"]}
    fingerprints |= {report["platform_fingerprint"] for report in reports}
    if len(fingerprints) != 1:
        defects.append(f"jobs and launches carry {len(fingerprints)} platform fingerprints, not one")
    if any(not r["fingerprint_matches_launch"] for report in reports for r in report["runs"]):
        defects.append("a job's platform fingerprint does not match its launch's")
    if any(r["host"]["fingerprint"]["arithmetic_canary"]["size"] != "full" for report in reports for r in report["runs"]):
        defects.append("a job ran with a reduced arithmetic canary")
    return defects


def _gain_shares(report: dict) -> tuple[dict[int, float], set[int]]:
    """Per seed g = (U - A) / (U - R) from OC-SAC, a diverged run given min(0, the lowest other g); and the diverged seeds."""
    base = report["baselines"]["0.0"]
    uniform, replanner = np.mean(base["uniform"]["expected_failure_rate"]), np.mean(base[REPLANNER]["expected_failure_rate"])
    runs = _all_runs(report, "oc-sac")
    g = {seed: float((uniform - np.mean(r["expected_failure_rate"])) / (uniform - replanner)) for seed, r in runs.items() if "diverged" not in r}
    diverged = {seed for seed, r in runs.items() if "diverged" in r}
    imputed = min(0.0, min(g.values(), default=0.0))
    return {seed: g.get(seed, imputed) for seed in sorted(runs)}, diverged


def _classify(d: list[float]) -> tuple[str, float, float]:
    """The class of a mean change in g, with its two-sided 95% t-interval."""
    n = len(d)
    mean = float(np.mean(d))
    half = t_quantile(0.975, n - 1) * float(np.std(d, ddof=1)) / math.sqrt(n)
    lo, hi = mean - half, mean + half
    if -DELTA < lo and hi < DELTA:
        label = "EQUIV"
    elif lo >= DELTA:
        label = "RISING"
    elif hi <= -DELTA:
        label = "DECLINING"
    else:
        label = "INCONCLUSIVE"
    return label, lo, hi


def _step(short: dict, long: dict) -> dict:
    """Whether one step L -> L' passes, and if not, why: the first of DIVERGING, DECLINING, FLAT_BELOW_MAJORITY, RISING, UNDECIDED."""
    (g_short, div_short), (g_long, div_long) = _gain_shares(short), _gain_shares(long)
    seeds = sorted(g_short)
    n = len(seeds)
    allowed, half = max(1, n // 10), math.ceil(n / 2)
    k_short = sum(g_short[s] > 0.5 for s in seeds)
    k_long = sum(g_long[s] > 0.5 for s in seeds)
    d = [g_long[s] - g_short[s] for s in seeds]
    label, lo, hi = _classify(d)
    passes = label == "EQUIV" and abs(k_long - k_short) <= allowed and k_short >= half and max(len(div_short), len(div_long)) <= allowed
    if passes:
        reason = None
    elif max(len(div_short), len(div_long)) > allowed:
        reason = "DIVERGING"
    elif label == "DECLINING" or k_long < k_short - allowed:
        reason = "DECLINING"
    elif label == "EQUIV" and k_short < half and k_long <= k_short + allowed:
        reason = "FLAT_BELOW_MAJORITY"
    elif label == "RISING" or k_long > k_short + allowed:
        reason = "RISING"
    else:
        reason = "UNDECIDED"
    return {"n": n, "mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)), "interval": [lo, hi], "class": label, "learned": [k_short, k_long], "diverged": [len(div_short), len(div_long)],
            "passes": passes, "reason": reason}  # fmt: skip


def step1(reports: list[dict], planner: dict) -> tuple[list[str], dict]:
    cells = _step1_cells(reports)
    condition = reports[0]["condition"]
    lines = []
    ok, line = _reference(planner, reports[0]["git"])
    lines.append(line)
    if not ok:
        return lines, {"verdict": "REFERENCE NOT ESTABLISHED", "length": None, "failed": True, "condition": condition}
    defects = _defects(cells)
    lines += [f"DEFECT {d}" for d in defects] or ["defects: none ((a), (b), pairing, fingerprints, baselines, seeds, commits)"]
    if defects:
        return lines, {"verdict": "DEFECT", "length": None, "failed": True, "condition": condition, "defects": defects}
    lengths, detail = {}, {}
    for sizes in INSTANCES:
        steps = {f"{a}->{b}": _step(cells[(sizes, a)], cells[(sizes, b)]) for a, b in ((12_000, 24_000), (24_000, 48_000), (12_000, 48_000))}
        for name, s in steps.items():
            lines.append(f"{list(sizes)} {name}: mean d {s['mean']:+.4f}, SD {s['sd']:.4f}, 95% interval [{s['interval'][0]:+.4f}, {s['interval'][1]:+.4f}] "
                         f"{s['class']}, learned {s['learned'][0]} -> "
                         f"{s['learned'][1]} of {s['n']}, diverged {s['diverged'][0]} and {s['diverged'][1]} -> {'PASS' if s['passes'] else 'FAIL'}")  # fmt: skip
        if all(steps[k]["passes"] for k in ("12000->24000", "24000->48000", "12000->48000")):
            lengths[sizes] = 12_000
        elif steps["24000->48000"]["passes"]:
            lengths[sizes] = 24_000
        else:
            lengths[sizes] = None
        detail[str(list(sizes))] = {"length": lengths[sizes], "steps": steps}
    failing = {sizes: detail[str(list(sizes))]["steps"]["24000->48000"]["reason"] for sizes, length in lengths.items() if length is None}
    if failing:
        for sizes, reason in failing.items():
            lines.append(f"{list(sizes)} does not converge by 48000: {reason}. STOP for the owner.")
        return lines, {"verdict": "STOP", "length": None, "failed": True, "condition": condition, "reasons": {str(list(s)): r for s, r in failing.items()},
                       "instances": detail}  # fmt: skip
    chosen = max(lengths.values())
    lines.append(f"L* = {chosen}, qualified for condition {condition} only (provisional until recorded in decisions/)")
    return lines, {"verdict": "CONVERGED", "length": chosen, "failed": False, "condition": condition, "instances": detail}


def _own_entropy(run: dict, variant: str) -> float:
    spent = run["policy_entropy"]
    return spent["action"] if variant == "sac" else spent["action"] - spent["within_class"]  # class entropy with tied drugs


def step2(reports: list[dict]) -> tuple[list[str], dict]:
    lines, failed, widen = [], False, False
    lowest: dict[tuple, float] = {}
    fingerprints = {report.get("platform_fingerprint") for report in reports}
    fingerprints |= {r["host"]["fingerprint"]["platform_fingerprint"] for report in reports for r in report["runs"] if "fingerprint" in r.get("host", {})}
    if len(fingerprints) != 1 or None in fingerprints:
        lines.append(f"platform: the reports and jobs carry {len(fingerprints)} platform fingerprints, not one -> FAIL")
        failed = True
    if any(r["host"]["fingerprint"]["arithmetic_canary"]["size"] != "full" for report in reports for r in report["runs"] if "fingerprint" in r.get("host", {})):
        lines.append("platform: a job ran with a reduced arithmetic canary -> FAIL")
        failed = True
    for report in reports:
        if report["condition"] not in CONDITIONS:
            raise ValueError(f"{_name(report)} has an unknown condition {report['condition']!r}")
        for run in report["runs"]:
            if "diverged" in run:
                lines.append(f"{_name(report)} {run['variant']} seed {run['seed']}: diverged ({run['diverged']}) -> FAIL")
                failed = True
        for variant in VARIANTS:
            cfg = report["sac_config"][variant]
            low, high = cfg["log_alpha_range"]
            for seed, run in sorted(_runs(report, variant).items()):
                key = (_instance(report), report["condition"], variant)
                lowest[key] = min(lowest.get(key, math.inf), _own_entropy(run, variant))
                label = f"{_name(report)} {variant} seed {seed}"
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
    for (sizes, condition, variant), reached in sorted(lowest.items()):
        report = next(r for r in reports if _instance(r) == sizes and r["condition"] == condition and _runs(r, variant))
        rng = next(iter(_runs(report, variant).values()))["entropy_range"]
        gap = (reached - rng["floor"]) / (rng["maximum"] - rng["floor"])
        ok = gap <= 0.1
        lines.append(f"floor: {variant} on {list(sizes)} ({condition}), lowest entropy reached sits {gap:.3f} of the range above the computed floor "
                     f"(limit 0.1) -> {'PASS' if ok else 'FAIL'}")  # fmt: skip
        failed |= not ok
    return lines, {"failed": failed, "widen_upward": widen}


def main() -> int:
    parser = _Parser()
    parser.add_argument("step", choices=("step1", "step2"))
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--planner", type=Path, help="planner-restart report at (8, 5, 4, 2, 1); required for step1")
    args = parser.parse_args()
    if args.step == "step1" and args.planner is None:
        parser.error("step1 needs --planner")
    try:
        if args.step == "step1":
            lines, result = step1(_load(args.reports), json.loads(args.planner.read_text(encoding="utf-8")))
        else:
            lines, result = step2(_load(args.reports))
    except (KeyError, ValueError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    print("\n".join(lines))
    print(json.dumps(result))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
