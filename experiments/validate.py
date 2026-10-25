"""Run the V&V harness: every verification check, then the validation measurements.

    python experiments/validate.py            # full seed protocol
    python experiments/validate.py --quick    # two seeds, for iteration
    python experiments/validate.py --verification-only

Writes a JSON report to runs/vv/ and exits non-zero if any verification check fails.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym

import ocmdp.envs  # noqa: F401  registers the Gymnasium ids
from ocmdp.envs.toy import ToyConfig
from ocmdp.eval.seeds import SEEDS
from ocmdp.eval.vv import comparator_table, distillation_study, lambda_misspecification, noise_robustness, run_verification
from ocmdp.theory.two_drug import solve_two_drug

COLUMNS = ["expected_failure_rate", "failure_rate", "regret", "first_fixation_q", "efficacy_reserve", "H_action", "H_pushforward"]
ENV_IDS = ["ocmdp/TwoDrug-v0", "ocmdp/PairedDrugs-v0"]


def _cell(agg: dict) -> str:
    if agg["mean"] is None:
        return "never"
    cell = f"{agg['mean']:.3f}" if agg["std"] is None else f"{agg['mean']:.3f} ± {agg['std']:.3f}"
    return f"{cell} ({agg['never']} never)" if agg["never"] else cell


def _report_path(out: Path) -> Path:
    """A fresh file per run: the pid and microseconds keep concurrent runs from overwriting each other."""
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() // 1000 % 1_000_000:06d}_{os.getpid()}"
    return out / f"vv_{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--quick", action="store_true", help="first two seeds only")
    parser.add_argument("--verification-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/vv"))
    args = parser.parse_args()
    seeds = tuple(args.seeds[:2] if args.quick else args.seeds)
    # fail before any work, not after an hour of it
    if not seeds:
        parser.error("--seeds needs at least one seed")
    if any(s < 0 for s in seeds):
        parser.error("seeds must be non-negative")
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    path = _report_path(args.out)
    report: dict = {"seeds": list(seeds)}

    def save() -> None:  # after every section, so a late crash keeps everything already measured
        path.write_text(json.dumps(report, indent=2))

    print("Verification")
    checks = run_verification()
    for c in checks:
        print(f"  [{c.status:^7}] {c.name:<45} {c.detail}")
    report["verification"] = [vars(c) for c in checks]
    failed = [c.name for c in checks if c.status not in ("pass", "pending")]
    save()

    if not args.verification_only:
        report["validation"] = {}
        for env_id in ENV_IDS:
            env = gym.make(env_id)
            table = comparator_table(env, seeds)
            report["validation"][env_id] = table
            save()
            width = max(len("policy"), *map(len, table)) + 2
            print(f"\nValidation: {env_id}  ({len(seeds)} seeds, {env.unwrapped.config.horizon_quarters} quarters)")
            print(f"{'policy':<{width}}" + "".join(f"{c:>23}" for c in COLUMNS))
            for name, row in table.items():
                print(f"{name:<{width}}" + "".join(f"{_cell(row[c]):>23}" for c in COLUMNS))

        env = gym.make(ENV_IDS[0])
        sweep = lambda_misspecification(env, seeds)
        report["lambda_misspecification"] = {str(k): v for k, v in sweep.items()}
        save()
        print(f"\nOracle under misspecified lam on {ENV_IDS[0]} (true lam = {env.unwrapped.config.lam})")
        for factor, row in sweep.items():
            print(f"  horizon x{factor:<5} lam' = {row['lam']:.3f}   expected failure rate {_cell(row['expected_failure_rate'])}")

        report["noise_robustness"] = {}
        for name, cfg in (("TwoDrug", ToyConfig.two_drug()), ("PairedDrugs", ToyConfig.paired())):
            study = noise_robustness(cfg, seeds)
            report["noise_robustness"][name] = {str(k): v for k, v in study.items()}
            save()
            print(f"\nExpected failure rate under resistance process noise, {name}")
            for sigma, row in study.items():
                print(f"  sigma {sigma}")
                for k, v in row.items():
                    print(f"    {k:<24} {_cell(v)}")

        study = distillation_study(ToyConfig.paired(), seeds)
        report["distillation"] = study
        save()
        print("\nC3 distilled from the oracle, PairedDrugs (expected failure rate)")
        for name, agg in study.items():
            print(f"  {name:<26} {_cell(agg)}")

        sol = solve_two_drug()
        report["two_drug_mean_field"] = vars(sol) | {"gap": sol.gap}
        save()
        print(f"\nMean-field two-drug: best stationary F = {sol.f_opt:.3f} at p = {sol.p_opt:.2f}; best deterministic F = {sol.f_deterministic:.3f}")

    print(f"\nReport: {path}")
    if failed:
        print(f"FAILED verification: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
