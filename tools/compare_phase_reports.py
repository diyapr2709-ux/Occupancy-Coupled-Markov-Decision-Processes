"""Reproduction check between two phase-diagram reports. Prints PASS or FAIL and counts, never a value.

Usage: python tools/compare_phase_reports.py FIRST.json SECOND.json [--ignore-arg NAME ...]

The reports' setup must agree: evaluation episodes, instance, policies, contrasts, and the
launch arguments apart from `out` and any argument named with --ignore-arg. An argument may be
ignored only under a decisions/ entry written before the second report's launch. Results are matched on (delay, lam, sigma, policy), and the
grids must be identical. Every per-episode `expected_failure_rate` value and every
`final_fail_prob` entry is compared through its IEEE 754 bytes, so the check is bitwise: it
also tells 0.0 from -0.0. `seconds` is ignored, because it measures wall time, not the result.
Two reports that compare no values, or that lack a setup field, FAIL rather than pass vacuously.
Exit code 0 on PASS, 1 on FAIL.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

FIELDS = ("expected_failure_rate", "final_fail_prob")
SETUP = ("evaluation_episode_seeds", "instance", "policies", "contrasts")


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def bits(value) -> bytes:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a number: {type(value).__name__}")
    return struct.pack("<d", float(value))


def keyed(report: dict) -> dict:
    out = {}
    for result in report["results"]:
        key = (result["delay"], result["lam"], result["sigma"], result["policy"])
        if key in out:
            raise ValueError(f"duplicate result for {key}")
        out[key] = result
    return out


def compare(first: dict, second: dict, ignore_args: tuple = ()) -> tuple[bool, str]:
    skipped = {"out", *ignore_args}
    missing = [name for name in SETUP if name not in first or name not in second]
    if missing:
        return False, f"FAIL: setup field missing from a report: {', '.join(missing)}"
    differing = [name for name in SETUP if first[name] != second[name]]
    args = [{k: v for k, v in r.get("args", {}).items() if k not in skipped} for r in (first, second)]
    if args[0] != args[1]:
        differing.append(f"args (apart from {', '.join(sorted(skipped))})")
    if differing:
        return False, f"FAIL: setup differs in {', '.join(differing)}"
    try:
        a, b = keyed(first), keyed(second)
    except ValueError as error:
        return False, f"FAIL: {error}"
    if set(a) != set(b):
        return False, f"FAIL: result grids differ ({len(set(a) ^ set(b))} keys in only one report)"
    compared = mismatched = malformed = 0
    for key in a:
        for field in FIELDS:
            x, y = a[key].get(field), b[key].get(field)
            if not isinstance(x, list) or not isinstance(y, list):
                malformed += 1
                continue
            x, y = list(flatten(x)), list(flatten(y))
            if len(x) != len(y):
                malformed += 1
                continue
            try:
                mismatched += sum(1 for u, v in zip(x, y) if bits(u) != bits(v))
            except TypeError:
                malformed += 1
                continue
            compared += len(x)
    passed = compared > 0 and mismatched == 0 and malformed == 0
    verdict = "PASS" if passed else "FAIL"
    return passed, f"{verdict}: {len(a)} results, {compared} values compared, {mismatched} mismatched, {malformed} missing or malformed fields"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--ignore-arg", action="append", default=[], metavar="NAME", help="a launch argument left out of the setup check")
    args = parser.parse_args()
    first, second = (json.loads(p.read_text(encoding="utf-8")) for p in (args.first, args.second))
    passed, line = compare(first, second, tuple(args.ignore_arg))
    if args.ignore_arg:
        line += f" (launch arguments not compared: {', '.join(sorted(args.ignore_arg))})"
    print(line)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
