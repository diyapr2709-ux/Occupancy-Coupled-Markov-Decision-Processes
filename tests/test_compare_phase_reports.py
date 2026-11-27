"""The reproduction comparator passes identical reports and fails every kind of difference, without printing values."""
import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_phase_reports", ROOT / "tools" / "compare_phase_reports.py")
compare_phase_reports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_phase_reports)
compare = compare_phase_reports.compare


def _report():
    results = [
        {"delay": d, "lam": 0.5, "sigma": 0.0, "policy": p, "seconds": 1.0,
         "expected_failure_rate": [0.25, 0.5], "final_fail_prob": [[0.1, 0.2], [0.3, 0.4]]}
        for d in (0, 1) for p in ("greedy", "planned-once")
    ]  # fmt: skip
    return {"results": results, "evaluation_episode_seeds": [10_000, 10_001], "instance": {"horizon_quarters": 6},
            "policies": ["greedy", "planned-once"], "contrasts": {}, "args": {"tier": "dev", "out": "runs/phase"}}  # fmt: skip


def test_identical_reports_pass_and_wall_time_and_output_directory_are_ignored():
    first, second = _report(), _report()
    second["results"][0]["seconds"] = 99.0
    second["args"]["out"] = "elsewhere"
    passed, line = compare(first, second)
    assert passed and line.startswith("PASS: 4 results, 24 values compared, 0 mismatched")


def test_one_value_off_by_one_ulp_fails():
    first, second = _report(), _report()
    second["results"][2]["expected_failure_rate"][1] = 0.5000000000000001
    passed, line = compare(first, second)
    assert not passed and "1 mismatched" in line and "0.5" not in line.split("results")[1]


def test_negative_zero_is_not_equal_to_zero():
    first, second = _report(), _report()
    first["results"][1]["final_fail_prob"][0][0] = 0.0
    second["results"][1]["final_fail_prob"][0][0] = -0.0
    assert not compare(first, second)[0]


def test_a_missing_field_fails():
    first, second = _report(), _report()
    del first["results"][3]["final_fail_prob"]
    del second["results"][3]["final_fail_prob"]
    passed, line = compare(first, second)
    assert not passed and "1 missing or malformed" in line


def test_a_different_grid_fails():
    first, second = _report(), _report()
    second["results"][0]["delay"] = 7
    passed, line = compare(first, second)
    assert not passed and "grids differ" in line


def test_reports_that_compare_nothing_or_lack_a_setup_field_fail():
    empty = _report()
    empty["results"] = []
    assert not compare(empty, copy.deepcopy(empty))[0]
    first, second = _report(), _report()
    del first["contrasts"], second["contrasts"]
    passed, line = compare(first, second)
    assert not passed and "missing" in line and "contrasts" in line


def test_a_worker_count_difference_fails_unless_that_argument_is_named_as_ignored():
    first, second = _report(), _report()
    first["args"]["workers"], second["args"]["workers"] = 30, 20
    assert not compare(first, second)[0]
    passed, line = compare(first, second, ("workers",))
    assert passed and line.startswith("PASS")
    second["args"]["tier"] = "val"
    assert not compare(first, second, ("workers",))[0]  # ignoring one argument leaves the others checked


def test_a_different_setup_fails_and_is_named():
    first, second = _report(), copy.deepcopy(_report())
    second["evaluation_episode_seeds"] = [10_000, 10_002]
    second["args"]["tier"] = "val"
    passed, line = compare(first, second)
    assert not passed and "evaluation_episode_seeds" in line and "args" in line
