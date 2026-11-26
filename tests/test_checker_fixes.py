"""Regressions for the dev-protocol checker, the phase scorer and null, checkpoint verification and the harness's seed gate."""

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _module(name: str):
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EXPERIMENTS))


def _step1_report(sizes, quarters, rate=0.32, commit="abc", seeds=range(10), episodes=(10_000, 10_001)):
    """A fixed-temperature-0 report: both agents end with the same weights; uniform 0.40, replanner 0.30."""
    runs = [{"variant": v, "sigma": 0.0, "seed": s, "expected_failure_rate": [rate, rate],
             "parameter_sha256": {"initial": f"init-{s}", "after_first_window": f"first-{s}", "final": f"final-{s}-{quarters}"}}
            for s in seeds for v in ("oc-sac", "sac")]  # fmt: skip
    config = {"temperature": "fixed", "alpha": 0.0, "total_quarters": quarters}
    return {"_path": f"train_{list(sizes)}_{quarters}.json", "class_structure": {"class_sizes": list(sizes)}, "runs": runs,
            "git": {"commit": commit, "code_diff_sha256": None}, "evaluation_episode_seeds": list(episodes),
            "sac_config": {"oc-sac": config, "sac": config},
            "baselines": {"0.0": {"uniform": {"expected_failure_rate": [0.40, 0.40]}, "oracle[replans every quarter]": {"expected_failure_rate": [0.30, 0.30]}}},
            "summary": {"0.0": {"oc-sac": {"runs_learned": 10}}}}  # fmt: skip


def _six(**overrides):
    return [_step1_report(sizes, q, **overrides) for sizes in ((2, 2), (8, 5, 4, 2, 1)) for q in (12_000, 24_000, 48_000)]


def test_step1_passes_the_full_set():
    lines, result = _module("check_dev_protocol").step1(_six())
    assert result == {"failed": False, "bitwise": True, "length": 12_000}, lines


def test_step1_fails_a_set_missing_an_instance():
    reports = [_step1_report((2, 2), q) for q in (12_000, 24_000, 48_000)]
    lines, result = _module("check_dev_protocol").step1(reports)
    assert result["failed"] and result["length"] is None
    assert any("needs exactly" in line for line in lines)


def test_step1_refuses_a_duplicate_report():
    with pytest.raises(ValueError, match="two reports"):
        _module("check_dev_protocol").step1(_six() + [_step1_report((2, 2), 12_000)])


@pytest.mark.parametrize("field, other", [("commit", {"commit": "def"}), ("seeds", {"seeds": range(1, 11)}), ("episodes", {"episodes": (30_000, 30_001)})])
def test_step1_refuses_reports_from_different_code_seeds_or_episodes(field, other):
    reports = _six()
    reports[-1] = _step1_report((8, 5, 4, 2, 1), 48_000, **other)
    with pytest.raises(ValueError, match="differ in"):
        _module("check_dev_protocol").step1(reports)


def _step2_report(alpha_path, alpha_eval, sizes=(2, 2), fixed=None, sac_entropy=0.05, commit="abc", quarters=12_000, ranges=None):
    ranges = ranges or {"oc-sac": {"floor": 0.0, "maximum": math.log(len(sizes)), "target": 0.89 * math.log(len(sizes))},
                        "sac": {"floor": 0.0, "maximum": math.log(sum(sizes)), "target": 0.89 * math.log(sum(sizes))}}  # fmt: skip
    cfg = {"temperature": "fixed" if fixed is not None else "auto", "alpha": fixed if fixed is not None else 0.0,
           "log_alpha_range": [-12.0, 2.0], "total_quarters": quarters, "target_fraction": None if fixed is not None else 0.89}  # fmt: skip
    runs = []
    for v in ("oc-sac", "sac"):
        entropy = {"action": math.log(2), "within_class": math.log(2)} if v == "oc-sac" else {"action": sac_entropy, "within_class": 0.0}
        runs.append({"variant": v, "sigma": 0.0, "seed": 0, "alpha_path": list(alpha_path), "alpha_at_evaluation": alpha_eval,
                     "entropy_path": [ranges[v]["target"]], "entropy_range": ranges[v], "policy_entropy": entropy})  # fmt: skip
    return {"_path": f"train_{list(sizes)}_{fixed}.json", "class_structure": {"class_sizes": list(sizes)}, "runs": runs,
            "git": {"commit": commit, "code_diff_sha256": None}, "sac_config": {"oc-sac": cfg, "sac": cfg}}


def test_step2_holds_a_fixed_temperature_stored_as_float32():
    stored = float(np.float32(0.1))  # what the agent logs for a fixed 0.1
    lines, result = _module("check_dev_protocol").step2([_step2_report([stored] * 4, stored, fixed=0.1)])
    assert not result["failed"], lines
    assert _module("check_dev_protocol").step2([_step2_report([stored, stored, 0.2, stored], stored, fixed=0.1)])[1]["failed"]


def test_step2_reports_the_floor_gap_without_failing():
    sizes = (8, 5, 4, 2, 1)
    rotating = 1.154  # mean action entropy of a rotation over the five classes, far above SAC's floor of 0
    lines, result = _module("check_dev_protocol").step2([_step2_report([1.0] * 4, 1.0, sizes=sizes, fixed=1.0, sac_entropy=rotating)])
    assert not result["failed"], lines
    assert any(line.startswith("floor (reported, not gating): sac") for line in lines)


def test_step2_fails_a_learned_temperature_held_near_the_clamp():
    near = math.exp(-12.0) * (399 + math.e) / 400  # window mean of 399 updates at the clamp and one just above
    assert math.log(near) > -12.0 + 1e-6  # the old strict-inside test let this through
    lines, result = _module("check_dev_protocol").step2([_step2_report([0.2, 0.15, near, near], near)])
    assert result["failed"], lines
    assert not _module("check_dev_protocol").step2([_step2_report([0.2, 0.15, 0.12, 0.11], 0.11)])[1]["failed"]


def test_step2_refuses_duplicates_and_mixed_code():
    protocol = _module("check_dev_protocol")
    with pytest.raises(ValueError, match="same instance, length and temperature"):
        protocol.step2([_step2_report([1.0] * 4, 1.0, fixed=1.0), _step2_report([1.0] * 4, 1.0, fixed=1.0)])
    with pytest.raises(ValueError, match="differ in"):
        protocol.step2([_step2_report([1.0] * 4, 1.0, fixed=1.0), _step2_report([3.0] * 4, 3.0, fixed=3.0, commit="def")])


def _phase_report(lines_at, sigmas=(0.0, 0.1, 0.2, 0.3, 0.4), episodes=6, kappa=0.1):
    """planned-once minus greedy equals sigma - crossing for each (delay, lam) in `lines_at`, plus small episode noise."""
    results = []
    for (delay, lam), crossing_at in lines_at.items():
        for s in sigmas:
            noise = [0.001 * ((e % 3) - 1) for e in range(episodes)]
            results.append({"delay": delay, "lam": lam, "sigma": s, "policy": "greedy", "expected_failure_rate": [0.4] * episodes})
            results.append({"delay": delay, "lam": lam, "sigma": s, "policy": "planned-once",
                            "expected_failure_rate": [0.4 + (s - crossing_at) + n for n in noise]})  # fmt: skip
    return {"args": {"tier": "dev"}, "instance": {"kappa": kappa, "delta": 0.05, "horizon_quarters": 40},
            "evaluation_episode_seeds": list(range(10_000, 10_000 + episodes)), "results": results}  # fmt: skip


def test_scorer_counts_lines_left_out_of_the_predictions():
    scorer = _module("score_phase_predictions")
    report = _phase_report({(0, 0.8): 0.2, (1, 0.8): 0.2, (2, 0.8): 0.2})
    lines, episodes = scorer.measured_lines([report], "planned_once_minus_greedy")
    only_one = {(0.1, 0.05, 40, 0, 0.8): (0.2, 1)}  # right on the one line it names
    out = scorer.score(lines, episodes, only_one, None, [], 1, math.log(math.sqrt(2)), 50, 0)
    assert out["E"] == math.inf and out["verdict"] == "fail"
    assert [row["delay"] for row in out["not_predicted"]] == [1, 2]
    every = {(0.1, 0.05, 40, d, 0.8): (0.2, 1) for d in (0, 1, 2)}
    full = scorer.score(lines, episodes, every, None, [], 1, math.log(math.sqrt(2)), 50, 0)
    assert full["E"] == pytest.approx(0.0, abs=0.02) and full["verdict"] == "pass" and full["not_predicted"] == []


def test_scorer_merges_a_line_split_across_reports_and_refuses_a_repeated_sigma():
    scorer = _module("score_phase_predictions")
    low = _phase_report({(1, 0.8): 0.2}, sigmas=(0.0, 0.1, 0.2))
    high = _phase_report({(1, 0.8): 0.2}, sigmas=(0.3, 0.4))
    lines, _ = scorer.measured_lines([low, high], "planned_once_minus_greedy")
    sigmas, values = lines[(0.1, 0.05, 40, 1, 0.8)]
    assert list(sigmas) == [0.0, 0.1, 0.2, 0.3, 0.4] and values.shape == (5, 6)
    with pytest.raises(ValueError, match="measured twice"):
        scorer.measured_lines([low, _phase_report({(1, 0.8): 0.2}, sigmas=(0.2, 0.3))], "planned_once_minus_greedy")


def test_null_reads_a_held_out_file_that_records_its_rule(tmp_path):
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(_phase_report({(1, 0.8): 0.2})), encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"rule": "one rate doubled at a time", "settings": [{"kappa": 0.2, "delta": 0.05, "horizon": 40},
                                                                                   {"kappa": 0.1, "delta": 0.1, "horizon": 40}]}), encoding="utf-8")  # fmt: skip
    done = subprocess.run([sys.executable, str(EXPERIMENTS / "null_phase_predictions.py"), str(dev), str(settings),
                           "--contrast", "planned_once_minus_greedy", "--sign-beyond-crossing", "1"], capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert {(row["kappa"], row["delta"]) for row in json.loads(done.stdout)["lines"]} == {(0.2, 0.05), (0.1, 0.1)}


def test_null_reads_the_committed_held_out_settings():
    settings = _module("null_phase_predictions").load_settings(EXPERIMENTS / "claim_a_heldout_settings.json")
    assert all({"kappa", "delta", "horizon"} <= set(s) for s in settings) and len(settings) == 2


def test_verify_checkpoints_fails_a_report_with_no_checkpoints(tmp_path):
    report = tmp_path / "train_x.json"
    report.write_text(json.dumps({"runs": [{"variant": "oc-sac", "sigma": 0.0, "seed": 0}, {"variant": "sac", "sigma": 0.0, "seed": 0}]}), encoding="utf-8")
    problems = _module("verify_checkpoints").verify(report)
    assert len(problems) == 2 and all(p.startswith("no checkpoint recorded") for p in problems)
    report.write_text(json.dumps({"runs": [{"variant": "sac", "sigma": 0.0, "seed": 0, "diverged": "nan loss"}]}), encoding="utf-8")
    assert _module("verify_checkpoints").verify(report) == []  # a diverged run has nothing to save


@pytest.mark.parametrize("flags", [["--seeds", "1000"], ["--noise-seeds", "1001"]])
def test_validate_refuses_test_tier_seeds(tmp_path, flags):
    done = subprocess.run([sys.executable, str(EXPERIMENTS / "validate.py"), "--verification-only", "--out", str(tmp_path / "vv"), *flags],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)  # fmt: skip
    assert done.returncode == 2 and "test tier" in done.stderr.replace("-", " "), done.stderr
    assert not (tmp_path / "vv").exists()


def test_stats_docstring_describes_the_exact_quantiles():
    doc = _module("stats").__doc__
    assert "3-decimal" not in doc and "1.96" not in doc and "_t975" in doc
