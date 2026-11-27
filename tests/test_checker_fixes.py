"""Regressions for the dev-protocol checker's step 2, the phase scorer and null, checkpoint verification and the harness's seed gate."""

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
    host = {"fingerprint": {"platform_fingerprint": "f" * 64, "arithmetic_canary": {"size": "full"}}}
    for run in runs:
        run["host"] = host
    return {"_path": f"train_{list(sizes)}_{fixed}.json", "condition": "c-given", "platform_fingerprint": "f" * 64,
            "class_structure": {"class_sizes": list(sizes)}, "runs": runs,
            "git": {"commit": commit, "code_diff_sha256": None}, "sac_config": {"oc-sac": cfg, "sac": cfg}}  # fmt: skip


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


def test_filter_keeps_only_held_out_lines_and_directions(tmp_path):
    filt = _module("claim_a_filter_predictions")
    base, held = {"kappa": 0.1, "delta": 0.05, "horizon": 40}, {"kappa": 0.2, "delta": 0.05, "horizon": 40}
    rows = [{**s, "delay": d, "lam": 0.8, "sigma_star": 0.2, "sign_at_max_sigma": 1} for s in (base, held) for d in (1, 0)]
    spec = {"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": 1, "lines": rows,
            "directions": [{"from": base, "to": held, "sign": 1}, {"from": held, "to": held, "sign": 1}]}  # fmt: skip
    out = filt.filter_predictions(spec, [held])
    assert [(r["kappa"], r["delay"]) for r in out["lines"]] == [(0.2, 0), (0.2, 1)] and out["directions"] == [{"from": held, "to": held, "sign": 1}]
    assert out["filtered_from"] == {"lines": 4, "directions": 2}
    with pytest.raises(ValueError, match="no line at held-out setting"):
        filt.filter_predictions(spec, [{"kappa": 0.1, "delta": 0.1, "horizon": 40}])
    src, settings = tmp_path / "p.json", tmp_path / "s.json"
    src.write_text(json.dumps(spec), encoding="utf-8")
    settings.write_text(json.dumps({"settings": [held]}), encoding="utf-8")
    done = subprocess.run([sys.executable, str(EXPERIMENTS / "claim_a_filter_predictions.py"), str(src), str(settings), "--out", str(tmp_path / "f.json")],
                          capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    assert done.returncode == 0 and done.stdout.split()[0] == __import__("hashlib").sha256((tmp_path / "f.json").read_bytes()).hexdigest()
    assert b"\r" not in (tmp_path / "f.json").read_bytes()


def test_the_committed_predictions_filter_to_the_forty_held_out_lines():
    filt, null = _module("claim_a_filter_predictions"), _module("null_phase_predictions")
    spec = json.loads((ROOT / "predictions/claim-a-2026-11-26/stage2/predictions.json").read_text(encoding="utf-8"))
    out = filt.filter_predictions(spec, null.load_settings(EXPERIMENTS / "claim_a_heldout_settings.json"))
    assert len(out["lines"]) == 40 and {(r["kappa"], r["delta"]) for r in out["lines"]} == {(0.2, 0.05), (0.1, 0.1)}
    assert out["directions"] == []  # the one direction check starts at the dev setting, which the test does not measure


def test_scorer_refuses_a_report_without_evaluation_episodes():
    scorer = _module("score_phase_predictions")
    report = _phase_report({(1, 0.8): 0.2})
    del report["evaluation_episode_seeds"]
    with pytest.raises(ValueError, match="evaluation_episode_seeds missing"):
        scorer.measured_lines([report], "planned_once_minus_greedy")


def _dev_map(spreads=(0.02, 0.03), zero_line=False):
    """A two-line dev map: per-episode contrasts with the given SDs; optionally a line whose contrast is 0 everywhere."""
    rng = np.random.default_rng(0)
    results = []
    for line, sd in enumerate(spreads):
        for s in (0.0, 0.1, 0.2, 0.4):
            greedy = [0.4] * 40
            planned = [0.4 + (s - 0.15) + rng.normal(0, sd) for _ in range(40)] if not (zero_line and line == 0) else greedy
            results.append({"delay": line, "lam": 0.8, "sigma": s, "policy": "greedy", "expected_failure_rate": greedy})
            results.append({"delay": line, "lam": 0.8, "sigma": s, "policy": "planned-once", "expected_failure_rate": planned})
    return {"args": {"tier": "dev"}, "instance": {"kappa": 0.1, "delta": 0.05, "horizon_quarters": 40},
            "evaluation_episode_seeds": list(range(10_000, 10_040)), "results": results}  # fmt: skip


def test_episode_count_follows_the_largest_spread_within_the_floor_and_cap():
    count = _module("claim_a_episode_count")
    stats = _module("stats")
    assert count.episode_count(_dev_map((0.001, 0.002))) == 40  # tiny spreads: the floor
    assert count.episode_count(_dev_map((0.02, 0.2))) == 100  # one wide cell: the cap binds
    n = count.episode_count(_dev_map((0.02, 0.02)))
    assert 40 < n < 100 and stats.t_quantile(0.975, n - 1) * 0.02 / math.sqrt(n) <= 0.005 * 1.6  # at the spread's own scale
    with pytest.raises(ValueError, match="zero contrast at every sigma"):
        count.episode_count(_dev_map(zero_line=True))


def test_gate_pins_recompute_from_the_files_the_preregistrations_name():
    gate = _module("claim_a_gate")
    for name in ("kappa-2", "delta-2"):
        prereg = ROOT / "preregistrations" / f"2026-11-27-claim-a-boundary-{name}.yaml"
        pins = gate.read_pins(prereg)
        computed = gate.computed_pins(ROOT / pins["dev_map"], ROOT / pins["heldout_settings"], ROOT / pins["predictions"], int(pins["sign_beyond_crossing"]))
        assert {k: pins[k] for k in computed} == computed, name
        code = gate.read_code_pins(prereg)
        assert {path: gate.file_hash(ROOT / path) for path in code} == code, name  # every pinned script is the one on disk
    kappa, delta = (gate.read_pins(ROOT / "preregistrations" / f"2026-11-27-claim-a-boundary-{n}-2.yaml") for n in ("kappa", "delta"))
    assert {k: v for k, v in kappa.items() if k != "setting"} == {k: v for k, v in delta.items() if k != "setting"}  # one rule, two launches
    assert (kappa["setting"], delta["setting"]) == ("0.2/0.05", "0.1/0.1")


def test_gate_refuses_a_preregistration_with_a_missing_or_doubled_pin(tmp_path):
    gate = _module("claim_a_gate")
    text = (ROOT / "preregistrations" / "2026-11-27-claim-a-boundary-kappa-2.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "x.yaml"
    broken.write_text(text.replace("\nseed: 0\n", "\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the pins"):
        gate.read_pins(broken)
    broken.write_text(text + "\nseed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        gate.read_pins(broken)


def test_scorer_verdict_against_the_null_fails_the_null_itself_and_refuses_a_mismatched_null(tmp_path):
    scorer = _module("score_phase_predictions")
    report = _phase_report({(0, 0.8): 0.2, (1, 0.8): 0.25})
    lines, episodes = scorer.measured_lines([report], "planned_once_minus_greedy")
    null = {(0.1, 0.05, 40, d, 0.8): (0.2, 1) for d in (0, 1)}
    out = scorer.score(lines, episodes, null, null, [], 1, math.log(math.sqrt(2)), 200, 0)
    assert out["E_minus_E_null_ci95"] == [0.0, 0.0] and out["verdict_vs_null"] == "fail"
    better = {(0.1, 0.05, 40, 0, 0.8): (0.2, 1), (0.1, 0.05, 40, 1, 0.8): (0.25, 1)}
    worse = {(0.1, 0.05, 40, 0, 0.8): (0.4, 1), (0.1, 0.05, 40, 1, 0.8): (0.4, 1)}
    assert scorer.score(lines, episodes, better, worse, [], 1, math.log(math.sqrt(2)), 200, 0)["verdict_vs_null"] == "pass"
    theory, other = tmp_path / "t.json", tmp_path / "n.json"
    rows = [{"kappa": 0.1, "delta": 0.05, "horizon": 40, "delay": d, "lam": 0.8, "sigma_star": 0.2, "sign_at_max_sigma": 1} for d in (0, 1)]
    theory.write_text(json.dumps({"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": 1, "lines": rows}), encoding="utf-8")
    other.write_text(json.dumps({"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": -1, "lines": rows}), encoding="utf-8")
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps(report), encoding="utf-8")
    done = subprocess.run([sys.executable, str(EXPERIMENTS / "score_phase_predictions.py"), str(theory), str(rep), "--null", str(other), "--resamples", "20"],
                          capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    assert done.returncode == 2 and "--null predicts" in done.stderr


def test_gate_reproduces_the_rerun_bitwise_but_refuses_it_as_run_one(capsys):
    gate = _module("claim_a_gate")
    args = type("A", (), {"run1": ROOT / "runs/phase/phase_20261126_213058_19724.json",
                          "prereg": ROOT / "preregistrations/2026-11-27-claim-a-boundary-kappa-2.yaml"})()
    assert gate.cmd_after_run1(args) == 1  # the 26 November rerun predates the preregistration, so it cannot be run 1
    out = capsys.readouterr().out
    assert "PASS: run 1 against the pinned map: PASS: 480 results, 96000 values compared, 0 mismatched" in out  # the comparator's suffix is allowed
    assert "PASS: null_sha256 rebuilt from run 1" in out and "PASS: episodes recomputed on run 1" in out
    assert "FAIL: run 1's commit holds this preregistration byte for byte" in out and "GATE FAIL" in out


def test_gate_refuses_held_out_reports_that_are_not_the_preregistered_ones():
    gate = _module("claim_a_gate")
    pins = gate.read_pins(ROOT / "preregistrations/2026-11-27-claim-a-boundary-kappa-2.yaml")
    report = _held_out_pair(episodes=100)[0]
    report["args"]["prereg"] = "preregistrations/2026-11-27-claim-a-boundary-kappa-2.yaml"
    problems = gate.check_held_out_report(report, ROOT / "x.json", pins)
    assert any("results, not 480" not in p for p in problems) and any("committed preregistration" in p or "prereg_sha256" in p for p in problems)
    report["args"]["tier"] = "dev"
    assert "not a test-tier report" in gate.check_held_out_report(report, ROOT / "x.json", pins)
    report["results"][0]["expected_failure_rate"][0] = float("nan")
    assert "a missing or non-finite value" in gate.check_held_out_report(report, ROOT / "x.json", pins)


def _held_out_pair(episodes=12):
    """Two synthetic held-out reports on shared episodes: lines cross at 0.15 (kappa 0.2) and never (delta 0.1)."""
    reports = []
    for kappa, delta, crossing_at in ((0.2, 0.05, 0.15), (0.1, 0.1, None)):
        results = []
        for d in (0, 1, 2, 4, 8):
            for lam in (0.0, 0.5, 0.8, 0.95):
                for s in (0.0, 0.1, 0.2, 0.4):
                    offset = (s - crossing_at) if crossing_at is not None else -0.05
                    noise = [0.002 * ((e % 3) - 1) for e in range(episodes)]
                    results.append({"delay": d, "lam": lam, "sigma": s, "policy": "greedy", "expected_failure_rate": [0.4] * episodes})
                    results.append({"delay": d, "lam": lam, "sigma": s, "policy": "planned-once", "expected_failure_rate": [0.4 + offset + n for n in noise]})
        reports.append({"args": {"tier": "test"}, "instance": {"kappa": kappa, "delta": delta, "horizon_quarters": 40},
                        "evaluation_episode_seeds": list(range(11_000, 11_000 + episodes)), "results": results})  # fmt: skip
    return reports


def test_descriptives_count_censoring_sides_and_bracket_deltas(tmp_path):
    desc, filt, null_mod = _module("claim_a_descriptives"), _module("claim_a_filter_predictions"), _module("null_phase_predictions")
    reports = _held_out_pair()
    settings = null_mod.load_settings(EXPERIMENTS / "claim_a_heldout_settings.json")
    rows = [{**s, "delay": d, "lam": lam, "sigma_star": 0.15, "sign_at_max_sigma": 1} for s in settings for d in (0, 1, 2, 4, 8) for lam in (0.0, 0.5, 0.8, 0.95)]
    theory = filt.filter_predictions({"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": 1, "lines": rows}, settings)
    null = {"contrast": "planned_once_minus_greedy", "sign_beyond_crossing": 1,
            "lines": [{**r, "sigma_star": None, "sign_at_max_sigma": -1} for r in rows]}  # the null says: never crosses
    out = desc.describe(reports, theory, null, 1, None)
    assert out["censoring"]["counts"]["measured"] == {"below": 0, "inside": 20, "above": 20}
    assert out["censoring"]["counts"]["theory"] == {"below": 0, "inside": 40, "above": 0}
    assert out["censoring"]["counts"]["null"] == {"below": 0, "inside": 0, "above": 40}
    assert out["censoring"]["lines_with_zero_error_by_same_side_censoring"] == {"theory": 0, "null": 20}
    assert out["delta"]["lines_with_interior_crossing"] == 20 and out["delta"]["delta_min"] == pytest.approx(0.1, abs=1e-9)
    assert len(out["cells"]) == 160 and all(c["half_width_95"] >= 0 for c in out["cells"])
