"""Regression tests for the launch guards and the summary fixed after the 26 Nov validation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    sys.path.insert(0, str(ROOT))
    try:
        from experiments import train_agent
    finally:
        sys.path.remove(str(ROOT))
    return train_agent


def _refusal(capsys, check, *args) -> str:
    with pytest.raises(SystemExit):
        check(argparse.ArgumentParser(), *args)
    return capsys.readouterr().err


def _done_run(seed: int, rates: list[float]) -> dict:
    return {
        "variant": "oc-sac", "sigma": 0.0, "seed": seed, "expected_failure_rate": rates, "argmax": {"expected_failure_rate": rates},
        "H_action": 0.5, "H_pushforward": 0.4, "policy_entropy": {"action": 0.5, "within_class": 0.1}, "critic": {}, "gap_attribution": {},
        "alpha_path": [0.1], "alpha_at_evaluation": 0.1, "discounted_failure_rate": rates, "checkpoints": [],
    }  # fmt: skip


def _diverged_run(seed: int) -> dict:
    return {"variant": "sac", "sigma": 0.0, "seed": seed, "host": {}, "diverged": "loss is not finite"}


def _baselines() -> dict:
    rates = {"uniform": [0.39, 0.39], "oracle": [0.34, 0.35], "oracle[replans every quarter]": [0.34, 0.35]}
    return {name: {"expected_failure_rate": r, "discounted_failure_rate": r} for name, r in rates.items()}


def test_summary_survives_a_variant_whose_runs_all_diverged():
    script = _script()
    runs = [_done_run(0, [0.35, 0.36]), _done_run(1, [0.37, 0.36]), _diverged_run(0), _diverged_run(1)]
    summary = script.summarise(runs, _baselines(), 0.0)
    assert summary["sac"] == {"all_runs_diverged": True, "runs": 0, "runs_learned": 0, "diverged": 2}
    assert summary["discounted_failure_rate"]["sac"] is None and summary["oc-sac"]["runs"] == 2
    assert summary["oc-sac minus sac (paired by seed)"] is None
    assert summary["paired comparison"] == {"seeds": [], "not_computed": "no seed has a finished run of both agents"}


class _InlinePool:
    """Stands in for ProcessPoolExecutor, so patched job functions run in this process."""

    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        value = fn(*args)
        return type("Done", (), {"result": lambda self: value})()

    def map(self, fn, items):
        return [fn(item) for item in items]


def test_a_launch_whose_sac_runs_all_diverged_still_writes_its_report(tmp_path, monkeypatch, capsys):
    script = _script()
    monkeypatch.setattr(script, "ProcessPoolExecutor", _InlinePool)
    monkeypatch.setattr(script, "evaluate_baselines", lambda job: {"sigma": job[0], "baselines": _baselines()})
    monkeypatch.setattr(script, "train_and_evaluate_pair", lambda job: [_done_run(job[1], [0.35, 0.36]), _diverged_run(job[1])])
    monkeypatch.setattr(sys, "argv", ["train_agent.py", "--tier", "dev", "--seeds", "0", "1", "--sigmas", "0.0", "--quarters", "200",
                                      "--episodes", "2", "--set", "learning_starts=64", "batch_size=32", "--out", str(tmp_path)])  # fmt: skip
    assert script.main() == 0
    (path,) = tmp_path.glob("train_*.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["summary"]["0.0"]["sac"]["all_runs_diverged"] is True and len(report["runs"]) == 4
    assert "all 2 runs diverged" in capsys.readouterr().out


def test_a_test_launch_writes_only_into_a_folder_directly_under_runs(tmp_path, capsys, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    (tmp_path / "runs" / "train").mkdir(parents=True)
    script.check_test_out(argparse.ArgumentParser(), tmp_path / "runs" / "train")
    for out in (tmp_path / "elsewhere", tmp_path / "runs", tmp_path / "runs" / "train" / "nested"):
        assert "directly under runs/" in _refusal(capsys, script.check_test_out, out)


def test_the_test_tier_cli_refusal_order_is_unchanged_by_the_out_check(tmp_path):
    result = subprocess.run([sys.executable, "experiments/train_agent.py", "--tier", "test", "--seeds", "1000", "--out", str(tmp_path / "x")],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert result.returncode == 2 and "--prereg" in result.stderr
    result = subprocess.run([sys.executable, "experiments/sweep_phase_diagram.py", "--tier", "test", "--out", str(tmp_path / "y")],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert result.returncode == 2 and "--prereg" in result.stderr


TEN = list(range(1000, 1010))


def test_test_seeds_must_be_explicit_enough_and_the_preregistrations_own(tmp_path, capsys, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    out = tmp_path / "runs" / "train"
    out.mkdir(parents=True)
    prereg = tmp_path / "planned.yaml"
    assert "explicitly" in _refusal(capsys, script.check_test_seeds, prereg, None, out)
    assert "at least 10 distinct seeds, got 9" in _refusal(capsys, script.check_test_seeds, prereg, TEN[:9], out)
    prereg.write_text("slug: planned\nplatform: rtx-1\n", encoding="utf-8")
    assert "exactly one `seeds:` line" in _refusal(capsys, script.check_test_seeds, prereg, TEN, out)
    prereg.write_text("seeds: [1000, 1001]\nseeds: 1002\n", encoding="utf-8")
    assert "exactly one `seeds:` line" in _refusal(capsys, script.check_test_seeds, prereg, TEN, out)
    prereg.write_text("seeds:\n  - 1000\n", encoding="utf-8")
    assert "exactly one `seeds:` line" in _refusal(capsys, script.check_test_seeds, prereg, TEN, out)
    prereg.write_text("seeds: 1000 1001 1002 1003 1004 1005 1006 1007 1008 1010\n", encoding="utf-8")
    assert "differ from the `seeds:` line" in _refusal(capsys, script.check_test_seeds, prereg, TEN, out)
    for line in ("seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]", "seeds: " + " ".join(map(str, reversed(TEN))),
                 "seeds: " + ",".join(map(str, TEN))):  # fmt: skip
        prereg.write_text(f"slug: planned\n{line}\nplatform: rtx-1\n", encoding="utf-8")
        script.check_test_seeds(argparse.ArgumentParser(), prereg, TEN, out)


def test_no_test_seed_is_trained_on_twice_whatever_the_preregistration(tmp_path, capsys, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    out = tmp_path / "runs" / "train"
    (out / "launches").mkdir(parents=True)
    prereg = tmp_path / "second.yaml"
    prereg.write_text("seeds: " + " ".join(map(str, TEN)) + "\n", encoding="utf-8")
    phase = {"launch_id": "phase_earlier", "tier": "test", "prereg_slug": "claim-a", "args": {"seeds": TEN}}
    (tmp_path / "runs" / "phase" / "launches").mkdir(parents=True)
    (tmp_path / "runs" / "phase" / "launches" / "phase_earlier.json").write_text(json.dumps(phase), encoding="utf-8")
    script.check_test_seeds(argparse.ArgumentParser(), prereg, TEN, out)  # a sweep's episodes are not training seeds
    val = {"launch_id": "train_val", "tier": "val", "prereg_slug": None, "args": {"seeds": TEN}}
    (out / "launches" / "train_val.json").write_text(json.dumps(val), encoding="utf-8")
    script.check_test_seeds(argparse.ArgumentParser(), prereg, TEN, out)
    first = {"launch_id": "train_first", "tier": "test", "prereg_slug": "first", "args": {"seeds": [1003, 1050]}}
    (out / "launches" / "train_first.json").write_text(json.dumps(first), encoding="utf-8")
    assert "test seeds [1003] were already used by launch train_first" in _refusal(capsys, script.check_test_seeds, prereg, TEN, out)


def test_main_checks_test_seeds_and_out_once_the_test_tier_checks_pass(tmp_path, capsys, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "ROOT", tmp_path)
    (tmp_path / "seeds.yaml").write_text((ROOT / "seeds.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(script, "git_state", lambda: {"commit": "fixture", "dirty": False, "code_diff_sha256": None, "untracked_code": []})
    monkeypatch.setattr(script, "check_test_launch", lambda parser, prereg, git, out, platform: "planned")  # stands in for a passed approval
    (tmp_path / "planned.yaml").write_text("seeds: " + " ".join(map(str, TEN)) + "\n", encoding="utf-8")
    base = ["train_agent.py", "--tier", "test", "--platform", "rtx-1", "--prereg", str(tmp_path / "planned.yaml")]
    for extra, message in (
        (["--out", str(tmp_path / "runs" / "train")], "explicitly"),
        (["--seeds", *map(str, TEN), "--out", str(tmp_path / "elsewhere")], "directly under runs/"),
        (["--seeds", *map(str, TEN[:9]), "--out", str(tmp_path / "runs" / "train")], "at least 10"),
    ):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit):
            script.main()
        assert message in capsys.readouterr().err
    assert not list(tmp_path.glob("runs/*/launches/*.json"))  # refused before any launch record


def test_set_cannot_change_the_run_length_and_is_checked_at_it(capsys):
    script = _script()
    assert "--set total_quarters is not accepted" in _refusal(capsys, script._parse_overrides, ["total_quarters=500"], 12_000)
    assert script._parse_overrides(argparse.ArgumentParser(), ["learning_starts=13000"], 48_000) == {"learning_starts": 13_000}
    assert "learning_starts < total_quarters" in _refusal(capsys, script._parse_overrides, ["learning_starts=13000"], 12_000)


def test_a_seed_file_cannot_relabel_val_seeds_as_dev(tmp_path, capsys):
    script = _script()
    shifted = tmp_path / "seeds.yaml"
    shifted.write_text("dev: [100, 199]\nval: [200, 299]\ntest: [1000, 1299]\n", encoding="utf-8")
    tiers = script.load_tiers(shifted)
    assert "the dev tier of seeds.yaml overlaps the repository's val seeds [100, 199]" in _refusal(
        capsys, script.check_seed_file, "dev", tiers["dev"], shifted
    )
    narrower = tmp_path / "narrower.yaml"
    narrower.write_text("dev: [0, 49]\nval: [100, 149]\ntest: [1000, 1099]\n", encoding="utf-8")
    script.check_seed_file(argparse.ArgumentParser(), "dev", script.load_tiers(narrower)["dev"], narrower)
