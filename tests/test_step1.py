"""Operations step 1 (arena/2026-11-25-step1-outcome.md): conditions, the pending-occupancy check, the platform
fingerprint, the test-launch fingerprint check and the step-1 convergence rule."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ocmdp.agents import oc_sac
from ocmdp.agents.oc_sac import SACConfig, train
from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
from ocmdp.envs.wrappers import MechanismObservation

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(learning_starts=48, batch_size=32, hidden_dim=16, dual_update_every=40)


def _script(name: str):
    sys.path.insert(0, str(ROOT / "experiments"))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(ROOT / "experiments"))


@pytest.fixture
def script():
    return _script("train_agent")


@pytest.fixture
def protocol():
    return _script("check_dev_protocol")


def _small_env(condition: str, class_sizes=(2, 2)):
    script = _script("train_agent")
    cfg = ToyConfig(cross_resistance=script.cross_resistance(class_sizes), horizon_quarters=6, decisions_per_quarter=3, **script.ENV_KWARGS)
    toy = ToyOCMDPEnv(cfg)
    return MechanismObservation(toy) if condition == "c-given" else MechanismObservation(toy, mechanism_map=np.eye(toy.coupling.n_drugs))


# --- conditions and the pending occupancy --------------------------------------------------------------------------------------


def test_the_no_c_observation_is_the_raw_drug_indexed_one(script):
    env = script.make_env(0.0, (3, 1), condition="no-c")
    raw = ToyOCMDPEnv(env.unwrapped.config)
    obs, _ = env.reset(seed=4)
    expected, _ = raw.reset(seed=4)
    np.testing.assert_array_equal(obs, expected)
    for _ in range(30):
        obs, *_ = env.step(3)
        expected, *_ = raw.step(3)
        np.testing.assert_array_equal(obs, expected)
    assert env.observation_space.shape == raw.observation_space.shape
    script.assert_pending_occupancy(env)
    with pytest.raises(ValueError, match="condition"):
        script.make_env(0.0, condition="c-hidden")


def test_the_c_given_environment_is_unchanged(script):
    env = script.make_env(0.0)
    assert np.array_equal(env.mechanism_map, env.unwrapped.coupling.matrix)


class _DroppedPending(MechanismObservation):
    def observation(self, obs):
        out = super().observation(obs)
        out[-self.mechanism_map.shape[1] * self.unwrapped.config.delay :] = 0.0
        return out


class _ShiftedPending(MechanismObservation):
    def observation(self, obs):
        out = super().observation(obs)
        width = self.mechanism_map.shape[1] * self.unwrapped.config.delay
        out[-width:] = np.roll(out[-width:], 1)
        return out


@pytest.mark.parametrize("wrapper", [_DroppedPending, _ShiftedPending])
@pytest.mark.parametrize("condition", ["c-given", "no-c"])
def test_a_mutated_pending_block_fails_the_check(script, wrapper, condition):
    toy = ToyOCMDPEnv(script.make_env(0.0, (2, 1), condition=condition).unwrapped.config)
    mapping = None if condition == "c-given" else np.eye(toy.coupling.n_drugs)
    with pytest.raises(RuntimeError, match="prescribing of the last"):
        script.assert_pending_occupancy(wrapper(toy, mechanism_map=mapping))


def test_the_check_refuses_an_observation_without_the_pending_stack(script):
    config = ToyConfig(cross_resistance=script.cross_resistance((2, 2)), markov_observation=False)
    with pytest.raises(RuntimeError, match="pending occupancy"):
        script.assert_pending_occupancy(MechanismObservation(ToyOCMDPEnv(config)))
    with pytest.raises(RuntimeError, match="MechanismObservation"):
        script.assert_pending_occupancy(ToyOCMDPEnv(config))


def test_drug_embedding_defaults_to_the_rows_of_c_and_one_hot_drops_them():
    assert SACConfig().drug_embedding == "mechanism"
    with pytest.raises(ValueError, match="drug_embedding"):
        SACConfig(drug_embedding="learned")
    cfg = dict(total_quarters=60, **TINY)
    given = train(_small_env("c-given", (3, 1)), SACConfig(**cfg), seed=0).agent
    hidden = train(_small_env("no-c", (3, 1)), SACConfig(**cfg, drug_embedding="one_hot"), seed=0).agent
    toy = _small_env("c-given", (3, 1)).unwrapped
    assert torch.equal(given.embeddings, torch.tensor(toy.coupling.matrix, dtype=torch.float32))
    assert torch.equal(hidden.embeddings, torch.eye(4))
    assert torch.equal(given.assignment, hidden.assignment)  # the pushforward entropy uses the same assignment either way


def test_a_no_c_pair_is_bitwise_identical_without_the_entropy_term():
    cfg = dict(total_quarters=400, temperature="fixed", alpha=0.0, drug_embedding="one_hot", **TINY)
    trained = [train(_small_env("no-c", (3, 1)), SACConfig(**cfg, entropy=e), seed=1) for e in ("pushforward", "action")]
    assert trained[0].episode_failure == trained[1].episode_failure
    assert trained[0].parameter_sha256 == trained[1].parameter_sha256
    assert oc_sac.parameter_sha256(trained[0].agent) == oc_sac.parameter_sha256(trained[1].agent)


def test_drug_embedding_cannot_be_set_by_hand(script):
    with pytest.raises(SystemExit):
        script._parse_overrides(argparse.ArgumentParser(), ["drug_embedding=one_hot"], 12_000)


def test_the_condition_is_required():
    result = subprocess.run([sys.executable, "experiments/train_agent.py", "--tier", "dev"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "--condition" in result.stderr


# --- platform fingerprint ------------------------------------------------------------------------------------------------------


def test_the_canary_leaves_training_untouched_and_is_not_a_job(script, monkeypatch):
    monkeypatch.setenv(script.CANARY_ENV, "tiny")
    monkeypatch.setattr(script, "_per_process", {})
    cfg = SACConfig(total_quarters=120, **TINY)
    before = train(_small_env("c-given"), cfg, seed=3)
    state, threads, jobs = torch.get_rng_state(), torch.get_num_threads(), script._jobs_in_process
    canary = script.arithmetic_canary()
    assert torch.equal(torch.get_rng_state(), state) and torch.get_num_threads() == threads and script._jobs_in_process == jobs
    after = train(_small_env("c-given"), cfg, seed=3)
    assert before.parameter_sha256 == after.parameter_sha256
    assert oc_sac.parameter_sha256(before.agent) == oc_sac.parameter_sha256(after.agent)
    assert canary["size"] == "tiny" and len(canary["sha256"]) == 64
    monkeypatch.setattr(script, "_per_process", {})
    assert script.arithmetic_canary() == canary  # the same arithmetic gives the same canary


def test_the_fingerprint_ignores_the_os_description_but_not_the_kernel_settings(script, monkeypatch):
    monkeypatch.setenv(script.CANARY_ENV, "tiny")
    monkeypatch.setattr(script, "_per_process", {})
    first = script.platform_fields()
    canary = {k: v for k, v in script._per_process.items() if k[0] == "canary"}
    monkeypatch.setattr(script, "_per_process", dict(canary))
    monkeypatch.setattr(script.platform, "platform", lambda: "another description")
    second = script.platform_fields()
    assert second["platform"] == "another description" and second["platform_fingerprint"] == first["platform_fingerprint"]
    monkeypatch.setattr(script, "_per_process", dict(canary))
    monkeypatch.setenv("MKL_CBWR", "COMPATIBLE" if first["environment"]["MKL_CBWR"] != "COMPATIBLE" else "AUTO")
    assert script.platform_fields()["platform_fingerprint"] != first["platform_fingerprint"]


def _pair(seed, pid=7, jobs=(1, 2), cpu=("x", "x")):
    return [{"variant": v, "sigma": 0.0, "seed": seed, "host": {"pid": pid, "job_in_process": j, "cpu": c}}
            for v, j, c in zip(("oc-sac", "sac"), jobs, cpu)]  # fmt: skip


def test_pairing_defects_find_split_or_interleaved_pairs(script):
    assert script.pairing_defects(_pair(0) + _pair(1, jobs=(3, 4))) == []
    assert "one job after the other" in script.pairing_defects(_pair(0, jobs=(1, 3)))[0]
    split = _pair(0)
    split[1]["host"]["pid"] = 8
    assert "one process" in script.pairing_defects(split)[0]
    assert "host records differ" in script.pairing_defects(_pair(0, cpu=("x", "y")))[0]
    assert "only" in script.pairing_defects(_pair(0)[:1])[0]


def _refusal(capsys, check, *args) -> str:
    with pytest.raises(SystemExit):
        check(argparse.ArgumentParser(), *args)
    return capsys.readouterr().err


def test_a_test_launch_needs_the_preregistered_fingerprint_and_unchanged_training_code(script, tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "ocmdp").mkdir(parents=True)
    (repo / "ocmdp" / "agent.py").write_text("x = 1\n", encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "val")
    val = git("rev-parse", "HEAD")
    monkeypatch.setattr(script, "ROOT", repo)
    fields = {"platform_fingerprint": "a" * 64, "arithmetic_canary": {"size": "full"}}
    prereg = tmp_path / "planned.yaml"
    prereg.write_text("slug: planned\n", encoding="utf-8")
    assert "platform_fingerprint:" in _refusal(capsys, script.check_test_fingerprint, prereg, fields)
    prereg.write_text(f"platform_fingerprint: {'a' * 64}\nval_commit: {val}\n", encoding="utf-8")
    script.check_test_fingerprint(argparse.ArgumentParser(), prereg, fields)
    assert "full arithmetic canary" in _refusal(capsys, script.check_test_fingerprint, prereg, {**fields, "arithmetic_canary": {"size": "tiny"}})
    assert "is not the preregistered" in _refusal(capsys, script.check_test_fingerprint, prereg, {**fields, "platform_fingerprint": "b" * 64})
    (repo / "notes.md").write_text("notes change nothing a run computes\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "notes")
    script.check_test_fingerprint(argparse.ArgumentParser(), prereg, fields)
    (repo / "ocmdp" / "agent.py").write_text("x = 2\n", encoding="utf-8")
    git("commit", "-q", "-am", "code")
    assert "differs from val_commit" in _refusal(capsys, script.check_test_fingerprint, prereg, fields)
    prereg.write_text(f"platform_fingerprint: {'a' * 64}\nval_commit: {'0' * 40}\n", encoding="utf-8")
    assert "is not a commit" in _refusal(capsys, script.check_test_fingerprint, prereg, fields)


def test_planner_restarts_builds_unequal_classes(tmp_path):
    result = subprocess.run([sys.executable, "experiments/planner_restarts.py", "--class-sizes", "3", "1", "--steps", "3", "--random-starts", "0",
                             "--planner-seeds", "2", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert result.returncode == 0, result.stderr
    (path,) = tmp_path.glob("planner_*.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["args"]["class_sizes"] == [3, 1] and report["n_drugs"] == 4 and len(report["planned_failure_rate"]) == 2


# --- the step-1 rule -----------------------------------------------------------------------------------------------------------

U, R = 0.40, 0.30
PLANNER = {"args": {"class_sizes": [8, 5, 4, 2, 1], "steps": 200, "random_starts": 4, "planner_seeds": 5}, "git": {"commit": "c" * 40, "code_diff_sha256": None},
           "uniform": 0.4, "planned_failure_rate": [0.3] * 5, "spread_share_of_gain_over_uniform": 0.01}  # fmt: skip


def _report(sizes, length, g, diverged=(), diverged_at=2_000):
    runs = []
    for seed, value in enumerate(g):
        for job, variant in enumerate(("oc-sac", "sac")):
            host = {"pid": 100 + seed, "job_in_process": 2 * seed + 1 + job, "cpu": "x",
                    "fingerprint": {"platform_fingerprint": "f" * 64, "arithmetic_canary": {"size": "full"}}}  # fmt: skip
            run = {"variant": variant, "sigma": 0.0, "seed": seed, "host": host, "fingerprint_matches_launch": True}
            if seed in diverged:
                run |= {"diverged": "critic loss is not finite (nan); training diverged", "diverged_at_decision": diverged_at}
            else:
                rate = U - value * (U - R)
                run |= {"expected_failure_rate": [rate, rate],
                        "parameter_sha256": {"initial": f"i{seed}", "after_first_window": f"w{seed}", "final": f"f{seed}-{length}"}}  # fmt: skip
            runs.append(run)
    cfg = {"temperature": "fixed", "alpha": 0.0, "total_quarters": length}
    return {"_path": f"train_{'-'.join(map(str, sizes))}_{length}.json", "report_schema": 3, "condition": "c-given", "platform_fingerprint": "f" * 64,
            "args": {"tier": "dev", "sigmas": [0.0], "seeds": list(range(len(g))), "delay": None, "lam": None, "set": ["temperature=fixed", "alpha=0"]},
            "git": {"commit": "c" * 40, "dirty": False, "code_diff_sha256": None}, "evaluation_episode_seeds": [10_000, 10_001],
            "class_structure": {"class_sizes": list(sizes)}, "sac_config": {"oc-sac": cfg, "sac": cfg}, "pairing_defects": [], "runs": runs,
            "baselines": {"0.0": {"uniform": {"expected_failure_rate": [U, U]}, "oracle[replans every quarter]": {"expected_failure_rate": [R, R]}}}}  # fmt: skip


LEARNED = [0.80, 0.82, 0.79, 0.81, 0.80]


def _six(g_by_length=None, other=None, diverged_by_length=None, diverged_at=2_000):
    """Six reports; `g_by_length` sets g on (8, 5, 4, 2, 1), and (2, 2) converges at 12 000 unless `other` sets it."""
    g_by_length = g_by_length or {}
    other = other or {}
    diverged_by_length = diverged_by_length or {}
    reports = []
    for sizes in ((2, 2), (8, 5, 4, 2, 1)):
        for length in (12_000, 24_000, 48_000):
            source = g_by_length if sizes == (8, 5, 4, 2, 1) else other
            diverged = diverged_by_length.get(length, ()) if sizes == (8, 5, 4, 2, 1) else ()
            reports.append(_report(sizes, length, source.get(length, LEARNED), diverged, diverged_at))
    return reports


def _shift(values, by):
    return [v + by for v in values]


def test_step1_converges_at_12000_and_then_at_24000(protocol):
    lines, result = protocol.step1(_six(), PLANNER)
    assert result["verdict"] == "CONVERGED" and result["length"] == 12_000 and not result["failed"], lines
    lines, result = protocol.step1(_six({12_000: _shift(LEARNED, -0.3)}), PLANNER)
    assert result["verdict"] == "CONVERGED" and result["length"] == 24_000, lines  # still rising 12 -> 24, flat 24 -> 48


def test_step1_needs_the_12_to_48_span_for_12000(protocol):
    # each doubling moves by 0.03, inside the margin, but 12 -> 48 moves by 0.06
    g = {12_000: LEARNED, 24_000: _shift(LEARNED, 0.03), 48_000: _shift(LEARNED, 0.06)}
    lines, result = protocol.step1(_six({k: [x - 0.1 for x in v] for k, v in g.items()}), PLANNER)
    assert result["length"] == 24_000, lines


@pytest.mark.parametrize(
    "g, reason",
    [
        ({12_000: [0.3] * 5, 24_000: [0.3] * 5, 48_000: [0.3] * 5}, "FLAT_BELOW_MAJORITY"),  # stagnation is not convergence
        ({48_000: _shift(LEARNED, -0.3)}, "DECLINING"),
        ({48_000: _shift(LEARNED, 0.15)}, "RISING"),
        ({48_000: [0.80 + d for d in (0.2, -0.2, 0.15, -0.15, 0.0)]}, "UNDECIDED"),  # too wide to call
    ],
)
def test_step1_stops_with_the_reason_and_never_a_default_length(protocol, g, reason):
    g = {12_000: LEARNED, 24_000: LEARNED, **g}
    lines, result = protocol.step1(_six(g), PLANNER)
    assert result["verdict"] == "STOP" and result["length"] is None and result["failed"]
    assert result["reasons"] == {"[8, 5, 4, 2, 1]": reason}, lines
    assert "STOP for the owner" in lines[-1]


def test_step1_counts_diverged_runs_and_stops_on_too_many(protocol):
    lines, result = protocol.step1(_six(diverged_by_length={24_000: (0, 1), 48_000: (0, 1)}, diverged_at=13_000), PLANNER)
    assert result["reasons"] == {"[8, 5, 4, 2, 1]": "DIVERGING"}, lines
    # a divergence before 12 000 decisions must appear in the 12 000 run too: at a fixed temperature runs are prefixes
    lines, result = protocol.step1(_six(diverged_by_length={24_000: (0,), 48_000: (0,)}, diverged_at=5_000), PLANNER)
    assert result["verdict"] == "DEFECT" and any("diverged at decision 5000" in d for d in result["defects"]), lines
    lines, result = protocol.step1(_six(diverged_by_length={48_000: (0,)}, diverged_at=30_000), PLANNER)
    assert result["verdict"] != "DEFECT", lines
    report = _report((2, 2), 12_000, [0.8, -0.2, 0.7, 0.9, 0.6], diverged=(2,))
    g, diverged = protocol._gain_shares(report)
    assert diverged == {2} and g[2] == pytest.approx(-0.2) and g[1] == pytest.approx(-0.2)  # min(0, the lowest other g)
    g, _ = protocol._gain_shares(_report((2, 2), 12_000, LEARNED, diverged=(0,)))
    assert g[0] == 0.0


def _mutated(mutate):
    reports = _six()
    mutate(reports)
    return reports


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rs: rs[1]["runs"][1]["parameter_sha256"].update(final="other"), "first differ at final"),
        (lambda rs: rs[1]["runs"][1]["parameter_sha256"].update(initial="other"), "first differ at initial"),
        (lambda rs: [r["parameter_sha256"].update(after_first_window="other") for r in rs[4]["runs"][:2]], "(b)"),
        (lambda rs: rs[0]["runs"][1].update(diverged="nan", diverged_at_decision=100), "divergence"),
        (lambda rs: [r.update(diverged="nan", diverged_at_decision=d) for r, d in zip(rs[0]["runs"][:2], (100, 200))], "divergence"),
        (lambda rs: rs[0]["runs"][1]["host"].update(job_in_process=9), "pairing"),
        (lambda rs: rs[0]["runs"][1]["host"].update(cpu="y"), "pairing"),
        (lambda rs: rs[3].update(platform_fingerprint="e" * 64), "platform fingerprints"),
        (lambda rs: rs[3]["runs"][0].update(fingerprint_matches_launch=False), "does not match"),
        (lambda rs: [r["host"]["fingerprint"].update(arithmetic_canary={"size": "tiny"}) for r in rs[2]["runs"]], "reduced arithmetic canary"),
        (lambda rs: rs[5]["baselines"]["0.0"]["uniform"].update(expected_failure_rate=[U, U + 1e-9]), "baseline uniform"),
        (lambda rs: rs[5].update(evaluation_episode_seeds=[10_000, 10_002]), "evaluation episodes"),
        (lambda rs: rs[2].update(git={"commit": "d" * 40, "dirty": False, "code_diff_sha256": None}), "different commits"),
        (lambda rs: rs[2].update(runs=rs[2]["runs"][:-2]), "seed sets differ"),
        (lambda rs: [r.update(seed=5) for r in rs[2]["runs"][-1:]] and rs[2].update(pairing_defects=[]), "seed sets differ"),
        (lambda rs: rs[4]["baselines"]["0.0"]["oracle[replans every quarter]"].update(expected_failure_rate=[U, U]), "no scale"),
    ],
)
def test_step1_stops_on_each_defect(protocol, mutate, message):
    lines, result = protocol.step1(_mutated(mutate), PLANNER)
    assert result["verdict"] == "DEFECT" and result["length"] is None, lines
    assert any(message in d for d in result["defects"]), result["defects"]


def test_step1_needs_the_reference_and_the_six_report_design(protocol):
    lines, result = protocol.step1(_six(), {**PLANNER, "spread_share_of_gain_over_uniform": 0.03})
    assert result["verdict"] == "REFERENCE NOT ESTABLISHED" and result["length"] is None
    with pytest.raises(ValueError, match="planner seeds"):
        protocol.step1(_six(), {**PLANNER, "args": {**PLANNER["args"], "planner_seeds": 1}})
    with pytest.raises(ValueError, match="commit and code"):
        protocol.step1(_six(), {**PLANNER, "git": {"commit": "d" * 40, "code_diff_sha256": None}})
    with pytest.raises(ValueError, match="no gain"):
        protocol.step1(_six(), {**PLANNER, "planned_failure_rate": [0.45] * 5})
    mutations = [
        (lambda r: r["args"].update(seeds=[5, 6, 7, 8, 9]), "dev seeds"),
        (lambda r: r.update(condition="no-c"), "c-given"),
        (lambda r: r["args"].update(delay=2), "default delay"),
        (lambda r: r["git"].update(dirty=True), "uncommitted"),
        (lambda r: r["sac_config"].update(sac={**r["sac_config"]["sac"], "dual_lr": 0.1}), "configuration"),
        (lambda r: r["sac_config"].update(sac={**r["sac_config"]["sac"], "total_quarters": 1}), "different lengths"),
    ]
    for mutate, message in mutations:
        reports = _six()
        mutate(reports[3])
        with pytest.raises(ValueError, match=message):
            protocol.step1(reports, PLANNER)
    with pytest.raises(ValueError, match="class sizes"):
        protocol.step1(_six(), {**PLANNER, "args": {**PLANNER["args"], "class_sizes": [5, 5, 5, 5]}})
    with pytest.raises(ValueError, match="random starts"):
        protocol.step1(_six(), {**PLANNER, "args": {**PLANNER["args"], "random_starts": 0}})
    with pytest.raises(ValueError, match="exactly one report"):
        protocol.step1(_six()[:5], PLANNER)
    with pytest.raises(ValueError, match="two reports"):
        protocol.step1(_six() + [_six()[0]], PLANNER)
    old = _six()
    old[0]["report_schema"] = 2
    with pytest.raises(ValueError, match="schema"):
        protocol.step1(old, PLANNER)
    warm = _six()
    warm[0]["sac_config"]["sac"] = {**warm[0]["sac_config"]["sac"], "alpha": 0.1}
    with pytest.raises(ValueError, match="temperature-0"):
        protocol.step1(warm, PLANNER)


def test_step1_cli_exits_by_verdict(tmp_path):
    paths = []
    for report in _six():
        path = tmp_path / report.pop("_path")
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(str(path))
    planner = tmp_path / "planner.json"
    planner.write_text(json.dumps(PLANNER), encoding="utf-8")
    run = [sys.executable, "experiments/check_dev_protocol.py", "step1"]
    converged = subprocess.run([*run, "--planner", str(planner), *paths], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert converged.returncode == 0 and json.loads(converged.stdout.splitlines()[-1])["length"] == 12_000, converged.stderr
    missing = subprocess.run([*run, *paths], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert missing.returncode == 1 and "--planner" in missing.stderr

