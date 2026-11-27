"""Train occupancy-coupled SAC and standard SAC across seeds and compare them with the baselines.

    python experiments/train_agent.py --tier dev --condition c-given                 # 5 seeds x 2 variants x 2 noise levels, in parallel
    python experiments/train_agent.py --tier dev --condition no-c --seeds 0 1 --workers 4   # agents not given C
    python experiments/train_agent.py --tier dev --condition c-given --backend modal   # one Modal container per job
    python experiments/train_agent.py --tier val --condition c-given --platform modal-a --set temperature=fixed alpha=0.1   # one temperature candidate
    python experiments/train_agent.py --tier val --condition c-given --platform modal-a --checkpoint-every 3000   # also save each agent at 3000, 6000, 9000
    python experiments/train_agent.py --tier dev --condition c-given --class-sizes 8 5 4 2 1   # 20 drugs in 5 unequal mechanism classes
    python experiments/train_agent.py --tier test --condition c-given --platform rtx-1 --seeds 1000 1001 ... --prereg preregistrations/<file>.yaml

Seeds come in three disjoint tiers, read from seeds.yaml: dev for exploration, val for
choosing hyperparameters, test for the one confirmatory run of a preregistration. A
tier fixes both the training seeds a run may use and the episodes it is evaluated on,
so nothing tuned on dev or val has seen a test episode. Every policy in a run is
evaluated on the same episodes (jittered starting states), so differences between
policies are paired. Agents are evaluated both sampling from their policy and taking
its most probable drug. Writes a JSON report to runs/train/, which records the tier,
the preregistration, the git commit, the instance and, per trained agent, its reachable
entropy range and temperature.

Both agents of a seed train in one job, one after the other, so a pair always shares a
process or container and its arithmetic. Every trained agent is saved as a checkpoint
(ocmdp.agents.oc_sac.load_checkpoint reads it), on whichever backend it trained, under
checkpoints/<tier>/<report name>/ next to the report; the report records each file's
path relative to its own folder, its SHA-256 and the training decisions it had seen.
Only test-tier checkpoints are tracked in git; dev and val ones are archived with their
hashes, and experiments/verify_checkpoints.py checks any folder against its report.
Intermediate checkpoints (--checkpoint-every) are for checkpoint choice on val only, at
most three per agent.

--condition says what the agents are given. "c-given": every drug-indexed block of the
observation is mapped through the true cross-resistance matrix C, and each drug enters
the networks as its row of C. "no-c": the observation stays drug-indexed and each drug
enters as a one-hot vector, so nothing of C reaches the agent's inputs. Either way the
environment is checked to carry the pending occupancy before anything trains
(assert_pending_occupancy). Baselines see the environment, not an observation, so they
are the same in both conditions.

Every launch writes launches/<launch id>.json next to its reports before any job starts,
so a launch that fails or is killed still leaves a record, and the report carries the
same id. --platform labels the account or machine (required for val and test); the
label is recorded but decides nothing. What decides whether two launches share
arithmetic is the platform fingerprint (platform_fields): the interpreter, CPU, torch
and numpy builds, the installed distributions, the environment variables that steer
the kernels, and an arithmetic canary, four short trainings whose final weights a
different rounding would change. The driver computes it before the launch record is
written (on Modal, in a probe container), every job computes its own, and each job also
records its host and hashes of the agent's weights at initialisation, after the first
update window and at the end: training is chaotic, so runs agree seed by seed only
where all of these agree. Both agents of a seed must have trained in one process, one
job after the other, with equal host records; the report lists every pair that did not
(`pairing_defects`) and the launch then exits 1.

A launch on the test tier is refused unless its preregistration is committed and
unmodified, approvals/<preregistration name>.ok exists, the code is committed, and no
earlier launch used that preregistration's test seeds: the rules of the repository's
tier guard, enforced here so they hold on any machine. The preregistration also carries
the fingerprint its val selection ran on (a `platform_fingerprint:` line) and the commit
val ran from (a `val_commit:` line). A test launch is refused unless this platform's
fingerprint equals it, the training code is unchanged since that commit, and the canary
is the full one; a test job whose own fingerprint differs from the launch's stops. A test
launch also lists its seeds explicitly, at least ten of them, exactly as the
preregistration's `seeds:` line names them, and none used by an earlier test launch of
this script under any preregistration. It writes into a folder directly under runs/,
where every later launch looks for earlier test launch records.

The instance is one mechanism class per entry of --class-sizes, each holding that many
interchangeable drugs; the default, two classes of two, is PairedDrugs. The report
carries the class sizes, their imbalance KL (log M - H(n / K), zero for equal classes)
and, per trained agent, the entropy its policy spends inside the classes.

Diagnostics of why an agent trails the planner. Every policy's failure rate is also
reported discounted at the agent's gamma, the objective the agent is trained on. Without
noise, each agent's critics and actor are checked against counterfactual values of every
class under the agent's own continuation (`critic`), and with --diagnose-gap its gap to
the planner is split by cause along its own trajectory (`gap_attribution`). Each agent's
temperature path, dual-variable path and temperature at evaluation are logged.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ENV_KWARGS = dict(expected_reward=True, markov_observation=True, r0_spread=0.04, mu0_spread=0.5)
ROOT = Path(__file__).resolve().parents[1]
TIERS = ("dev", "val", "test")
DEFAULT_SEEDS = 5  # per variant and noise level when --seeds is not given: the first seeds of the tier
MIN_TEST_SEEDS = 10  # a test launch names at least this many seeds, the repository's tier guard's minimum
REPORT_SCHEMA = 3  # 3 adds the condition, platform fingerprints and pairing checks, and counts runs_learned against the every-quarter replanner with diverged runs counted
MODAL_ENV = ("MODAL_TASK_ID", "MODAL_REGION", "MODAL_CLOUD_PROVIDER", "MODAL_IMAGE_ID")  # set inside Modal containers
_jobs_in_process = 0  # jobs this process (a worker, or a reused Modal container) has started
VARIANTS = {"oc-sac": {"entropy": "pushforward"}, "sac": {"entropy": "action"}}
PAIRED_DRUGS = (2, 2)  # class sizes of the default instance
REPLANNER = "oracle[replans every quarter]"  # the planner to compare against under noise, where none is a bound
CONDITIONS = {"c-given": "mechanism", "no-c": "one_hot"}  # condition -> SACConfig.drug_embedding
JOB_THREADS = 1  # torch threads every job and the canary train with
FINGERPRINT_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_CBWR", "ATEN_CPU_CAPABILITY",
                   "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")  # fmt: skip
CANARY_ENV = "OCMDP_CANARY"  # "full" unless set; "tiny" keeps the test suite fast, and a test launch refuses it
CANARY_SIZES = {
    "full": {"total_quarters": 2_000},
    "tiny": {"total_quarters": 240, "learning_starts": 48, "batch_size": 32, "hidden_dim": 16, "dual_update_every": 40},
}
CANARY_SEED = 0  # the first dev seed; main() checks it lies in the dev tier
VAL_CODE_PATHS = ("ocmdp", "experiments/train_agent.py", "experiments/stats.py")  # must not change between val and test
_per_process: dict[tuple[str, int], dict] = {}  # (what, pid) -> value: computed once per process, never inherited by a fork


def cross_resistance(class_sizes: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """One mechanism per class, selected for only by the drugs of that class."""
    return tuple(tuple(float(j == m) for j in range(len(class_sizes))) for m, size in enumerate(class_sizes) for _ in range(size))


def make_env(sigma: float, class_sizes: tuple[int, ...] = PAIRED_DRUGS, delay: int | None = None, lam: float | None = None,
             condition: str = "c-given"):  # fmt: skip
    """The training environment; delay and lam fall back to ToyConfig's defaults when None.

    Under "c-given" every drug-indexed block of the observation is mapped through the true
    C; under "no-c" the identity map leaves it drug-indexed. The pending occupancy is
    checked to be in the observation before the environment is returned.
    """
    from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv
    from ocmdp.envs.wrappers import MechanismObservation

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition!r}")
    coupling = {name: value for name, value in (("delay", delay), ("lam", lam)) if value is not None}
    config = ToyConfig(cross_resistance=cross_resistance(class_sizes), resistance_noise=sigma, **coupling, **ENV_KWARGS)
    toy = ToyOCMDPEnv(config)
    env = MechanismObservation(toy) if condition == "c-given" else MechanismObservation(toy, mechanism_map=np.eye(toy.coupling.n_drugs))
    assert_pending_occupancy(env)
    return env


def assert_pending_occupancy(env) -> None:
    """Raise RuntimeError unless the observation ends with the pending occupancy, the last `delay` quarters of prescribing.

    The kernel sees prescribing only `delay` quarters late, so without that block the
    observation does not determine the next state. Checked on a copy, over delay + 1
    quarters with a different drug each quarter: after each, the observation must have the
    shape the wrapper's map implies, and its trailing block must equal the one-hot shares
    of the drugs given in the last `delay` quarters, oldest first and front-padded, mapped
    like every drug-indexed block, and nonzero when delay >= 1. The expected block is built
    from the drugs given, not from the environment's own record. A RuntimeError rather than
    an assert, so the check also runs under python -O.
    """
    from ocmdp.envs.wrappers import MechanismObservation

    if not isinstance(env, MechanismObservation):
        raise RuntimeError(f"the training environment must be a MechanismObservation over the toy environment, got {type(env).__name__}")
    probe = copy.deepcopy(env)
    toy = probe.unwrapped
    delay, n_drugs, mapping = toy.config.delay, toy.coupling.n_drugs, probe.mechanism_map
    width = mapping.shape[1]
    dim = toy.coupling.n_mechanisms + width + 1 + width * (1 + delay)
    if not toy.config.markov_observation or probe.observation_space.shape != (dim,):
        raise RuntimeError(f"the observation must carry the pending occupancy: expected shape ({dim},), got "
                           f"{probe.observation_space.shape} with markov_observation={toy.config.markov_observation}")  # fmt: skip
    obs, info = probe.reset(seed=0)
    given = []  # one drug per quarter, a different one each quarter, so each row of the block is identifiable
    for q in range(min(delay + 1, toy.config.horizon_quarters - 1)):
        drug = q % n_drugs
        while info["quarter"] == q:
            obs, _, _, _, info = probe.step(drug)
        given.append(drug)
        if obs.shape != (dim,):
            raise RuntimeError(f"the observation has shape {obs.shape}, not ({dim},)")
        pending = np.zeros((delay, n_drugs))
        recent = given[-delay:] if delay else []
        for row, d in enumerate(recent):
            pending[delay - len(recent) + row, d] = 1.0
        expected = np.clip(pending.astype(np.float32) @ mapping, 0.0, 1.0).astype(np.float32).ravel()
        observed = obs[dim - delay * width :]
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"after quarter {q + 1}, the trailing block of the observation is not the prescribing of the last {delay} quarters")
        if delay >= 1 and not observed.any():
            raise RuntimeError("the pending block of the observation is zero after a quarter of prescribing")


def imbalance_kl(class_sizes: tuple[int, ...]) -> float:
    """KL from the mechanism distribution of a uniform policy over drugs to the uniform one: log M - H(n / K)."""
    share = np.asarray(class_sizes, dtype=float) / sum(class_sizes)
    return float(np.log(len(share)) + np.sum(share * np.log(share)))


def load_tiers(path: Path) -> dict[str, range]:
    """Read `tier: [first, last]` lines (inclusive) into seed ranges; the three tiers must be present and disjoint."""
    found = re.findall(r"^(\w+):\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", path.read_text(encoding="utf-8"), re.M)
    tiers = {name: range(int(first), int(last) + 1) for name, first, last in found}
    if set(tiers) != set(TIERS) or len(found) != len(TIERS) or any(len(r) == 0 for r in tiers.values()):
        raise ValueError(f"{path} must define a non-empty [first, last] range for each of {', '.join(TIERS)}, once")
    ordered = sorted(tiers.values(), key=lambda r: r.start)
    if any(a.stop > b.start for a, b in zip(ordered, ordered[1:])):
        raise ValueError(f"the seed tiers in {path} overlap")
    return tiers


CODE_PATHS = ("ocmdp", "experiments", "pyproject.toml", "seeds.yaml")  # what a run executes or reads


def git_state() -> dict:
    """Commit the run was launched from, whether the code differs from it, and a hash of the difference.

    Only CODE_PATHS count: notes, gathered material and earlier reports change nothing a
    run computes. Two runs with the same commit and the same `code_diff_sha256` ran the
    same code, tracked edits and untracked files included.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    try:
        diff = git("diff", "HEAD", "--", *CODE_PATHS)
        untracked = sorted(name for name in git("ls-files", "-z", "--others", "--exclude-standard", "--", *CODE_PATHS).split("\0") if name)
        digest = hashlib.sha256(diff.encode())
        for name in untracked:
            digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
        dirty = bool(diff or untracked)
        return {
            "commit": git("rev-parse", "HEAD").strip(),
            "dirty": dirty,
            "code_diff_sha256": digest.hexdigest() if dirty else None,
            "untracked_code": untracked,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "code_diff_sha256": None, "untracked_code": None}


def _cpu() -> str:
    """The CPU model the operating system reports, which a sandboxed container may hide."""
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
    except OSError:
        pass
    return cpu


def _numpy_cpu_features() -> list[str] | None:
    """The CPU features numpy detected for itself and dispatches on."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return None
    return sorted(name for name, enabled in __cpu_features__.items() if enabled)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def installed_distributions() -> list[str]:
    """Every installed distribution as name==version, sorted: the whole environment a run imports from."""
    from importlib.metadata import distributions

    return sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})


def arithmetic_canary() -> dict:
    """Four short trainings whose final weights stand for this process's arithmetic.

    OC-SAC and SAC, in both conditions, on PairedDrugs without noise, from dev seed 0, at
    a learned temperature so that the entropy and temperature updates run too. Training is
    chaotic, so two processes whose kernels round differently end with different weights.
    Computed once per process, under a forked torch generator and at JOB_THREADS threads,
    so a job's own training is the same with or without it, and not counted as a job.
    """
    key = ("canary", os.getpid())
    if key in _per_process:
        return _per_process[key]
    import torch

    from ocmdp.agents.oc_sac import SACConfig, parameter_sha256, train

    size = os.environ.get(CANARY_ENV, "full")
    if size not in CANARY_SIZES:
        raise ValueError(f"{CANARY_ENV} must be one of {sorted(CANARY_SIZES)}, got {size!r}")
    finals = []
    threads = torch.get_num_threads()
    torch.set_num_threads(JOB_THREADS)
    try:
        with torch.random.fork_rng(devices=[]):
            for condition, embedding in CONDITIONS.items():
                for variant in VARIANTS:
                    cfg = SACConfig(**CANARY_SIZES[size], **VARIANTS[variant], drug_embedding=embedding)
                    try:
                        finals.append(parameter_sha256(train(make_env(0.0, PAIRED_DRUGS, condition=condition), cfg, CANARY_SEED).agent))
                    except FloatingPointError as exc:
                        finals.append(f"diverged: {exc}")
    finally:
        torch.set_num_threads(threads)
    _per_process[key] = {"size": size, "decisions": CANARY_SIZES[size]["total_quarters"], "sha256": _sha256_text("\n".join(finals))}
    return _per_process[key]


def platform_fields() -> dict:
    """What decides this process's arithmetic, and `platform_fingerprint`, a SHA-256 over all of it except `platform`.

    `platform`, the operating system's own description, is recorded for reading but not
    hashed: it changes with updates that leave the arithmetic alone. `torch_threads` is
    the count jobs train with. Computed once per process.
    """
    key = ("platform", os.getpid())
    if key in _per_process:
        return _per_process[key]
    import gymnasium
    import torch

    fields = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu": _cpu(),
        "numpy_cpu_features": _numpy_cpu_features(),
        "simd": torch.backends.cpu.get_cpu_capability(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "torch_config_sha256": _sha256_text(torch.__config__.show()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_threads": JOB_THREADS,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numpy": np.__version__,
        "numpy_config_sha256": _sha256_text(json.dumps(np.show_config(mode="dicts"), sort_keys=True, default=str)),
        "gymnasium": gymnasium.__version__,
        "distributions_sha256": _sha256_text("\n".join(installed_distributions())),
        "environment": {name: os.environ.get(name) for name in FINGERPRINT_ENV},
        "arithmetic_canary": arithmetic_canary(),
    }
    _per_process[key] = {"platform": platform.platform(), **fields, "platform_fingerprint": _sha256_text(json.dumps(fields, sort_keys=True))}
    return _per_process[key]


def platform_probe(job: tuple = ()) -> dict:
    """This process's platform fields, for a driver whose jobs run on another backend."""
    import torch

    torch.set_num_threads(JOB_THREADS)
    return platform_fields()


def host(job_in_process: int | None = None) -> dict:
    """The machine and process a job ran on. Training is chaotic, so results reproduce only where the arithmetic is identical.

    `simd` is the level torch dispatches to, which a pinned environment forces; `cpu` is
    the model the operating system reports, which a sandboxed container may hide, so the
    CPU features numpy detects for itself are recorded too. `fingerprint` is the
    process's platform_fields. The machine itself is named only by the launch's
    --platform label: a host name can carry a person's name. `job_in_process` counts the
    jobs one process or reused container has started, since reuse decides which jobs
    share arithmetic.
    """
    import torch

    return {
        "machine": platform.machine(),
        "cpu": _cpu(),
        "simd": torch.backends.cpu.get_cpu_capability(),
        "numpy_cpu_features": _numpy_cpu_features(),
        "torch_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "pid": os.getpid(),
        "job_in_process": job_in_process,
        "modal": {name: os.environ[name] for name in MODAL_ENV if name in os.environ},
        "fingerprint": platform_fields(),
    }


def pairing_defects(runs: list[dict]) -> list[str]:
    """Every (noise level, seed) whose two agents did not train in one process, one job after the other, on one host.

    Both runs of a pair must share a pid, have consecutive job counts in VARIANTS order
    and otherwise equal host records; anything else means the pair may not have shared
    its arithmetic.
    """
    by_pair: dict[tuple, dict[str, dict]] = {}
    for run in runs:
        by_pair.setdefault((run["sigma"], run["seed"]), {})[run["variant"]] = run
    defects = []
    for (sigma, seed), pair in sorted(by_pair.items()):
        if set(pair) != set(VARIANTS):
            defects.append(f"sigma {sigma}, seed {seed}: runs of {sorted(pair)} only")
            continue
        first, second = (pair[v]["host"] for v in VARIANTS)
        if first["pid"] != second["pid"] or second["job_in_process"] != first["job_in_process"] + 1:
            defects.append(f"sigma {sigma}, seed {seed}: not one process, one job after the other")
        elif {k: v for k, v in first.items() if k != "job_in_process"} != {k: v for k, v in second.items() if k != "job_in_process"}:
            defects.append(f"sigma {sigma}, seed {seed}: host records differ")
    return defects


def check_seed_file(parser: argparse.ArgumentParser, tier_name: str, tier: range, seeds_file: Path) -> None:
    """Stop a launch whose seed file would relabel the repository's seeds of one tier as another's.

    --seeds-file may define its own ranges, but no tier of it may overlap a different tier
    of the repository's seeds.yaml, so val or test seeds never run as dev; a test-tier
    launch must use that file itself.
    """
    repository = load_tiers(ROOT / "seeds.yaml")
    if tier_name == "test" and seeds_file.resolve() != (ROOT / "seeds.yaml").resolve():
        parser.error("the test tier must use the repository's seeds.yaml")
    for name, own in load_tiers(seeds_file).items():
        for other, theirs in repository.items():
            if other != name and own.start < theirs.stop and theirs.start < own.stop:
                parser.error(f"the {name} tier of {seeds_file.name} overlaps the repository's {other} seeds [{theirs.start}, {theirs.stop - 1}]")


def earlier_test_launches(slug: str, out: Path) -> list[str]:
    """Ids of earlier test-tier launches of preregistration `slug`, from the launch records under runs/ and `out`."""
    found = []
    for record in sorted({*ROOT.glob("runs/*/launches/*.json"), *out.glob("launches/*.json")}):
        earlier = json.loads(record.read_text(encoding="utf-8"))
        if earlier.get("tier") == "test" and earlier.get("prereg_slug") == slug:
            found.append(earlier.get("launch_id"))
    return found


def check_test_launch(parser: argparse.ArgumentParser, prereg: Path | None, git: dict, out: Path, platform_label: str | None,
                      fingerprinted: bool = False) -> str:  # fmt: skip
    """The preregistration's name, if a launch on the test tier may go ahead; otherwise stop with the reason.

    The name is the file's own, read after resolving the path, and a link is refused, so
    one preregistration's approval cannot be borrowed by another. With `fingerprinted`
    the platform is decided by check_test_fingerprint and the label is not compared;
    otherwise the preregistration's `platform:` line must equal the label.
    """
    if prereg is None or not prereg.is_file():
        parser.error("the test tier needs --prereg pointing at an existing preregistration file")
    if prereg.is_symlink():
        parser.error("--prereg must name the preregistration file itself, not a link to it")
    resolved = prereg.resolve()
    try:
        relative = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        parser.error(f"--prereg {prereg} must lie inside the repository")
    if resolved.parent != (ROOT / "preregistrations").resolve():
        parser.error(f"--prereg must be a file in preregistrations/, got {relative}")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True).returncode == 0
    unmodified = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT, capture_output=True).returncode == 0
    if not (tracked and unmodified):
        parser.error(f"the test tier needs a committed, unmodified preregistration; {relative} is not")
    slug = resolved.stem
    if not fingerprinted:
        named = re.search(r"^platform:\s*(\S+)\s*$", resolved.read_text(encoding="utf-8"), re.M)
        if named is None:
            parser.error(f"{relative} must name the platform its val selection ran on, in a `platform:` line")
        if named.group(1) != platform_label:
            parser.error(f"{relative} was preregistered for platform {named.group(1)}, not {platform_label}")
    if git["dirty"] is not False:
        parser.error("the test tier needs a clean, committed tree, so the report names the exact code that ran")
    used = earlier_test_launches(slug, out)
    if used:
        parser.error(f"the test seeds of {slug} were already used by launch {used[0]}")
    if not (ROOT / "approvals" / f"{slug}.ok").is_file():
        parser.error(f"the test tier needs approvals/{slug}.ok, which only the project owner creates")
    return slug


def check_test_out(parser: argparse.ArgumentParser, out: Path) -> None:
    """Stop a test-tier launch that would write outside a folder directly under runs/.

    Earlier test launches are found from the launch records under runs/*/launches, so a
    record written anywhere else would let a second launch of the same preregistration
    through.
    """
    if out.resolve().parent != (ROOT / "runs").resolve():
        parser.error(f"--tier test writes into a folder directly under runs/ (such as runs/train), not {out}")


def check_test_seeds(parser: argparse.ArgumentParser, prereg: Path | None, seeds: list[int] | None, out: Path) -> None:
    """Stop a test-tier training launch whose seeds are not the preregistration's own, unused ones.

    The seeds must be given explicitly, at least ten of them, and match as a set the
    preregistration's `seeds:` line (integers separated by spaces or commas, brackets
    allowed). No seed may appear in an earlier test-tier launch of this script, whatever
    its preregistration, so test preregistrations train on disjoint seeds.
    """
    if seeds is None:
        parser.error("the test tier needs --seeds listing the preregistration's test seeds explicitly")
    if len(set(seeds)) < MIN_TEST_SEEDS:
        parser.error(f"the test tier needs at least {MIN_TEST_SEEDS} distinct seeds, got {len(set(seeds))}")
    if prereg is not None and prereg.is_file():
        named = re.findall(r"^seeds:[ \t]*\[?([0-9, \t]*)\]?[ \t]*$", prereg.read_text(encoding="utf-8"), re.M)
        if len(named) != 1 or not named[0].strip(" \t,"):
            parser.error(f"{prereg.name} must list its test seeds in exactly one `seeds:` line")
        listed = {int(s) for s in re.split(r"[, \t]+", named[0].strip(" \t,"))}
        if listed != set(seeds):
            parser.error(f"--seeds differ from the `seeds:` line of {prereg.name}: {sorted(listed)}")
    for record in sorted({*ROOT.glob("runs/*/launches/*.json"), *out.glob("launches/*.json")}):
        earlier = json.loads(record.read_text(encoding="utf-8"))
        if earlier.get("tier") == "test" and str(earlier.get("launch_id", "")).startswith("train_"):
            reused = sorted(set(seeds) & set(earlier.get("args", {}).get("seeds") or []))
            if reused:
                parser.error(f"test seeds {reused} were already used by launch {earlier.get('launch_id')}")


def check_test_fingerprint(parser: argparse.ArgumentParser, prereg: Path, fields: dict) -> None:
    """Stop a test launch unless it runs on the platform and the training code its val selection ran on.

    The preregistration names both: a `platform_fingerprint:` line, the fingerprint every
    job of its val reports carried, and a `val_commit:` line, the commit val ran from.
    VAL_CODE_PATHS must not differ between that commit and HEAD, and the canary must be
    the full one.
    """
    text = prereg.resolve().read_text(encoding="utf-8")
    fingerprint = re.search(r"^platform_fingerprint:\s*([0-9a-f]{64})\s*$", text, re.M)
    commit = re.search(r"^val_commit:\s*([0-9a-f]{7,40})\s*$", text, re.M)
    if fingerprint is None or commit is None:
        parser.error(f"{prereg.name} must carry a `platform_fingerprint:` line (64 hex digits) and a `val_commit:` line")
    if fields["arithmetic_canary"]["size"] != "full":
        parser.error(f"the test tier needs the full arithmetic canary; {CANARY_ENV} is {fields['arithmetic_canary']['size']!r}")
    if fields["platform_fingerprint"] != fingerprint.group(1):
        parser.error(f"this platform's fingerprint {fields['platform_fingerprint']} is not the preregistered {fingerprint.group(1)}")
    diff = subprocess.run(["git", "diff", "--quiet", commit.group(1), "HEAD", "--", *VAL_CODE_PATHS], cwd=ROOT, capture_output=True)
    if diff.returncode == 1:
        parser.error(f"the training code ({', '.join(VAL_CODE_PATHS)}) differs from val_commit {commit.group(1)}")
    if diff.returncode != 0:
        parser.error(f"val_commit {commit.group(1)} is not a commit of this repository")


def training_jobs(args: argparse.Namespace, overrides: dict, first_episode: int, instance: tuple, fingerprint: str) -> list[tuple]:
    """One train_and_evaluate_pair job per noise level and seed: both agents of a seed in one process or container.

    Containers and workers can differ in arithmetic, and training is chaotic, so a pair
    split across two of them would differ by more than its entropy space. Each job carries
    the launch's fingerprint, and on the test tier a job whose own fingerprint differs stops.
    """
    return [(s, seed, args.quarters, args.episodes, overrides, first_episode, instance, args.diagnose_gap, args.checkpoint_every,
             args.condition, fingerprint, args.tier == "test") for s in args.sigmas for seed in args.seeds]  # fmt: skip


def train_and_evaluate_pair(job: tuple) -> list[dict]:
    """Both variants of one (noise level, seed), trained and evaluated one after the other in this process."""
    sigma, seed, *rest = job
    return [train_and_evaluate((variant, sigma, seed, *rest)) for variant in VARIANTS]


def launch_arguments(args: argparse.Namespace) -> dict:
    """The command-line arguments as recorded: paths inside the repository relative to it, so no machine's directories appear.

    Paths are written with forward slashes, so a report reads the same whichever platform wrote it.
    """
    def recorded(value):
        if isinstance(value, Path):
            resolved = value.resolve()
            return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else f"<outside the repository>/{value.name}"
        return value

    return {k: recorded(v) for k, v in vars(args).items()}


def prereg_digest(prereg: Path | None) -> str | None:
    """SHA-256 of the preregistration, read at launch so the record names the file as it stood when the run began.

    A preregistration committed and unchanged against HEAD is hashed from its committed blob, so
    the digest depends on the commit alone and not on how the checkout converted line endings.
    Any other file (uncommitted, modified, or outside the repository) is hashed from its bytes on disk.
    """
    if prereg is None:
        return None
    resolved = prereg.resolve()
    if resolved.is_relative_to(ROOT):
        relative = resolved.relative_to(ROOT).as_posix()
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT, capture_output=True).returncode == 0
        unchanged = tracked and subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT, capture_output=True).returncode == 0
        if unchanged:
            blob = subprocess.run(["git", "cat-file", "blob", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
            return hashlib.sha256(blob).hexdigest()
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def record_launch(out: Path, launch_id: str, args: dict, git: dict, backend: str, prereg_slug: str | None, prereg_sha256: str | None,
                  fields: dict | None = None) -> None:  # fmt: skip
    """Write launches/<launch_id>.json before any job starts, so a launch that never finishes still leaves a record.

    With `fields` (the driver's or the probe's platform_fields) the record also carries
    them and every distribution installed where the driver runs.
    """
    folder = out / "launches"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"launch_id": launch_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tier": args["tier"], "platform": args["platform"],
              "backend": backend, "prereg_slug": prereg_slug, "prereg_sha256": prereg_sha256, "git": git, "args": args}  # fmt: skip
    if fields is not None:
        record |= {"platform_fields": fields, "installed_distributions": installed_distributions()}
    (folder / f"{launch_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")


def _rollouts(env, policy, episodes: int, first_episode: int) -> list:
    from ocmdp.eval.rollout import rollout
    from ocmdp.eval.seeds import eval_seed

    return [rollout(env, policy, eval_seed(first_episode + e)) for e in range(episodes)]


def quarterly_failure(log) -> np.ndarray:
    """(Q,) mean failure probability of the drugs given in each quarter of an episode."""
    given = log.fail_prob[log.decision_quarter, log.actions]
    return np.bincount(log.decision_quarter, weights=given) / np.bincount(log.decision_quarter)


def discounted_failure_rate(failure: np.ndarray, gamma: float) -> float:
    """Failure rate with quarter q weighted by gamma**q: what an agent discounting at gamma is trained to lower."""
    weights = gamma ** np.arange(len(failure))
    return float(weights @ failure / weights.sum())


def _episode_stats(env, logs: list, gamma: float) -> dict:
    from ocmdp.eval.metrics import expected_failure_rate, occupancy_entropy, pushforward_occupancy_entropy

    assignment = env.unwrapped.coupling.mechanism_assignment()
    return {
        "expected_failure_rate": [expected_failure_rate(log) for log in logs],
        "discounted_failure_rate": [discounted_failure_rate(quarterly_failure(log), gamma) for log in logs],
        "final_fail_prob": [log.fail_prob[-1].tolist() for log in logs],  # per drug at the horizon: any efficacy threshold applies later
        "H_action": float(np.mean([occupancy_entropy(log.occupancy) for log in logs])),
        "H_pushforward": float(np.mean([pushforward_occupancy_entropy(log.occupancy, assignment) for log in logs])),
    }


def _policy_entropy(probs: np.ndarray, classes: list[tuple[int, ...]]) -> dict:
    """Mean over the visited decisions of the policy's entropy over drugs and of the part spent inside classes."""
    import torch

    from ocmdp.objectives.pushforward_entropy import categorical_entropy, within_class_entropy

    probs = torch.as_tensor(probs)
    return {"action": categorical_entropy(probs).mean().item(), "within_class": within_class_entropy(probs, classes).mean().item()}


def soft_returns(reward: np.ndarray, bonus: np.ndarray, gamma: float) -> np.ndarray:
    """(E, Q) per-quarter rewards and entropy bonuses alpha H -> (E, Q) realised soft return from each decision.

    The return the reward critics are trained to predict for the drug chosen at quarter
    q: that quarter's reward, then every later quarter's reward plus its entropy bonus,
    discounted, with nothing beyond the horizon.
    """
    out = np.zeros_like(reward, dtype=float)
    out[:, -1] = reward[:, -1]
    for q in range(reward.shape[1] - 2, -1, -1):
        out[:, q] = reward[:, q] + gamma * (bonus[:, q + 1] + out[:, q + 1])
    return out


def _continuation_return(snapshot, drug: int, agent, mask_rng: np.random.Generator) -> float:
    """Soft return of giving `drug` for one quarter from `snapshot`, then following the agent's most probable drug.

    Runs on a copy of the environment, so with noise every counterfactual meets the same
    shocks (common random numbers) and differences between drugs are paired.
    """
    import torch

    from ocmdp.components.constraint_filter import sample_feasible

    cfg = agent.cfg
    env = copy.deepcopy(snapshot)
    state = env.unwrapped.state()
    quarter, mask, n_drugs = state["quarter"], state["action_mask"], env.action_space.n
    rewards, bonuses, done = [], [0.0], False
    while not done:
        total, n = 0.0, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(sample_feasible(np.eye(n_drugs)[drug], mask, mask_rng))
            total, n, done, mask = total + reward, n + 1, terminated or truncated, info["action_mask"]
            if info["quarter"] != quarter:
                break
        rewards.append(cfg.reward_scale * total / n)
        quarter = info["quarter"]
        if not done:
            with torch.no_grad():
                probs = agent.policy(torch.as_tensor(obs, dtype=torch.float32)[None])
                bonuses.append(float(agent.temperature() * agent.entropy(probs)[0]))
            drug = int(probs[0].argmax())
    return float(soft_returns(np.array([rewards]), np.array([bonuses]), cfg.gamma)[0, 0])


def _critic_check(agent, snapshots: list, observations: np.ndarray, classes: list[tuple[int, ...]]) -> dict:
    """Do the reward critics, and the actor, rank the classes the way the environment does?

    At every quarterly decision of the sampled evaluation, each class (one drug stands for
    its class: drugs in a class are interchangeable) is tried for one quarter on a copy of
    the environment, then the agent's most probable drug is followed to the horizon. The
    soft return that follows is the counterfactual value of that class under the agent's
    own continuation; the critics estimate it for the sampling policy, which differs only
    as far as the policy is not deterministic.

    Values are in units of the scaled reward. Regret is the counterfactual value lost by
    the class the critics (or the actor) rank first, against the best class; the mean gap
    between the best and second-best class is the scale it is read against.
    """
    import torch

    rng = np.random.default_rng(0)  # only used to resolve contraindications, of which the training environments have none
    representatives = [c[0] for c in classes]
    counterfactual = np.array([[_continuation_return(s, d, agent, rng) for d in representatives] for s in snapshots])
    obs = torch.as_tensor(observations)
    with torch.no_grad():
        critic = agent.q(obs)[:, representatives].double().numpy()
        probs = agent.policy(obs)
    actor = np.stack([probs[:, list(c)].sum(-1).double().numpy() for c in classes], axis=-1)
    best = counterfactual.max(axis=1)
    ordered = np.sort(counterfactual, axis=1)

    def regret(scores: np.ndarray) -> np.ndarray:
        return best - counterfactual[np.arange(len(scores)), scores.argmax(axis=1)]

    def centred(values: np.ndarray) -> np.ndarray:
        return values - values.mean(axis=1, keepdims=True)

    return {
        "states": len(snapshots),
        "critic_ranks_best_class_first": float(np.mean(regret(critic) <= 1e-9)),
        "actor_ranks_best_class_first": float(np.mean(regret(actor) <= 1e-9)),
        "critic_choice_regret": float(regret(critic).mean()),
        "actor_choice_regret": float(regret(actor).mean()),
        "mean_gap_best_to_second": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "centred_value_rmse": float(np.sqrt(np.mean((centred(critic) - centred(counterfactual)) ** 2))),
        "value_bias": float((critic - counterfactual).mean()),
    }


def _planned_failure(snapshot, first_drug: int | None = None, planner_steps: int = 200) -> float:
    """Failure from `snapshot` to the horizon, in expected-failure-rate units, under the noise-free planner.

    Optionally `first_drug` is given for the rest of the current quarter first. The
    planner plans once from the state it starts in, on a copy of the environment, so the
    snapshot is left untouched. A mixed plan is sampled patient by patient, so the planner's
    random stream restarts where it takes over: the planner's value of a state is then
    the same whether the state is a snapshot or reached through `first_drug`, which is
    what lets the losses of _gap_attribution telescope. Without noise the model replays
    the environment, but the value is still one sample of how a mixed plan is realised.
    """
    from ocmdp.baselines.oracle import FullInformationOracle
    from ocmdp.components.constraint_filter import sample_feasible

    env = copy.deepcopy(snapshot)
    toy = env.unwrapped
    rng = np.random.default_rng(0)  # only resolves contraindications, of which the training environments have none
    state = toy.state()
    info = {"quarter": state["quarter"], "resistance": state["resistance"], "action_mask": state["action_mask"]}
    obs, total, done = None, 0.0, False
    if first_drug is not None:
        quarter = info["quarter"]
        while not done and info["quarter"] == quarter:
            obs, reward, terminated, truncated, info = env.step(sample_feasible(np.eye(env.action_space.n)[first_drug], info["action_mask"], rng))
            total, done = total - reward, terminated or truncated
    rng = np.random.default_rng(0)
    planner = FullInformationOracle(env, replan_every=toy.config.horizon_quarters, steps=planner_steps, name="planned-once")
    while not done:
        obs, reward, terminated, truncated, info = env.step(planner.act(obs, info, rng))
        total, done = total - reward, terminated or truncated
    return total / (toy.config.horizon_quarters * toy.config.decisions_per_quarter)


def _gap_attribution(agent, snapshots: list, observations: np.ndarray, chosen: np.ndarray, logs: list,
                     classes: list[tuple[int, ...]], planner_steps: int = 200) -> dict:  # fmt: skip
    """Where along its own trajectory the agent loses against the noise-free planner, and why. Without noise only.

    At every quarterly decision s_t of the sampled evaluation: V*(s_t) is the planner's
    failure from s_t to the horizon, and Q*(s_t, c) the failure of class c for this quarter
    followed by the planner. The losses V*(s_t) - Q*(s_t, a_t) (sign flipped: lower failure
    is better) telescope along each episode to the agent's gap to the planner on it, which
    `closure_error` checks. Each loss splits into what one drug per quarter costs against a
    plan that may mix (V* against the best Q*) and the choice of a worse class, and the
    choice loss is attributed, in this order, to the first link that fails: the agent's own
    continuation ranking classes differently from the planner, the critics misranking
    classes against that continuation, or the class taken departing from the critics'.
    The split depends on that order. The first link also counts the difference between the
    agent's discounted soft objective and the planner's undiscounted failure, not only a
    local optimum of the agent's policy; the last includes ordinary sampling from a
    stochastic policy. All values are in expected-failure-rate units and are relative to
    the planner, which is the best schedule it finds, not a proven optimum; each planner
    value is one realisation of how a mixed plan is sampled patient by patient.
    """
    import torch

    rng = np.random.default_rng(0)
    owner = {d: c for c, members in enumerate(classes) for d in members}
    representatives = [c[0] for c in classes]
    per_episode = len(snapshots) // len(logs)
    obs = torch.as_tensor(observations)
    with torch.no_grad():
        critic_rank = agent.q(obs)[:, representatives].double().numpy().argmax(axis=1)
    totals = {"policy_class": 0.0, "continuation_ranks_differently": 0.0, "critic_misranks": 0.0, "actor_departs_from_critic": 0.0}
    closure, gaps = [], []
    for e, log in enumerate(logs):
        index = range(e * per_episode, (e + 1) * per_episode)
        planned = [_planned_failure(snapshots[i], planner_steps=planner_steps) for i in index]
        loss_sum = 0.0
        for k, i in enumerate(index):
            q_star = np.array([_planned_failure(snapshots[i], d, planner_steps) for d in representatives])
            q_own = np.array([_continuation_return(snapshots[i], d, agent, rng) for d in representatives])  # higher is better
            taken = owner[int(chosen[i])]
            totals["policy_class"] += q_star.min() - planned[k]
            choice_loss = q_star[taken] - q_star.min()
            if choice_loss > 1e-12:  # rounding in a tie between classes is not a choice
                if q_own.argmax() != q_star.argmin():
                    totals["continuation_ranks_differently"] += choice_loss
                elif critic_rank[i] != q_own.argmax():
                    totals["critic_misranks"] += choice_loss
                else:
                    totals["actor_departs_from_critic"] += choice_loss
            loss_sum += q_star[taken] - planned[k]  # Q*(s_t, a_t) - V*(s_t), which telescopes over the episode
        gap = float(np.mean(log.fail_prob[log.decision_quarter, log.actions])) - planned[0]
        gaps.append(gap)
        closure.append(abs(loss_sum - gap))
    n = len(logs)
    return {
        "gap_to_planner": float(np.mean(gaps)),
        "closure_error": float(max(closure)),
        **{f"lost_to_{name}": value / n for name, value in totals.items()},
    }


def train_and_evaluate(job: tuple) -> dict:
    """One agent trained and evaluated. The job is (variant, sigma, seed, quarters, episodes, overrides, first episode,
    instance, diagnose, checkpoint_every, condition, the launch's fingerprint, strict).

    With `strict` (the test tier) a job whose platform fingerprint differs from the
    launch's stops before training; otherwise the mismatch is recorded.
    """
    (variant, sigma, seed, total_quarters, episodes, overrides, first_episode, instance, diagnose, checkpoint_every,
     condition, launch_fingerprint, strict) = job  # fmt: skip
    import torch

    from ocmdp.agents.oc_sac import QuarterlyChoicePolicy, SACConfig, checkpoint_bytes, parameter_sha256, train

    global _jobs_in_process
    _jobs_in_process += 1
    torch.set_num_threads(JOB_THREADS)
    own = platform_fields()["platform_fingerprint"]
    if strict and own != launch_fingerprint:
        raise RuntimeError(f"this job's platform fingerprint {own} is not the launch's {launch_fingerprint}")
    env = make_env(sigma, *instance, condition=condition)
    cfg = SACConfig(**{"total_quarters": total_quarters, **overrides, **VARIANTS[variant], "drug_embedding": CONDITIONS[condition]})
    start = time.time()
    try:
        result = train(env, cfg, seed, checkpoint_every)
    except FloatingPointError as exc:
        return {"variant": variant, "sigma": sigma, "seed": seed, "host": host(_jobs_in_process), "fingerprint_matches_launch": own == launch_fingerprint,
                "diverged": str(exc), "diverged_at_decision": exc.decisions}  # fmt: skip
    agent = result.agent
    observations, snapshots, chosen = [], [], []  # every quarterly decision of the sampled evaluation, in order

    def choose(obs, rng):
        observations.append(np.array(obs, dtype=np.float32))
        snapshots.append(copy.deepcopy(env))  # before the quarter's first patient
        chosen.append(agent.choose(obs, rng))
        return chosen[-1]

    policy = QuarterlyChoicePolicy(choose, env.action_space.n, variant)
    argmax = QuarterlyChoicePolicy(lambda obs, rng: agent.choose(obs, rng, greedy=True), env.action_space.n, variant)
    logs = _rollouts(env, policy, episodes, first_episode)
    observations, chosen = np.stack(observations), np.array(chosen)
    classes = env.unwrapped.coupling.identical_row_classes()
    # with noise each counterfactual follows one noise path, which a single draw cannot separate from the critic's error
    critic = _critic_check(agent, snapshots, observations, classes) if sigma == 0 else {}
    gap = _gap_attribution(agent, snapshots, observations, chosen, logs, classes) if diagnose and sigma == 0 else {}
    return {
        "variant": variant,
        "sigma": sigma,
        "seed": seed,
        "seconds": time.time() - start,
        "host": host(_jobs_in_process),
        "fingerprint_matches_launch": own == launch_fingerprint,
        "parameter_sha256": {**result.parameter_sha256, "final": parameter_sha256(agent)},
        "learning_curve": [float(np.mean(result.episode_failure[i : i + 25])) for i in range(0, len(result.episode_failure), 25)],
        **_episode_stats(env, logs, cfg.gamma),
        "policy_entropy": _policy_entropy(np.stack([agent.probs(obs) for obs in observations]), env.unwrapped.coupling.identical_row_classes()),
        "critic": critic,
        "gap_attribution": gap,
        "alpha_path": result.alpha,  # mean temperature over each window of dual_update_every updates
        "entropy_path": result.entropy,  # mean batch entropy in the agent's own entropy space, same windows
        "xi_path": result.xi,
        "alpha_at_evaluation": float(agent.temperature()),
        "entropy_range": {"floor": agent.entropy_floor, "maximum": agent.entropy_max, "target": agent.entropy_target},
        "argmax": _episode_stats(env, _rollouts(env, argmax, episodes, first_episode), cfg.gamma),
        "checkpoints": [*result.checkpoints, (cfg.total_quarters, checkpoint_bytes(agent, cfg.total_quarters))],
    }


def write_checkpoints(out: Path, tier: str, report_name: str, run: dict) -> list[dict]:
    """Write a run's checkpoints to out/checkpoints/<tier>/<report_name>/ and return what the report records about each.

    Each path is recorded relative to `out` with forward slashes, so reports from every platform read alike.
    """
    written = []
    for decisions, blob in run.pop("checkpoints", []):
        relative = Path("checkpoints") / tier / report_name / f"{run['variant']}_sigma{run['sigma']}_seed{run['seed']}_q{decisions}.pt"
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        (out / relative).write_bytes(blob)
        written.append({"path": relative.as_posix(), "decisions": decisions, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
    return written


def committed_per_quarter(base):
    """`base`'s most probable drug at the first decision of each quarter, given to every patient in that quarter.

    The agents' policy class: one drug per quarter. The oracle may split a quarter
    between drugs, which no agent can do, so this is the oracle's behaviour as far as
    that class can copy it: a reference for the class, not an upper bound on it.
    """
    from ocmdp.baselines.heuristics import HeuristicPolicy

    class CommittedPerQuarter(HeuristicPolicy):
        name = f"{base.name}[one drug per quarter]"

        def reset(self) -> None:
            base.reset()
            self._quarter, self._drug = None, None

        def probs(self, obs: np.ndarray, info: dict) -> np.ndarray:
            if info["quarter"] != self._quarter:
                self._quarter, self._drug = info["quarter"], int(np.argmax(base.probs(obs, info)))
            return np.eye(len(info["fail_prob"]))[self._drug]

    policy = CommittedPerQuarter()
    policy.reset()
    return policy


def evaluate_baselines(job: tuple[float, int, int, tuple, float]) -> dict:
    sigma, episodes, first_episode, instance, gamma = job
    from ocmdp.baselines.heuristics import MyopicGreedy, uniform_mixing
    from ocmdp.baselines.oracle import FullInformationOracle, tuned_cycling

    env = make_env(sigma, *instance)
    policies = [
        uniform_mixing(env.action_space.n),
        MyopicGreedy(),
        tuned_cycling(env.unwrapped.config),
        committed_per_quarter(FullInformationOracle(env)),
        FullInformationOracle(env),
        FullInformationOracle(env, replan_every=1, name=REPLANNER),
    ]
    return {"sigma": sigma, "baselines": {p.name: _episode_stats(env, _rollouts(env, p, episodes, first_episode), gamma) for p in policies}}


def iqm(values) -> float:
    """Interquartile mean: the mean after dropping int(n / 4) values from each end (scipy's trim_mean at 0.25)."""
    v = np.sort(np.asarray(values, dtype=float))
    cut = int(0.25 * v.size)
    return float(v[cut : v.size - cut].mean())


def probability_of_improvement(x, y) -> float:
    """P(X < Y) over all pairs of runs, ties counting one half: how often a run of x fails less than a run of y."""
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(np.mean((x < y) + 0.5 * (x == y)))


def bootstrap_ci95(values, resamples: int = 10_000, seed: int = 0) -> list[float]:
    """Percentile 95% interval of the mean, resampling the given values (seeds) with replacement."""
    v = np.asarray(values, dtype=float)
    means = v[np.random.default_rng(seed).integers(v.size, size=(resamples, v.size))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def two_level_bootstrap_ci95(differences: np.ndarray, resamples: int = 10_000, seed: int = 0) -> list[float]:
    """Percentile 95% interval of the mean of a (seeds, episodes) array of paired differences.

    Every seed is evaluated on the same episodes, so seeds and episodes are crossed, not
    nested: each resample draws seeds and episodes independently and keeps the same
    episodes for every drawn seed, so an effect shared by an episode across seeds keeps
    its full weight. Resampling seeds alone treats the evaluation episodes as fixed. Where
    seed variance dominates, as in every logged report, a synthetic check put its coverage
    at about 0.86 to 0.88 with five seeds, so the t interval over seeds stays primary.
    """
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(d.shape[0], size=(resamples, d.shape[0]))
    episodes = rng.integers(d.shape[1], size=(resamples, d.shape[1]))
    means = d[seeds[:, :, None], episodes[:, None, :]].mean(axis=(1, 2))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(runs: list[dict], baselines: dict, sigma: float, equivalence_margin: float | None = None) -> dict:
    from ocmdp.eval.vv import aggregate

    base = {name: np.array(stats["expected_failure_rate"]) for name, stats in baselines.items()}
    uniform, oracle, replanner = base["uniform"].mean(), base["oracle"].mean(), base[REPLANNER].mean()
    out = {"baselines": {name: aggregate(rates) for name, rates in base.items()}}
    out["discounted_failure_rate"] = {name: aggregate(stats["discounted_failure_rate"]) for name, stats in baselines.items()}
    per_variant, by_seed = {}, {}
    for variant in VARIANTS:
        done = [r for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" not in r]
        per_seed = {r["seed"]: np.mean(r["expected_failure_rate"]) for r in done}
        per_variant[variant] = per_seed
        by_seed[variant] = {r["seed"]: r["expected_failure_rate"] for r in done}
        rates = np.array(list(per_seed.values()))
        diverged = sum(1 for r in runs if r["variant"] == variant and r["sigma"] == sigma and "diverged" in r)
        replanner_share = (uniform - rates) / (uniform - replanner)
        out[variant] = {
            "expected_failure_rate": aggregate(rates),
            "iqm_expected_failure_rate": iqm(rates),
            "argmax_expected_failure_rate": aggregate([np.mean(r["argmax"]["expected_failure_rate"]) for r in done]),
            # the planner bounds the agents only without noise; with noise it is the noise-free planner and the replanner is the reference
            ("share_of_oracle_gain" if sigma == 0 else "share_of_noise_free_planner_gain"): aggregate((uniform - rates) / (uniform - oracle)),
            "share_of_replanner_gain": aggregate(replanner_share),
            "runs_learned": int(np.sum(replanner_share > 0.5)),  # more than half the replanner's gain over uniform; a diverged run has not learned
            "runs": len(rates) + diverged,  # diverged runs included
            "diverged": diverged,
            "H_action": float(np.mean([r["H_action"] for r in done])),
            "H_pushforward": float(np.mean([r["H_pushforward"] for r in done])),
            "policy_entropy": {k: float(np.mean([r["policy_entropy"][k] for r in done])) for k in ("action", "within_class")},
            "critic": {k: float(np.mean([r["critic"][k] for r in done])) for k in done[0]["critic"]} if done else {},
            "gap_attribution": {k: float(np.mean([r["gap_attribution"][k] for r in done])) for k in done[0]["gap_attribution"]} if done else {},
            "alpha_last_training_window": aggregate([r["alpha_path"][-1] for r in done if r["alpha_path"]]) if any(r["alpha_path"] for r in done) else None,
            "alpha_at_evaluation": aggregate([r["alpha_at_evaluation"] for r in done]) if done else None,
            "minus_baseline_paired_by_episode": {
                name: aggregate([np.mean(np.asarray(r["expected_failure_rate"]) - base[name]) for r in done]) for name in base
            },
        }
        out["discounted_failure_rate"][variant] = aggregate([np.mean(r["discounted_failure_rate"]) for r in done])
    shared = sorted(set(per_variant["oc-sac"]) & set(per_variant["sac"]))
    diffs = [per_variant["oc-sac"][s] - per_variant["sac"][s] for s in shared]
    out["oc-sac minus sac (paired by seed)"] = aggregate(diffs)
    out["paired comparison"] = {
        "seeds": shared,
        "probability_oc_sac_fails_less": probability_of_improvement([per_variant["oc-sac"][s] for s in shared], [per_variant["sac"][s] for s in shared]),
        "seeds_where_oc_sac_fails_less": int(np.sum(np.asarray(diffs) < 0)),
        "bootstrap_ci95_of_mean_difference": bootstrap_ci95(diffs),
        "two_level_bootstrap_ci95_of_mean_difference": two_level_bootstrap_ci95(
            [np.asarray(by_seed["oc-sac"][s]) - np.asarray(by_seed["sac"][s]) for s in shared]
        ),
        "iqm_of_difference": iqm(diffs),
    }
    if equivalence_margin is not None and len(diffs) > 1:
        from stats import equivalent

        out["paired comparison"]["equivalence"] = equivalent(diffs, equivalence_margin)
    return out


def _parse_overrides(parser: argparse.ArgumentParser, pairs: list[str], quarters: int) -> dict:
    """--set KEY=VALUE pairs as SACConfig overrides, checked at the run's own length (--quarters, which --set may not change)."""
    from ocmdp.agents.oc_sac import SACConfig

    defaults, overrides = asdict(SACConfig()), {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key == "total_quarters":
            parser.error("--set total_quarters is not accepted: the run length is --quarters")
        if key in ("entropy", "drug_embedding") or type(defaults.get(key)) not in (int, float, str) or not value:
            parser.error(f"--set takes KEY=VALUE for a numeric SACConfig field or temperature (entropy is set per variant and "
                         f"drug_embedding by --condition), got {pair!r}")  # fmt: skip
        try:
            overrides[key] = type(defaults[key])(value)
        except ValueError:
            parser.error(f"--set {key} needs a {type(defaults[key]).__name__}, got {value!r}")
    try:
        SACConfig(**{"total_quarters": quarters, **overrides})
    except ValueError as exc:
        parser.error(f"--set: {exc}")
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=TIERS, required=True, help="seed tier: fixes the allowed training seeds and the evaluation episodes")
    parser.add_argument("--prereg", type=Path, help="preregistration this run belongs to; required for --tier test")
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True, help="what the agents are given: c-given (C in their inputs) or no-c")
    parser.add_argument("--platform", help="label for the account or machine the jobs run on, not a personal name; required for val and test")
    parser.add_argument("--seeds", type=int, nargs="+", help=f"training seeds inside the tier (default: its first {DEFAULT_SEEDS})")
    parser.add_argument("--seeds-file", type=Path, default=ROOT / "seeds.yaml", help="tier definitions")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--class-sizes", type=int, nargs="+", default=list(PAIRED_DRUGS), metavar="N", help="drugs per mechanism class")
    parser.add_argument("--delay", type=int, help="quarters before prescribing becomes selection pressure (ToyConfig's default when omitted)")
    parser.add_argument("--lam", type=float, help="occupancy memory per quarter (ToyConfig's default when omitted)")
    parser.add_argument("--quarters", type=int, default=12_000, help="agent decisions per training run")
    parser.add_argument("--episodes", type=int, default=10, help="held-out evaluation episodes per policy")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="local processes (--backend local)")
    parser.add_argument("--backend", choices=("local", "modal"), default="local", help="where the jobs run; see experiments/modal_backend.py")
    parser.add_argument("--equivalence-margin", type=float, help="margin of a two one-sided test on the paired difference (none by default)")
    parser.add_argument("--diagnose-gap", action="store_true", help="attribute each agent's gap to the planner (no noise only; about 3 plans per decision)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="SACConfig overrides, applied to both variants")
    parser.add_argument("--checkpoint-every", type=int, metavar="Q", help="also save each agent every Q training decisions (the final agent is always saved)")
    parser.add_argument("--out", type=Path, default=Path("runs/train"))
    args = parser.parse_args()
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        tier = load_tiers(args.seeds_file)[args.tier]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    check_seed_file(parser, args.tier, tier, args.seeds_file)
    if CANARY_SEED not in load_tiers(ROOT / "seeds.yaml")["dev"]:
        parser.error(f"the arithmetic canary's seed {CANARY_SEED} must lie in the repository's dev tier")
    given_seeds = args.seeds
    if args.seeds is None:
        args.seeds = list(tier[:DEFAULT_SEEDS])
    outside = sorted(set(args.seeds) - set(tier))
    if outside or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"--seeds must be distinct and inside the {args.tier} tier [{tier.start}, {tier.stop - 1}]; rejected: {outside or 'duplicates'}")
    if not 1 <= args.episodes <= len(tier):
        parser.error(f"--episodes must lie in [1, {len(tier)}]: evaluation episodes are drawn from the {args.tier} tier")
    if len(args.class_sizes) < 2 or min(args.class_sizes) < 1:
        parser.error("--class-sizes needs at least two classes of at least one drug each")
    if args.equivalence_margin is not None and not args.equivalence_margin > 0:
        parser.error("--equivalence-margin must be positive")
    if (args.delay is not None and args.delay < 0) or (args.lam is not None and not 0.0 <= args.lam < 1.0):
        parser.error("--delay must be non-negative and --lam must lie in [0, 1)")
    if args.checkpoint_every is not None and (args.tier != "val" or not args.quarters / 4 <= args.checkpoint_every < args.quarters):
        parser.error("--checkpoint-every is for checkpoint choice on val only, in [--quarters / 4, --quarters): at most three per agent")
    overrides = _parse_overrides(parser, args.set, args.quarters)
    git = git_state()  # read at launch: the tree may change while the run is in progress
    prereg_slug = check_test_launch(parser, args.prereg, git, args.out, args.platform, fingerprinted=True) if args.tier == "test" else None
    if args.tier == "test":
        check_test_out(parser, args.out)
        check_test_seeds(parser, args.prereg, given_seeds, args.out)
    if args.tier != "dev" and not args.platform:
        parser.error(f"--tier {args.tier} needs --platform naming the account or machine the jobs run on")
    if args.backend == "modal":
        import modal_backend

        (fields,) = modal_backend.run_jobs([("train_agent", "platform_probe", ())])  # one container stands for the platform
    else:
        fields = platform_fields()
    if args.tier == "test":
        check_test_fingerprint(parser, args.prereg, fields)
    if args.tier == "val" and git["dirty"] is not False:
        print("warning: a val-tier launch from uncommitted code; only code_diff_sha256 identifies what ran", file=sys.stderr)
    launch_id = f"train_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    path = args.out / f"{launch_id}.json"
    launch_args = launch_arguments(args)
    prereg_sha256 = prereg_digest(args.prereg)
    record_launch(args.out, launch_id, launch_args, git, args.backend, prereg_slug, prereg_sha256, fields)

    from ocmdp.agents.oc_sac import SACConfig
    from ocmdp.eval.seeds import eval_seed

    config = SACConfig(**{"total_quarters": args.quarters, **overrides})
    sizes = tuple(args.class_sizes)
    instance = (sizes, args.delay, args.lam)
    jobs = training_jobs(args, overrides, tier.start, instance, fields["platform_fingerprint"])
    baseline_jobs = [(s, args.episodes, tier.start, instance, config.gamma) for s in args.sigmas]
    if args.backend == "modal":
        backend = modal_backend.description()
        calls = [("train_agent", "evaluate_baselines", job) for job in baseline_jobs]
        calls += [("train_agent", "train_and_evaluate_pair", job) for job in jobs]
        results = modal_backend.run_jobs(calls)
        baseline_results, runs = results[: len(baseline_jobs)], [run for pair in results[len(baseline_jobs) :] for run in pair]
    else:
        from importlib.metadata import version

        backend = {"name": "local", "platform": platform.platform(), "workers": args.workers,
                   "packages": {name: version(name) for name in ("numpy", "torch", "gymnasium")}}  # fmt: skip
        with ProcessPoolExecutor(args.workers) as pool:
            baseline_futures = [pool.submit(evaluate_baselines, job) for job in baseline_jobs]
            runs = [run for pair in pool.map(train_and_evaluate_pair, jobs) for run in pair]
            baseline_results = [f.result() for f in baseline_futures]
    baselines = {result["sigma"]: result["baselines"] for result in baseline_results}
    for run in runs:
        run["checkpoints"] = write_checkpoints(args.out, args.tier, path.stem, run)

    toy = make_env(0.0, *instance).unwrapped
    report = {
        "report_schema": REPORT_SCHEMA,
        "launch_id": launch_id,
        "platform": args.platform,
        "platform_fingerprint": fields["platform_fingerprint"],
        "condition": args.condition,
        "prereg_sha256": prereg_sha256,
        "args": launch_args,
        "git": git,
        "backend": backend,
        "evaluation_episode_seeds": [eval_seed(tier.start + e) for e in range(args.episodes)],
        "instance": {k: v for k, v in asdict(toy.config).items() if k != "resistance_noise"},
        "class_structure": {"class_sizes": list(sizes), "imbalance_kl_nats": imbalance_kl(sizes)},
        "sac_config": {v: asdict(SACConfig(**{"total_quarters": args.quarters, **overrides, **VARIANTS[v], "drug_embedding": CONDITIONS[args.condition]}))
                       for v in VARIANTS},  # fmt: skip
        "pairing_defects": pairing_defects(runs),
        "runs": runs,
        "baselines": {str(s): baselines[s] for s in args.sigmas},  # per-episode values, so every contrast can be recomputed
    }
    report["summary"] = {}
    for s in args.sigmas:
        try:
            report["summary"][str(s)] = summarise(runs, baselines[s], s, args.equivalence_margin)
        except ValueError as exc:  # e.g. every run of a variant diverged: the report is still written, with its runs
            report["summary"][str(s)] = {"not_summarised": str(exc)}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    unsummarised = [s for s in args.sigmas if "not_summarised" in report["summary"][str(s)]]
    for sigma in args.sigmas:
        summary = report["summary"][str(sigma)]
        if sigma in unsummarised:
            print(f"\nResistance noise {sigma}: not summarised ({summary['not_summarised']})")
            continue
        print(f"\nClass sizes {list(sizes)}, condition {args.condition}, resistance noise {sigma}: expected failure rate on {args.episodes} {args.tier}-tier episodes")
        for name, agg in summary["baselines"].items():
            print(f"  {name:30} {agg['mean']:.4f} ± {agg['ci95']:.4f}")
        for variant in VARIANTS:
            row = summary[variant]
            print(
                f"  {variant:30} {row['expected_failure_rate']['mean']:.4f} ± {row['expected_failure_rate']['ci95'] or 0:.4f}"
                f"   argmax {row['argmax_expected_failure_rate']['mean']:.4f}"
                f"   replanner gain captured {row['share_of_replanner_gain']['mean']:.0%}"
                f"   runs learned {row['runs_learned']}/{row['runs']}   diverged {row['diverged']}"
                f"   H_action {row['H_action']:.3f}  H_pushforward {row['H_pushforward']:.3f}"
                f"   policy entropy inside classes {row['policy_entropy']['within_class']:.3f}"
            )
        diff, paired = summary["oc-sac minus sac (paired by seed)"], summary["paired comparison"]
        boot = paired["bootstrap_ci95_of_mean_difference"]
        print(
            f"  oc-sac - sac (paired by seed): {diff['mean']:+.4f} ± {diff['ci95'] or 0:.4f}   bootstrap [{boot[0]:+.4f}, {boot[1]:+.4f}]"
            f"   IQM {paired['iqm_of_difference']:+.4f}   P(oc-sac fails less) {paired['probability_oc_sac_fails_less']:.2f}"
            f"   oc-sac lower on {paired['seeds_where_oc_sac_fails_less']}/{len(paired['seeds'])} seeds"
        )
        discounted = summary["discounted_failure_rate"]
        print(f"  failure rate discounted at {config.gamma} per quarter: " + "   ".join(f"{name} {agg['mean']:.4f}" for name, agg in discounted.items()))
        for variant in VARIANTS:
            critic, row = summary[variant]["critic"], summary[variant]
            alpha = row["alpha_at_evaluation"]["mean"] if row["alpha_at_evaluation"] else float("nan")
            versus = row["minus_baseline_paired_by_episode"][REPLANNER]
            if row["gap_attribution"]:
                g = row["gap_attribution"]
                print(
                    f"  {variant} gap to the planner {g['gap_to_planner']:+.4f} (closure error {g['closure_error']:.1e}): policy class"
                    f" {g['lost_to_policy_class']:+.4f}, own continuation ranks differently {g['lost_to_continuation_ranks_differently']:+.4f},"
                    f" critic misranks {g['lost_to_critic_misranks']:+.4f}, actor departs {g['lost_to_actor_departs_from_critic']:+.4f}"
                )
            if not critic:
                continue
            print(
                f"  {variant}: critic ranks the best class first in {critic['critic_ranks_best_class_first']:.0%} of states,"
                f" actor in {critic['actor_ranks_best_class_first']:.0%}; regret critic {critic['critic_choice_regret']:.3f},"
                f" actor {critic['actor_choice_regret']:.3f} (mean gap {critic['mean_gap_best_to_second']:.3f}); alpha at evaluation {alpha:.2e};"
                f" minus {REPLANNER} {versus['mean']:+.4f} ± {versus['ci95'] or 0:.4f}"
            )
    print(f"\nReport: {path}")
    if report["pairing_defects"]:
        print("pairing defects, so these pairs may not share arithmetic:\n  " + "\n  ".join(report["pairing_defects"]), file=sys.stderr)
        return 1
    return 1 if unsummarised else 0


if __name__ == "__main__":
    sys.exit(main())
