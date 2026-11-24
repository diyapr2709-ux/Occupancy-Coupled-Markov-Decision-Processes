"""Run the jobs of an experiment script on Modal instead of in local processes.

    python experiments/train_agent.py --tier dev --backend modal
    python experiments/sweep_phase_diagram.py --tier dev --backend modal

A job is a call `module.function(job)` into one of SCRIPTS, and each runs in its own
one-CPU container. The image pins numpy, torch (CPU build) and gymnasium to the versions
installed locally, and the ocmdp package and the scripts are added from the working
tree when the app starts, so a remote run executes the code the report's git state
describes. Containers are x86-64 Linux: floating-point results need not match a local
arm64 run bit for bit, which is why each report records its backend. Every job of one
report runs on the same backend.
"""

import importlib
import platform
from importlib.metadata import version
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SCRIPTS = ("train_agent", "sweep_phase_diagram")
PYTHON = ".".join(platform.python_version_tuple()[:2])
PINNED = {name: version(name) for name in ("numpy", "torch", "gymnasium")}
JOB_TIMEOUT = 4 * 3600  # seconds; a 12 000-quarter training run with its evaluation has taken up to about 15 minutes

# Containers land on different x86 CPUs, and torch and MKL pick SIMD kernels per CPU, which changes
# rounding and, in a chaotic training run, the outcome. Both are pinned to their portable paths, but
# that does not make containers reproduce each other: numpy's and OpenBLAS's dispatch are not pinned,
# and two identical pinned launches trained identically on only 7 of 10 jobs. Seeds are draws.
ARITHMETIC = {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE"}

image = (
    modal.Image.debian_slim(python_version=PYTHON)
    .uv_pip_install(f"numpy=={PINNED['numpy']}", f"gymnasium=={PINNED['gymnasium']}")
    .uv_pip_install(f"torch=={PINNED['torch']}", index_url="https://download.pytorch.org/whl/cpu")
    .env(ARITHMETIC)
    .add_local_python_source("ocmdp")
)
for script in SCRIPTS:
    image = image.add_local_file(HERE / f"{script}.py", f"/root/{script}.py")
app = modal.App("ocmdp-experiments", image=image)


@app.function(cpu=1.0, timeout=JOB_TIMEOUT)
def run_job(module: str, function: str, job: tuple):
    return getattr(importlib.import_module(module), function)(job)


def run_jobs(calls: list[tuple[str, str, tuple]]) -> list:
    """(module, function, job) triples -> their results, in the same order."""
    unknown = {module for module, _, _ in calls} - set(SCRIPTS)
    if unknown:
        raise ValueError(f"jobs must call into {SCRIPTS}, got {sorted(unknown)}")
    with modal.enable_output(), app.run():
        return list(run_job.starmap(calls))


def description() -> dict:
    return {"name": "modal", "python": PYTHON, "packages": PINNED, "cpu_per_job": 1.0, "environment": ARITHMETIC}
