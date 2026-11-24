"""Predicted failure-rate gap between class entropy and drug entropy, from the mean-field model alone.

    python experiments/predict_entropy_gap.py --class-sizes 3 1 --alphas 0 0.001 0.01 0.1
    python experiments/predict_entropy_gap.py --from-report runs/train/train_<stamp>.json

With exactly interchangeable drugs and networks that tie them, a policy is uniform within
each class, so its entropy over drugs is H(q) + sum_j q_j log n_j, with q the class
masses. Drug-entropy SAC therefore maximises the class-entropy objective plus
alpha * sum_j q_j log n_j: a bonus for mass on larger classes, zero in difference when
the classes are equal. For each temperature this solves both problems on the
mean-field model, as open-loop schedules of class masses from the nominal start,

    maximise  sum_q gamma**q * (alpha * H(q_q) - reward_scale * failure_q)      class entropy (OC-SAC)
    maximise  the same + gamma**q * alpha * sum_j q_qj log n_j                   drug entropy (SAC)

and reports each optimum's undiscounted failure rate and their difference. With
--from-report the temperatures are each agent's temperature at evaluation (the median
over evaluation states for reports from the earlier state-dependent temperature),
averaged over seeds, so the prediction is made at the temperatures the agents actually
reached. Approximations, each a way the prediction can miss: open loop from the
nominal start where the agents act on the jittered state; a mixed schedule where the
agents give one drug per quarter; one temperature where the agents' alpha depends on the
state. Writes a JSON report to runs/predict/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train_agent import ENV_KWARGS, PAIRED_DRUGS, cross_resistance, git_state

MODES = ("class entropy", "drug entropy")


def soft_optimal_schedule(model, class_sizes, alpha: float, mode: str, gamma: float, reward_scale: float,
                          steps: int, n_random: int, seed: int) -> dict:  # fmt: skip
    """Best open-loop schedule of class masses for one entropy mode and temperature, by Adam from several starts."""
    from ocmdp.baselines.oracle import best_rotation

    sizes = torch.tensor(class_sizes, dtype=torch.float64)
    n_quarters, n_classes = model.config.horizon_quarters, len(class_sizes)
    owner = torch.repeat_interleave(torch.arange(n_classes), sizes.long())  # class of each drug
    discount = gamma ** torch.arange(n_quarters, dtype=torch.float64)

    def schedule(masses: torch.Tensor) -> torch.Tensor:
        return masses[..., owner] / sizes[owner]  # uniform within each class

    def objective(masses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        failure = model.simulate(schedule(masses))[0]
        entropy = -(masses * masses.clamp_min(1e-300).log()).sum(-1)
        if mode == "drug entropy":
            entropy = entropy + (masses * sizes.log()).sum(-1)
        return ((alpha * entropy - reward_scale * failure) * discount).sum(-1), failure

    classes = [tuple(int(d) for d in torch.nonzero(owner == c).flatten()) for c in range(n_classes)]
    rotation, _, _ = best_rotation(model, model.initial_state(), classes, n_quarters)
    rotation_masses = torch.stack([torch.from_numpy(rotation[:, list(c)].sum(-1)) for c in classes], -1)
    generator = torch.Generator().manual_seed(seed)
    starts = torch.cat([
        torch.zeros(1, n_quarters, n_classes, dtype=torch.float64),
        rotation_masses.clamp_min(1e-6).log()[None],
        torch.randn(n_random, n_quarters, n_classes, generator=generator, dtype=torch.float64),
    ])  # fmt: skip
    logits = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    best = (-float("inf"), None)
    for step in range(steps + 1):
        value, _ = objective(torch.softmax(logits, -1))
        i = int(torch.nan_to_num(value.detach(), nan=-float("inf")).argmax())
        if value[i].item() > best[0]:
            best = (value[i].item(), logits.detach()[i].clone())
        if step < steps:
            opt.zero_grad()
            (-value.sum()).backward()
            opt.step()
    masses = torch.softmax(best[1], -1)
    with torch.no_grad():
        _, failure = objective(masses)
    class_entropy = -(masses * masses.clamp_min(1e-300).log()).sum(-1)
    return {
        "soft_return": best[0],
        "expected_failure_rate": float(failure.mean()),
        "mean_class_entropy": float(class_entropy.mean()),
        "mean_mass_by_class": masses.mean(0).tolist(),
    }


def predict(class_sizes: tuple[int, ...], alphas: dict[str, float], gamma: float, reward_scale: float,
            steps: int, n_random: int, seed: int) -> dict:  # fmt: skip
    from ocmdp.envs.toy import ToyConfig
    from ocmdp.theory.mean_field import MeanFieldModel

    model = MeanFieldModel(ToyConfig(cross_resistance=cross_resistance(class_sizes), **ENV_KWARGS))
    out = {mode: soft_optimal_schedule(model, class_sizes, alphas[mode], mode, gamma, reward_scale, steps, n_random, seed) for mode in MODES}
    out["alphas"] = dict(alphas)
    out["class entropy minus drug entropy"] = out["class entropy"]["expected_failure_rate"] - out["drug entropy"]["expected_failure_rate"]
    return out


def temperatures_from_report(path: Path) -> tuple[tuple[int, ...], dict[str, float], float, float]:
    """Class sizes, each variant's evaluation temperature averaged over seeds, gamma and reward scale.

    Reports from agents with a state-dependent temperature store its quantiles over the
    evaluation states, and the median is used; later reports store the one scalar.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    variant_mode = {"oc-sac": "class entropy", "sac": "drug entropy"}
    alphas = {}
    for variant, mode in variant_mode.items():
        at_evaluation = [r["alpha_at_evaluation"] for r in report["runs"] if r["variant"] == variant and "alpha_at_evaluation" in r]
        medians = [a["q50"] if isinstance(a, dict) else a for a in at_evaluation]
        if not medians:
            raise ValueError(f"{path} logs no evaluation temperatures for {variant}")
        alphas[mode] = float(np.mean(medians))
    config = report["sac_config"]["oc-sac"]
    return tuple(report["class_structure"]["class_sizes"]), alphas, config["gamma"], config["reward_scale"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-sizes", type=int, nargs="+", default=list(PAIRED_DRUGS), metavar="N")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1], help="temperatures, applied to both modes")
    parser.add_argument("--from-report", type=Path, help="a runs/train report: predict at the temperatures its agents reached")
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=400, help="Adam steps per schedule")
    parser.add_argument("--random-starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/predict"))
    args = parser.parse_args()
    if len(args.class_sizes) < 2 or min(args.class_sizes) < 1:
        parser.error("--class-sizes needs at least two classes of at least one drug each")
    if min(args.alphas) < 0 or not 0 < args.gamma <= 1 or args.steps < 1 or args.random_starts < 0:
        parser.error("need alphas >= 0, gamma in (0, 1], steps >= 1 and random starts >= 0")
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out {args.out} exists and is not a directory")
    torch.set_num_threads(1)
    git = git_state()

    if args.from_report:
        try:
            sizes, alphas, gamma, reward_scale = temperatures_from_report(args.from_report)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        predictions = [predict(sizes, alphas, gamma, reward_scale, args.steps, args.random_starts, args.seed)]
    else:
        sizes, gamma, reward_scale = tuple(args.class_sizes), args.gamma, args.reward_scale
        predictions = [predict(sizes, {m: a for m in MODES}, gamma, reward_scale, args.steps, args.random_starts, args.seed) for a in args.alphas]

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git": git,
        "class_sizes": list(sizes),
        "gamma": gamma,
        "reward_scale": reward_scale,
        "predictions": predictions,
    }
    path = args.out / f"predict_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Class sizes {list(sizes)}: predicted expected failure rate of each soft-optimal schedule")
    for p in predictions:
        a = p["alphas"]
        print(
            f"  alpha class {a['class entropy']:.2e} / drug {a['drug entropy']:.2e}:"
            f"  class entropy {p['class entropy']['expected_failure_rate']:.4f}  drug entropy {p['drug entropy']['expected_failure_rate']:.4f}"
            f"  difference {p['class entropy minus drug entropy']:+.5f}"
        )
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
