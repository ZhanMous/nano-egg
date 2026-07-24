"""Frozen-backbone mechanism search for Evo-TRQR.

This is the discovery-stage runner.  It evolves only the 16-dimensional
controller and a small discrete mechanism genome.  Backbone weights are never
perturbed or updated here; EGGROLL weight adaptation is a later experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.getcwd())
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from evo_trqr import (
    CONTINUOUS_DIM,
    CandidateMetrics,
    DiscreteGenome,
    apply_evo_trqr,
    ask_antithetic,
    better_candidate,
    candidate_order_key,
    initial_continuous_genome,
    mutate_discrete,
    tell_antithetic,
)
from experiments.spikformer_es_smoke import (
    Args,
    ParamSpec,
    cross_entropy,
    dequant,
    init_params,
    layer_norm,
    load_cifar10_arrays,
    make_specs,
    mlp_forward,
    preset_config,
    sps_forward,
    ssa_forward,
    synthetic_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=[
            "tiny",
            "smoke",
            "spikformer_2_128",
            "spikformer_4_256",
            "spikformer_dvs_2_256",
        ],
    )
    parser.add_argument("--data-source", default="synthetic", choices=["synthetic", "cifar10", "event_npz"])
    parser.add_argument(
        "--event-cache",
        default="",
        help="NPZ with train_frames/train_labels/eval_frames/eval_labels; frames are [B,T,H,W,C]",
    )
    parser.add_argument("--backbone-state", default="", help="Named nano-egg checkpoint NPZ")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--cifar-train-samples", type=int, default=512)
    parser.add_argument("--cifar-eval-samples", type=int, default=128)
    parser.add_argument("--cifar-cache-dir", default="cached_files/cifar10_jax")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--energy-budget", type=float, default=0.9)
    parser.add_argument("--budget-ceiling", type=float, default=0.8)
    parser.add_argument("--mask-seeds", default="0,1")
    parser.add_argument(
        "--progress-grid",
        default="1.0",
        help="Comma-separated schedule positions. Use 1.0 for frozen final-policy discovery.",
    )
    parser.add_argument("--structure-interval", type=int, default=2)
    parser.add_argument("--structure-proposals", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/evo_trqr_frozen_search")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lif-mode", default="hard_spike", choices=["hard_spike", "soft_spike", "leaky_clip", "leaky_tanh"])
    parser.add_argument("--soft-spike-width", type=float, default=0.25)
    parser.add_argument("--continuous-clip", type=float, default=1.0)
    return parser.parse_args()


def parse_csv_numbers(raw: str, cast) -> list:
    values = [cast(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def load_named_checkpoint(path: str, specs: Sequence[ParamSpec]) -> dict[str, jax.Array]:
    with np.load(path, allow_pickle=False) as checkpoint:
        if "param_names" not in checkpoint:
            raise ValueError(f"{path} is not a named nano-egg checkpoint")
        names = [str(value) for value in checkpoint["param_names"].tolist()]
        encoded = {name: np.asarray(checkpoint[f"p{index:05d}"]) for index, name in enumerate(names)}
    params: dict[str, jax.Array] = {}
    for spec in specs:
        if spec.name not in encoded:
            raise ValueError(f"checkpoint is missing {spec.name}")
        value = encoded[spec.name]
        if value.shape != spec.shape:
            raise ValueError(f"checkpoint shape mismatch for {spec.name}: {value.shape} != {spec.shape}")
        params[spec.name] = jnp.asarray(value)
    return params


def take_fixed_batch(
    frames: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[jax.Array, jax.Array]:
    if frames.shape[0] != labels.shape[0]:
        raise ValueError("frame and label counts differ")
    if batch_size > frames.shape[0]:
        raise ValueError(f"batch_size={batch_size} exceeds split size={frames.shape[0]}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(frames.shape[0], size=batch_size, replace=False)
    return jnp.asarray(frames[indices], dtype=jnp.float32), jnp.asarray(labels[indices], dtype=jnp.int32)


def load_batches(cli: argparse.Namespace, cfg, model_args: Args) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    if cli.data_source == "synthetic":
        return (
            synthetic_batch(cfg, cli.seed, 0, cli.batch_size),
            synthetic_batch(cfg, cli.seed, 1, cli.eval_batch_size),
        )
    if cli.data_source == "cifar10":
        arrays = load_cifar10_arrays(model_args, cfg)
        return (
            take_fixed_batch(
                arrays["train_images"],
                arrays["train_labels"],
                batch_size=cli.batch_size,
                seed=cli.seed + 101,
            ),
            take_fixed_batch(
                arrays["eval_images"],
                arrays["eval_labels"],
                batch_size=cli.eval_batch_size,
                seed=cli.seed + 202,
            ),
        )
    if not cli.event_cache:
        raise ValueError("--event-cache is required for data-source=event_npz")
    with np.load(cli.event_cache, allow_pickle=False) as data:
        required = ("train_frames", "train_labels", "eval_frames", "eval_labels")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"event cache is missing keys: {missing}")
        arrays = {key: np.asarray(data[key]) for key in required}
    for split in ("train", "eval"):
        frames = arrays[f"{split}_frames"]
        if frames.ndim != 5:
            raise ValueError(f"{split}_frames must be [B,T,H,W,C], got {frames.shape}")
        expected = (cfg.time_steps, cfg.image_size, cfg.image_size, cfg.in_channels)
        if tuple(frames.shape[1:]) != expected:
            raise ValueError(f"{split}_frames shape tail {frames.shape[1:]} != expected {expected}")
    return (
        take_fixed_batch(
            arrays["train_frames"],
            arrays["train_labels"],
            batch_size=cli.batch_size,
            seed=cli.seed + 101,
        ),
        take_fixed_batch(
            arrays["eval_frames"],
            arrays["eval_labels"],
            batch_size=cli.eval_batch_size,
            seed=cli.seed + 202,
        ),
    )


def clean_forward(params, spec_by_name, images, cfg, model_args: Args) -> tuple[jax.Array, float]:
    x = sps_forward(params, spec_by_name, images, cfg, model_args, 0, None, 1)
    activity = 0.0
    for block_idx in range(cfg.depth):
        x = x + ssa_forward(params, spec_by_name, x, cfg, model_args, 0, None, 1, block_idx)
        x = x + mlp_forward(params, spec_by_name, x, cfg, model_args, 0, None, 1, block_idx)
        activity += float(jnp.sum(jnp.abs(x)))
    pooled = x.mean(axis=(0, 2))
    pooled = layer_norm(
        pooled,
        dequant(params["head/norm/gamma"]),
        dequant(params["head/norm/beta"]),
    )
    if model_args.head_feature_centering == "batch" and pooled.shape[0] > 1:
        pooled = pooled - jnp.mean(pooled, axis=0, keepdims=True)
    logits = pooled @ dequant(params["head/w"]) + dequant(params["head/b"])
    return logits, activity


def controlled_forward(
    params,
    spec_by_name,
    images,
    cfg,
    model_args: Args,
    continuous: np.ndarray,
    discrete: DiscreteGenome,
    *,
    progress: float,
    mask_seed: int,
    master_strength: float,
    budget_ceiling: float,
) -> tuple[jax.Array, float, list[dict[str, float]]]:
    x = sps_forward(params, spec_by_name, images, cfg, model_args, 0, None, 1)
    activity = 0.0
    diagnostics: list[dict[str, float]] = []
    base_key = jax.random.PRNGKey(mask_seed)
    for block_idx in range(cfg.depth):
        x = x + ssa_forward(params, spec_by_name, x, cfg, model_args, 0, None, 1, block_idx)
        x = x + mlp_forward(params, spec_by_name, x, cfg, model_args, 0, None, 1, block_idx)
        pre_controller_activity = float(jnp.sum(jnp.abs(x)))
        if block_idx < 4:
            x, diag = apply_evo_trqr(
                x,
                continuous,
                discrete,
                layer_index=block_idx,
                progress=progress,
                random_key=jax.random.fold_in(base_key, block_idx),
                master_strength=master_strength,
                budget_ceiling=budget_ceiling,
            )
            diagnostics.append(
                {
                    "layer": block_idx,
                    "local_mi": float(diag.local_mi_mean),
                    "global_mi": float(diag.global_mi_mean),
                    "mask_probability": float(diag.mask_probability_mean),
                    "mask_fraction": float(diag.mask_fraction),
                    "removed_activity_fraction": float(diag.removed_activity_fraction),
                }
            )
            activity += pre_controller_activity * (1.0 - float(diag.removed_activity_fraction))
        else:
            activity += pre_controller_activity
    pooled = x.mean(axis=(0, 2))
    pooled = layer_norm(
        pooled,
        dequant(params["head/norm/gamma"]),
        dequant(params["head/norm/beta"]),
    )
    if model_args.head_feature_centering == "batch" and pooled.shape[0] > 1:
        pooled = pooled - jnp.mean(pooled, axis=0, keepdims=True)
    logits = pooled @ dequant(params["head/w"]) + dequant(params["head/b"])
    return logits, activity, diagnostics


def metrics_from_logits(logits: jax.Array, labels: jax.Array, energy_ratio: float) -> CandidateMetrics:
    loss = float(cross_entropy(logits, labels))
    accuracy = float(jnp.mean(jnp.argmax(logits, axis=-1) == labels))
    return CandidateMetrics(accuracy=accuracy, loss=loss, energy_ratio=float(energy_ratio))


def average_metrics(metrics: Sequence[CandidateMetrics]) -> CandidateMetrics:
    return CandidateMetrics(
        accuracy=float(np.mean([item.accuracy for item in metrics])),
        loss=float(np.mean([item.loss for item in metrics])),
        energy_ratio=float(np.mean([item.energy_ratio for item in metrics])),
    )


def evaluate_candidate(
    params,
    spec_by_name,
    batch,
    cfg,
    model_args,
    continuous,
    discrete,
    *,
    progress_grid: Sequence[float],
    mask_seeds: Sequence[int],
    baseline_activity: float,
    budget_ceiling: float,
) -> tuple[CandidateMetrics, list[dict[str, float]]]:
    images, labels = batch
    samples: list[CandidateMetrics] = []
    latest_diagnostics: list[dict[str, float]] = []
    for progress in progress_grid:
        for mask_seed in mask_seeds:
            logits, activity, latest_diagnostics = controlled_forward(
                params,
                spec_by_name,
                images,
                cfg,
                model_args,
                continuous,
                discrete,
                progress=progress,
                mask_seed=mask_seed,
                master_strength=1.0,
                budget_ceiling=budget_ceiling,
            )
            samples.append(metrics_from_logits(logits, labels, activity / max(baseline_activity, 1e-8)))
    return average_metrics(samples), latest_diagnostics


def save_best(
    output_dir: Path,
    continuous: np.ndarray,
    discrete: DiscreteGenome,
    search_metrics: CandidateMetrics,
    eval_metrics: CandidateMetrics,
) -> None:
    payload = {
        "continuous": np.asarray(continuous).tolist(),
        "discrete": asdict(discrete),
        "search_metrics": asdict(search_metrics),
        "eval_metrics": asdict(eval_metrics),
    }
    (output_dir / "best_genome.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    cli = parse_args()
    if cli.generations <= 0 or cli.pairs <= 0:
        raise ValueError("--generations and --pairs must be positive")
    if not 0.0 < cli.energy_budget <= 1.0:
        raise ValueError("--energy-budget must be in (0, 1]")
    if cli.data_source == "event_npz" and cli.preset != "spikformer_dvs_2_256":
        raise ValueError("event_npz currently requires --preset spikformer_dvs_2_256")
    if cli.data_source == "event_npz" and not cli.backbone_state:
        raise ValueError("event_npz discovery requires a trained --backbone-state")

    mask_seeds = parse_csv_numbers(cli.mask_seeds, int)
    progress_grid = parse_csv_numbers(cli.progress_grid, float)
    if any(not 0.0 <= value <= 1.0 for value in progress_grid):
        raise ValueError("--progress-grid values must be in [0, 1]")

    cfg = preset_config(cli.preset)
    specs = make_specs(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
    params = load_named_checkpoint(cli.backbone_state, specs) if cli.backbone_state else init_params(specs, cli.seed)
    model_args = Args(
        preset=cli.preset,
        data_source="cifar10" if cli.data_source == "cifar10" else "synthetic",
        seed=cli.seed,
        batch_size=max(cli.batch_size, cli.eval_batch_size),
        cifar_cache_dir=cli.cifar_cache_dir,
        cifar_train_samples=cli.cifar_train_samples,
        cifar_eval_samples=cli.cifar_eval_samples,
        lif_mode=cli.lif_mode,
        soft_spike_width=cli.soft_spike_width,
        continuous_clip=cli.continuous_clip,
        head_feature_centering="batch",
        use_jit=False,
    )
    search_batch, eval_batch = load_batches(cli, cfg, model_args)
    clean_search_logits, search_baseline_activity = clean_forward(
        params, spec_by_name, search_batch[0], cfg, model_args
    )
    clean_eval_logits, eval_baseline_activity = clean_forward(
        params, spec_by_name, eval_batch[0], cfg, model_args
    )
    clean_search = metrics_from_logits(clean_search_logits, search_batch[1], 1.0)
    clean_eval = metrics_from_logits(clean_eval_logits, eval_batch[1], 1.0)

    center = initial_continuous_genome()
    discrete = DiscreteGenome()
    identity_logits, _, _ = controlled_forward(
        params,
        spec_by_name,
        search_batch[0],
        cfg,
        model_args,
        center,
        discrete,
        progress=1.0,
        mask_seed=mask_seeds[0],
        master_strength=0.0,
        budget_ceiling=cli.budget_ceiling,
    )
    identity_max_abs_error = float(jnp.max(jnp.abs(identity_logits - clean_search_logits)))
    if identity_max_abs_error != 0.0:
        raise RuntimeError(f"identity gate is not exact: max_abs_error={identity_max_abs_error}")

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    best_path = output_dir / "best_genome.json"
    config_path = output_dir / "run_config.json"
    if not cli.overwrite and (metrics_path.exists() or best_path.exists() or config_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains search artifacts; choose a new --output-dir "
            "or pass --overwrite"
        )
    config = {
        "args": vars(cli),
        "model_config": asdict(cfg),
        "continuous_dim": CONTINUOUS_DIM,
        "clean_search": asdict(clean_search),
        "clean_eval": asdict(clean_eval),
        "identity_max_abs_error": identity_max_abs_error,
        "energy_metric": "retained_pre_recalibration_stage_activity_proxy",
        "warning": "activity proxy is not a hardware latency or energy measurement",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if metrics_path.exists():
        metrics_path.unlink()

    rng = np.random.default_rng(cli.seed)
    active_dimensions = np.ones((CONTINUOUS_DIM,), dtype=np.float32)
    for layer_idx in range(cfg.depth, 4):
        active_dimensions[layer_idx] = 0.0
        active_dimensions[4 + layer_idx] = 0.0
    if progress_grid == [1.0]:
        active_dimensions[14:16] = 0.0

    center_metrics, center_diag = evaluate_candidate(
        params,
        spec_by_name,
        search_batch,
        cfg,
        model_args,
        center,
        discrete,
        progress_grid=progress_grid,
        mask_seeds=mask_seeds,
        baseline_activity=search_baseline_activity,
        budget_ceiling=cli.budget_ceiling,
    )
    best_center = center.copy()
    best_discrete = discrete
    best_search = center_metrics
    best_eval, _ = evaluate_candidate(
        params,
        spec_by_name,
        eval_batch,
        cfg,
        model_args,
        best_center,
        best_discrete,
        progress_grid=[1.0],
        mask_seeds=mask_seeds,
        baseline_activity=eval_baseline_activity,
        budget_ceiling=cli.budget_ceiling,
    )
    save_best(output_dir, best_center, best_discrete, best_search, best_eval)

    print(
        f"neutrality_max_abs_error={identity_max_abs_error:.1f} "
        f"clean_search_acc={clean_search.accuracy:.4f} clean_eval_acc={clean_eval.accuracy:.4f}"
    )
    print(
        f"initial acc={center_metrics.accuracy:.4f} loss={center_metrics.loss:.6f} "
        f"energy_ratio={center_metrics.energy_ratio:.4f} discrete={asdict(discrete)}"
    )

    for generation in range(cli.generations):
        start = time.time()
        plus, minus, noise = ask_antithetic(
            center,
            pairs=cli.pairs,
            sigma=cli.sigma,
            rng=rng,
            active_dimensions=active_dimensions,
        )
        plus_metrics: list[CandidateMetrics] = []
        minus_metrics: list[CandidateMetrics] = []
        for pair_idx in range(cli.pairs):
            plus_result, _ = evaluate_candidate(
                params,
                spec_by_name,
                search_batch,
                cfg,
                model_args,
                plus[pair_idx],
                discrete,
                progress_grid=progress_grid,
                mask_seeds=mask_seeds,
                baseline_activity=search_baseline_activity,
                budget_ceiling=cli.budget_ceiling,
            )
            minus_result, _ = evaluate_candidate(
                params,
                spec_by_name,
                search_batch,
                cfg,
                model_args,
                minus[pair_idx],
                discrete,
                progress_grid=progress_grid,
                mask_seeds=mask_seeds,
                baseline_activity=search_baseline_activity,
                budget_ceiling=cli.budget_ceiling,
            )
            plus_metrics.append(plus_result)
            minus_metrics.append(minus_result)

        center = tell_antithetic(
            center,
            noise,
            plus_metrics,
            minus_metrics,
            energy_budget=cli.energy_budget,
            sigma=cli.sigma,
            learning_rate=cli.learning_rate,
        )
        center_metrics, center_diag = evaluate_candidate(
            params,
            spec_by_name,
            search_batch,
            cfg,
            model_args,
            center,
            discrete,
            progress_grid=progress_grid,
            mask_seeds=mask_seeds,
            baseline_activity=search_baseline_activity,
            budget_ceiling=cli.budget_ceiling,
        )

        structure_accepted = False
        if cli.structure_interval > 0 and (generation + 1) % cli.structure_interval == 0:
            for _ in range(cli.structure_proposals):
                proposal = mutate_discrete(discrete, rng)
                proposal_metrics, proposal_diag = evaluate_candidate(
                    params,
                    spec_by_name,
                    search_batch,
                    cfg,
                    model_args,
                    center,
                    proposal,
                    progress_grid=progress_grid,
                    mask_seeds=mask_seeds,
                    baseline_activity=search_baseline_activity,
                    budget_ceiling=cli.budget_ceiling,
                )
                if better_candidate(proposal_metrics, center_metrics, cli.energy_budget):
                    discrete = proposal
                    center_metrics = proposal_metrics
                    center_diag = proposal_diag
                    structure_accepted = True

        if better_candidate(center_metrics, best_search, cli.energy_budget):
            best_center = center.copy()
            best_discrete = discrete
            best_search = center_metrics
            best_eval, _ = evaluate_candidate(
                params,
                spec_by_name,
                eval_batch,
                cfg,
                model_args,
                best_center,
                best_discrete,
                progress_grid=[1.0],
                mask_seeds=mask_seeds,
                baseline_activity=eval_baseline_activity,
                budget_ceiling=cli.budget_ceiling,
            )
            save_best(output_dir, best_center, best_discrete, best_search, best_eval)

        row = {
            "generation": generation,
            "center": center.tolist(),
            "discrete": asdict(discrete),
            "center_metrics": asdict(center_metrics),
            "best_search_metrics": asdict(best_search),
            "best_eval_metrics": asdict(best_eval),
            "feasible": center_metrics.energy_ratio <= cli.energy_budget,
            "order_key": candidate_order_key(center_metrics, cli.energy_budget),
            "structure_accepted": structure_accepted,
            "diagnostics": center_diag,
            "seconds": time.time() - start,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"generation={generation} acc={center_metrics.accuracy:.4f} "
            f"loss={center_metrics.loss:.6f} energy_ratio={center_metrics.energy_ratio:.4f} "
            f"feasible={row['feasible']} structure_accepted={structure_accepted} "
            f"seconds={row['seconds']:.2f}"
        )

    print(
        f"best search_acc={best_search.accuracy:.4f} search_loss={best_search.loss:.6f} "
        f"search_energy={best_search.energy_ratio:.4f} eval_acc={best_eval.accuracy:.4f} "
        f"eval_energy={best_eval.energy_ratio:.4f}"
    )
    print(f"best_genome={output_dir / 'best_genome.json'}")


if __name__ == "__main__":
    main()
