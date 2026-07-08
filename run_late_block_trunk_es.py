"""Cached-prefix ES for training the last Spikformer block.

This runner is deliberately narrower than full-model ES. It freezes SPS and
early blocks, caches the token tensor before a target block, then applies
EGGROLL-style ES only to the original quantized parameters of that late block.
The readout adapter/head can be initialized from a feature-adapter checkpoint
and is kept fixed, so any improvement must come from trunk representation
changes in the target block.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax
import jax.numpy as jnp
import numpy as np

from experiments.spikformer_es_smoke import (
    Args,
    DTYPE,
    PARAM_MAX,
    PARAM_MIN,
    PARAM_SCALE,
    ParamSpec,
    dequant,
    init_params,
    layer_norm,
    load_cifar10_arrays,
    make_specs,
    matrix_noise,
    mlp_forward,
    noise_key,
    preset_config,
    sps_forward,
    ssa_forward,
    train_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="spikformer_4_256", choices=["tiny", "smoke", "spikformer_2_128", "spikformer_4_256"])
    parser.add_argument("--target-block", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--noise-rank", type=int, default=1)
    parser.add_argument("--update-fraction", type=float, default=0.001)
    parser.add_argument("--param-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lif-mode", choices=["hard_spike", "soft_spike", "leaky_clip", "leaky_tanh"], default="soft_spike")
    parser.add_argument("--soft-spike-width", type=float, default=0.25)
    parser.add_argument("--continuous-clip", type=float, default=1.0)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--init-block-state", default="")
    parser.add_argument("--adapter-scale", type=float, default=0.25)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--cifar-train-samples", type=int, default=50000)
    parser.add_argument("--cifar-eval-samples", type=int, default=10000)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--cifar-cache-dir", default="cached_files/cifar10_jax")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def ce_vector(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsum
    return -log_probs[np.arange(labels.shape[0]), labels]


def centered_rank_pair_weights(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    losses = np.concatenate([plus, minus])
    fitness = -losses
    order = np.argsort(fitness)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(fitness), dtype=np.float32)
    centered = ranks / max(len(fitness) - 1, 1) - 0.5
    centered = centered / (centered.std() + 1e-8)
    return centered[: plus.shape[0]] - centered[plus.shape[0] :]


def load_adapter_state(path: str, dim: int) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    expected = {
        "down_w": (dim, loaded["down_w"].shape[1]),
        "down_b": (loaded["down_w"].shape[1],),
        "up_w": (loaded["down_w"].shape[1], dim),
        "up_b": (dim,),
        "head_w": (dim, 10),
        "head_b": (10,),
    }
    missing = [name for name in expected if name not in loaded]
    if missing:
        raise ValueError(f"adapter state {path} is missing keys: {missing}")
    state = {name: loaded[name].astype(np.int8) for name in expected}
    for name, shape in expected.items():
        if state[name].shape != shape:
            raise ValueError(f"adapter state shape mismatch for {name}: {state[name].shape} != {shape}")
    return state


def encoded_param_key(index: int) -> str:
    return f"p{index:05d}"


def save_block_state(params: dict[str, jax.Array], specs: list[ParamSpec], path: Path) -> None:
    arrays = {encoded_param_key(i): np.asarray(params[spec.name]) for i, spec in enumerate(specs)}
    arrays["param_names"] = np.asarray([spec.name for spec in specs])
    arrays["param_shapes"] = np.asarray([json.dumps(spec.shape) for spec in specs])
    np.savez_compressed(path, **arrays)


def load_block_state(params: dict[str, jax.Array], specs: list[ParamSpec], path: Path) -> dict[str, jax.Array]:
    loaded = np.load(path)
    names = [str(x) for x in loaded["param_names"]]
    if names != [spec.name for spec in specs]:
        raise ValueError(f"block state parameter names do not match {path}")
    updated = dict(params)
    for i, spec in enumerate(specs):
        value = loaded[encoded_param_key(i)].astype(np.int8)
        if value.shape != spec.shape:
            raise ValueError(f"block state shape mismatch for {spec.name}: {value.shape} != {spec.shape}")
        updated[spec.name] = jnp.asarray(value)
    return updated


def main() -> None:
    cli = parse_args()
    if cli.population_size % 2 != 0:
        raise ValueError("--population-size must be even")
    if cli.population_size // 2 % cli.chunk != 0:
        raise ValueError("--population-size / 2 must be divisible by --chunk")
    if not 0 < cli.update_fraction <= 1:
        raise ValueError("--update-fraction must be in (0, 1]")

    cfg = preset_config(cli.preset)
    if not 0 <= cli.target_block < cfg.depth:
        raise ValueError(f"--target-block must be in [0, {cfg.depth})")

    specs = make_specs(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
    train_specs = [
        spec for spec in specs
        if spec.name.startswith(f"blocks/{cli.target_block}/")
        and not spec.name.endswith("/norm2/gamma")
        and not spec.name.endswith("/norm2/beta")
    ]
    params = init_params(specs, cli.seed)
    adapter_state_np = load_adapter_state(cli.adapter_state, cfg.dim)
    adapter_state = {name: jnp.asarray(value) for name, value in adapter_state_np.items()}

    args = Args(
        preset=cli.preset,
        data_source="cifar10",
        seed=cli.seed,
        epochs=cli.epochs,
        population_size=cli.population_size,
        batch_size=cli.batch_size,
        sigma=cli.sigma,
        noise_rank=cli.noise_rank,
        update_fraction=cli.update_fraction,
        param_step=cli.param_step,
        cifar_train_samples=cli.cifar_train_samples,
        cifar_eval_samples=cli.cifar_eval_samples,
        cifar_cache_dir=cli.cifar_cache_dir,
        lif_mode=cli.lif_mode,
        soft_spike_width=cli.soft_spike_width,
        continuous_clip=cli.continuous_clip,
        update_scope=f"block{cli.target_block}",
    )
    data = load_cifar10_arrays(args, cfg)
    arrays = {
        "train_images": jnp.asarray(data["train_images"]),
        "train_labels": jnp.asarray(data["train_labels"].astype(int)),
    }
    eval_images = np.asarray(data["eval_images"])
    eval_labels = np.asarray(data["eval_labels"].astype(int))
    if cli.eval_limit > 0:
        eval_images = eval_images[:cli.eval_limit]
        eval_labels = eval_labels[:cli.eval_limit]

    output_dir = Path(cli.output_dir or f"runs/{cli.preset}_block{cli.target_block}_trunk_es")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    state_path = output_dir / "block_state.npz"
    best_path = output_dir / "best_block_state.npz"

    if state_path.exists() and results_path.exists():
        params = load_block_state(params, train_specs, state_path)
        results = json.loads(results_path.read_text())
        start_epoch = int(results[-1]["epoch"]) + 1 if results and results[-1]["epoch"] >= 0 else 0
        best_acc = max(float(row["test_acc"]) for row in results)
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)
    else:
        if cli.init_block_state:
            params = load_block_state(params, train_specs, Path(cli.init_block_state))
        results = []
        start_epoch = 0
        best_acc = -1.0

    @jax.jit
    def prefix_features(images):
        x = sps_forward(params, spec_by_name, images, cfg, args, 0, None, 1)
        for block_idx in range(cli.target_block):
            x = x + ssa_forward(params, spec_by_name, x, cfg, args, 0, None, 1, block_idx)
            x = x + mlp_forward(params, spec_by_name, x, cfg, args, 0, None, 1, block_idx)
        return x

    def adapter_logits(feats):
        down_w = adapter_state["down_w"].astype(jnp.float32) / PARAM_SCALE
        down_b = adapter_state["down_b"].astype(jnp.float32) / PARAM_SCALE
        up_w = adapter_state["up_w"].astype(jnp.float32) / PARAM_SCALE
        up_b = adapter_state["up_b"].astype(jnp.float32) / PARAM_SCALE
        head_w = adapter_state["head_w"].astype(jnp.float32) / PARAM_SCALE
        head_b = adapter_state["head_b"].astype(jnp.float32) / PARAM_SCALE
        hidden = jnp.maximum(feats @ down_w + down_b, 0.0)
        adapted = feats + cli.adapter_scale * (hidden @ up_w + up_b)
        return adapted @ head_w + head_b

    def logits_from_prefix_core(params_in, prefix, epoch, pair_idx, sign):
        x = prefix
        x = x + ssa_forward(params_in, spec_by_name, x, cfg, args, epoch, pair_idx, sign, cli.target_block)
        x = x + mlp_forward(params_in, spec_by_name, x, cfg, args, epoch, pair_idx, sign, cli.target_block)
        pooled = x.mean(axis=(0, 2))
        pooled = layer_norm(pooled, dequant(params_in["head/norm/gamma"]), dequant(params_in["head/norm/beta"]))
        if pooled.shape[0] > 1:
            pooled = pooled - jnp.mean(pooled, axis=0, keepdims=True)
        return adapter_logits(pooled)

    @jax.jit
    def clean_logits_from_prefix(params_in, prefix):
        return logits_from_prefix_core(params_in, prefix, jnp.int32(0), None, 1)

    @jax.jit
    def loss_chunk(params_in, prefix, labels, epoch, start):
        idx = start + jnp.arange(cli.chunk, dtype=jnp.int32)

        def one_loss(pair_idx, sign):
            logits = logits_from_prefix_core(params_in, prefix, epoch, pair_idx, sign)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            picked = jnp.take_along_axis(log_probs, labels[:, None], axis=1).squeeze(-1)
            return -jnp.mean(picked)

        plus = jax.vmap(lambda pair_idx: one_loss(pair_idx, 1))(idx)
        minus = jax.vmap(lambda pair_idx: one_loss(pair_idx, -1))(idx)
        return plus, minus

    update_fns = {}
    for spec in train_specs:
        def z_step(weights, epoch, start, _spec=spec):
            idx = start + jnp.arange(cli.chunk, dtype=jnp.int32)
            keys = jax.vmap(lambda i: noise_key(cli.seed, epoch, i, _spec))(idx)
            noises = jax.vmap(lambda key: matrix_noise(_spec, key, cli.noise_rank))(keys)
            return (weights.reshape((-1,) + (1,) * len(_spec.shape)) * noises).sum(axis=0)
        update_fns[spec.name] = jax.jit(z_step)

    def eval_params(params_in) -> tuple[float, float]:
        logits_all = []
        for start in range(0, len(eval_images), cli.eval_batch_size):
            images = jnp.asarray(eval_images[start : start + cli.eval_batch_size])
            prefix = prefix_features(images)
            logits = clean_logits_from_prefix(params_in, prefix)
            logits_all.append(np.asarray(logits))
        logits_np = np.concatenate(logits_all, axis=0)
        losses = ce_vector(logits_np, eval_labels)
        return float(losses.mean()), float((logits_np.argmax(axis=-1) == eval_labels).mean() * 100.0)

    def eval_prefix(params_in, prefix, labels_np: np.ndarray) -> tuple[float, float]:
        logits_np = np.asarray(clean_logits_from_prefix(params_in, prefix))
        losses = ce_vector(logits_np, labels_np)
        return float(losses.mean()), float((logits_np.argmax(axis=-1) == labels_np).mean() * 100.0)

    summary = {
        "target_block": cli.target_block,
        "train_params": int(sum(np.prod(spec.shape) for spec in train_specs)),
        "train_arrays": len(train_specs),
        "adapter_state": cli.adapter_state,
    }
    print(
        f"preset={cli.preset} block={cli.target_block} train_params={summary['train_params']:,} "
        f"arrays={summary['train_arrays']} population={cli.population_size} batch={cli.batch_size} "
        f"chunk={cli.chunk} sigma={cli.sigma} update_fraction={cli.update_fraction} lif_mode={cli.lif_mode}",
        flush=True,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps({"args": vars(cli), "summary": summary}, indent=2) + "\n"
    )

    if cli.eval_only:
        test_loss, test_acc = eval_params(params)
        row = {"epoch": -1, "phase": "eval_only", "test_loss": test_loss, "test_acc": test_acc}
        results_path.write_text(json.dumps([row], indent=2) + "\n")
        print(f"Eval-only: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%", flush=True)
        return

    if not results:
        test_loss, test_acc = eval_params(params)
        initial = {"epoch": -1, "test_loss": test_loss, "test_acc": test_acc}
        results.append(initial)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        save_block_state(params, train_specs, state_path)
        save_block_state(params, train_specs, best_path)
        best_acc = test_acc
        print(f"Epoch -1: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%", flush=True)

    n_pairs = cli.population_size // 2
    for epoch in range(start_epoch, cli.epochs):
        start_time = time.time()
        train_images, train_labels = train_batch(args, cfg, arrays, epoch)
        train_labels_np = np.asarray(train_labels).astype(np.int32)
        prefix = prefix_features(train_images)
        train_loss_before, train_acc_before = eval_prefix(params, prefix, train_labels_np)

        plus = np.empty(n_pairs, dtype=np.float32)
        minus = np.empty(n_pairs, dtype=np.float32)
        for start in range(0, n_pairs, cli.chunk):
            p, m = loss_chunk(params, prefix, train_labels, jnp.int32(epoch), jnp.int32(start))
            plus[start : start + cli.chunk] = np.asarray(p)
            minus[start : start + cli.chunk] = np.asarray(m)

        pair_weights = centered_rank_pair_weights(plus, minus).astype(np.float32)
        changed = {}
        for spec in train_specs:
            z_total = np.zeros(spec.shape, dtype=np.float32)
            for start in range(0, n_pairs, cli.chunk):
                z = update_fns[spec.name](
                    jnp.asarray(pair_weights[start : start + cli.chunk]),
                    jnp.int32(epoch),
                    jnp.int32(start),
                )
                z_total += np.asarray(z)
            abs_z = np.abs(z_total)
            if cli.update_fraction < 1.0:
                threshold = np.percentile(abs_z.reshape(-1), 100.0 * (1.0 - cli.update_fraction))
                mask = abs_z >= threshold
            else:
                mask = abs_z > 0
            old = np.asarray(params[spec.name])
            updated = np.clip(
                old.astype(np.int16) + np.where(mask, np.sign(z_total).astype(np.int16) * cli.param_step, 0),
                PARAM_MIN,
                PARAM_MAX,
            ).astype(np.int8)
            changed[spec.name] = float(np.mean(updated != old))
            params[spec.name] = jnp.asarray(updated)

        train_loss_after, train_acc_after = eval_prefix(params, prefix, train_labels_np)
        test_loss, test_acc = eval_params(params)
        seconds = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss_before": train_loss_before,
            "train_acc_before": train_acc_before,
            "train_loss_after": train_loss_after,
            "train_acc_after": train_acc_after,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "plus_loss_mean": float(plus.mean()),
            "minus_loss_mean": float(minus.mean()),
            "pair_weight_std": float(pair_weights.std()),
            "seconds": seconds,
            "changed_mean": float(np.mean(list(changed.values()))),
            "changed": changed,
        }
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        save_block_state(params, train_specs, state_path)
        if test_acc >= best_acc:
            best_acc = test_acc
            save_block_state(params, train_specs, best_path)
        print(
            f"Epoch {epoch:>2}: train {train_loss_before:.4f}->{train_loss_after:.4f} "
            f"acc {train_acc_before:.2f}->{train_acc_after:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
            f"changed={row['changed_mean']:.6f} time={seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
