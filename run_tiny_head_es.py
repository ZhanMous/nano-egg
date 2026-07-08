"""Head-only ES gate for a calibrated Spikformer preset.

This freezes the random trunk, precomputes eval features once, and updates
only the quantized linear classification head. It is intended as a fast check
that the feature scale and ES direction are usable before returning to full
model ES.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"

import jax
import jax.numpy as jnp
import numpy as np

from experiments.spikformer_es_smoke import (
    Args,
    DTYPE,
    PARAM_MAX,
    PARAM_MIN,
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
    parser.add_argument("--preset", default="tiny", choices=["tiny", "smoke", "spikformer_2_128", "spikformer_4_256"])
    parser.add_argument("--rule", choices=["rank_full", "sign_full", "sign_top1"], default="rank_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def ce_vector(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsum
    return -log_probs[np.arange(labels.shape[0]), labels]


def ce_population(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsum
    picked = np.take_along_axis(log_probs, labels[None, :, None], axis=2).squeeze(-1)
    return -picked.mean(axis=1)


def centered_rank_pair_weights(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    losses = np.concatenate([plus, minus])
    fitness = -losses
    order = np.argsort(fitness)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(fitness), dtype=np.float32)
    centered = ranks / max(len(fitness) - 1, 1) - 0.5
    centered = centered / (centered.std() + 1e-8)
    return centered[: plus.shape[0]] - centered[plus.shape[0] :]


def main() -> None:
    cli = parse_args()
    if cli.population_size % 2 != 0:
        raise ValueError("--population-size must be even")

    cfg = preset_config(cli.preset)
    specs = make_specs(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
    head_w_spec = spec_by_name["head/w"]
    head_b_spec = spec_by_name["head/b"]
    params0 = init_params(specs, cli.seed)

    args = Args(
        preset=cli.preset,
        data_source="cifar10",
        seed=cli.seed,
        epochs=cli.epochs,
        population_size=cli.population_size,
        batch_size=cli.batch_size,
        sigma=cli.sigma,
        noise_rank=1,
        update_fraction=(0.01 if cli.rule == "sign_top1" else 1.0),
        param_step=1,
        cifar_train_samples=50000,
        cifar_eval_samples=10000,
        cifar_cache_dir="cached_files/cifar10_jax",
    )
    data = load_cifar10_arrays(args, cfg)
    arrays = {
        "train_images": jnp.asarray(data["train_images"]),
        "train_labels": jnp.asarray(data["train_labels"].astype(int)),
    }
    eval_labels = np.asarray(data["eval_labels"].astype(int))

    output_dir = Path(cli.output_dir or f"runs/{cli.preset}_head_es_{cli.rule}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def features(images: jax.Array) -> np.ndarray:
        x = sps_forward(params0, spec_by_name, images, cfg, args, 0, None, 1)
        for block_idx in range(cfg.depth):
            x = x + ssa_forward(params0, spec_by_name, x, cfg, args, 0, None, 1, block_idx)
            x = x + mlp_forward(params0, spec_by_name, x, cfg, args, 0, None, 1, block_idx)
        pooled = x.mean(axis=(0, 2))
        pooled = layer_norm(pooled, dequant(params0["head/norm/gamma"]), dequant(params0["head/norm/beta"]))
        if pooled.shape[0] > 1:
            pooled = pooled - jnp.mean(pooled, axis=0, keepdims=True)
        return np.asarray(pooled, dtype=np.float32)

    def eval_head(weight_q: np.ndarray, bias_q: np.ndarray, feats: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        logits = feats @ (weight_q.astype(np.float32) / 16.0) + (bias_q.astype(np.float32) / 16.0)
        losses = ce_vector(logits, labels)
        pred = logits.argmax(axis=-1)
        return float(losses.mean()), float((pred == labels).mean() * 100.0)

    def noise_chunk(epoch: int, start: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        idx = jnp.arange(start, start + size, dtype=jnp.int32)
        keys_w = jax.vmap(lambda i: noise_key(cli.seed, epoch, i, head_w_spec))(idx)
        keys_b = jax.vmap(lambda i: noise_key(cli.seed, epoch, i, head_b_spec))(idx)
        noise_w = np.asarray(jax.vmap(lambda key: matrix_noise(head_w_spec, key, 1))(keys_w), dtype=np.float32)
        noise_b = np.asarray(jax.vmap(lambda key: matrix_noise(head_b_spec, key, 1))(keys_b), dtype=np.float32)
        return noise_w, noise_b

    def population_losses(feats: np.ndarray, labels: np.ndarray, weight_q: np.ndarray, bias_q: np.ndarray, epoch: int):
        n_pairs = cli.population_size // 2
        weight = weight_q.astype(np.float32) / 16.0
        bias = bias_q.astype(np.float32) / 16.0
        base_logits = feats @ weight + bias
        plus = np.empty(n_pairs, dtype=np.float32)
        minus = np.empty(n_pairs, dtype=np.float32)
        for start in range(0, n_pairs, cli.chunk):
            size = min(cli.chunk, n_pairs - start)
            noise_w, noise_b = noise_chunk(epoch, start, size)
            noise_logits = np.einsum("bd,pdc->pbc", feats, noise_w) + noise_b[:, None, :]
            plus[start : start + size] = ce_population(base_logits[None, :, :] + cli.sigma * noise_logits, labels)
            minus[start : start + size] = ce_population(base_logits[None, :, :] - cli.sigma * noise_logits, labels)
        return plus, minus

    def z_from_pair_weights(pair_weights: np.ndarray, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        z_w = np.zeros(head_w_spec.shape, dtype=np.float32)
        z_b = np.zeros(head_b_spec.shape, dtype=np.float32)
        for start in range(0, pair_weights.shape[0], cli.chunk):
            size = min(cli.chunk, pair_weights.shape[0] - start)
            noise_w, noise_b = noise_chunk(epoch, start, size)
            weights = pair_weights[start : start + size].astype(np.float32)
            z_w += (weights[:, None, None] * noise_w).sum(axis=0)
            z_b += (weights[:, None] * noise_b).sum(axis=0)
        return z_w, z_b

    def apply_update(weight_q: np.ndarray, bias_q: np.ndarray, z_w: np.ndarray, z_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fraction = 0.01 if cli.rule == "sign_top1" else 1.0
        next_w = weight_q.astype(np.int16).copy()
        next_b = bias_q.astype(np.int16).copy()
        for arr, z in ((next_w, z_w), (next_b, z_b)):
            abs_z = np.abs(z)
            if fraction < 1.0:
                threshold = np.percentile(abs_z.reshape(-1), 100.0 * (1.0 - fraction))
                mask = abs_z >= threshold
            else:
                mask = abs_z > 0
            arr[:] = np.clip(arr + np.where(mask, np.sign(z).astype(np.int16), 0), PARAM_MIN, PARAM_MAX)
        return next_w.astype(DTYPE), next_b.astype(DTYPE)

    print(
        f"preset={cli.preset} rule={cli.rule} population={cli.population_size} "
        f"batch={cli.batch_size} sigma={cli.sigma}"
    )
    print("Precomputing eval features...", flush=True)
    eval_features = np.concatenate(
        [features(jnp.asarray(data["eval_images"][start : start + 256])) for start in range(0, len(data["eval_images"]), 256)],
        axis=0,
    )
    print(f"eval_features={eval_features.shape}", flush=True)

    weight_q = np.asarray(params0["head/w"])
    bias_q = np.asarray(params0["head/b"])
    results = []
    test_loss, test_acc = eval_head(weight_q, bias_q, eval_features, eval_labels)
    results.append({"epoch": -1, "test_loss": test_loss, "test_acc": test_acc})
    best_acc = test_acc
    np.savez_compressed(output_dir / "best_head_state.npz", head_w=weight_q, head_b=bias_q)
    print(f"Epoch -1: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")

    for epoch in range(cli.epochs):
        start_time = time.time()
        train_images, train_labels = train_batch(args, cfg, arrays, epoch)
        train_features = features(train_images)
        train_labels_np = np.asarray(train_labels)
        train_loss_before, train_acc_before = eval_head(weight_q, bias_q, train_features, train_labels_np)
        plus, minus = population_losses(train_features, train_labels_np, weight_q, bias_q, epoch)
        if cli.rule == "rank_full":
            pair_weights = centered_rank_pair_weights(plus, minus)
        else:
            pair_weights = np.sign(minus - plus)
        z_w, z_b = z_from_pair_weights(pair_weights, epoch)
        weight_q, bias_q = apply_update(weight_q, bias_q, z_w, z_b)
        train_loss_after, train_acc_after = eval_head(weight_q, bias_q, train_features, train_labels_np)
        test_loss, test_acc = eval_head(weight_q, bias_q, eval_features, eval_labels)
        seconds = time.time() - start_time
        row = {
            "epoch": epoch,
            "train_loss_before": train_loss_before,
            "train_acc_before": train_acc_before,
            "train_loss_after": train_loss_after,
            "train_acc_after": train_acc_after,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": seconds,
        }
        results.append(row)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(output_dir / "head_state.npz", head_w=weight_q, head_b=bias_q)
        if test_acc >= best_acc:
            best_acc = test_acc
            np.savez_compressed(output_dir / "best_head_state.npz", head_w=weight_q, head_b=bias_q)
        print(
            f"Epoch {epoch:>2}: train {train_loss_before:.4f}->{train_loss_after:.4f} "
            f"acc {train_acc_before:.2f}->{train_acc_after:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% time={seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
