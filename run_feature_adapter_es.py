"""Feature-adapter ES on top of a frozen calibrated Spikformer trunk.

This is the next diagnostic after head-only ES. It keeps the expensive SNN
trunk frozen and precomputes pooled trunk features, but trains a residual
bottleneck adapter plus the linear head using centered-rank ES. The goal is to
test whether ES can improve parameters beyond the classifier head without the
full-model forward cost.
"""
import argparse
import json
import math
import os
import sys
import time
import zlib
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--adapter-scale", type=float, default=0.25)
    parser.add_argument("--update-fraction", type=float, default=1.0)
    parser.add_argument("--param-step", type=int, default=1)
    parser.add_argument("--init-head-state", default="")
    parser.add_argument("--init-adapter-state", default="")
    parser.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lif-mode", choices=["hard_spike", "soft_spike", "leaky_clip", "leaky_tanh"], default="hard_spike")
    parser.add_argument("--soft-spike-width", type=float, default=0.25)
    parser.add_argument("--continuous-clip", type=float, default=1.0)
    parser.add_argument("--plastic-head-lr", type=float, default=0.0)
    parser.add_argument("--plastic-support-fraction", type=float, default=0.5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def name_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def ce_vector(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsum
    return -log_probs[np.arange(labels.shape[0]), labels]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def centered_rank_pair_weights(plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    losses = np.concatenate([plus, minus])
    fitness = -losses
    order = np.argsort(fitness)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(fitness), dtype=np.float32)
    centered = ranks / max(len(fitness) - 1, 1) - 0.5
    centered = centered / (centered.std() + 1e-8)
    return centered[: plus.shape[0]] - centered[plus.shape[0] :]


def adapter_specs(dim: int, bottleneck: int) -> dict[str, ParamSpec]:
    return {
        "down_w": ParamSpec("adapter/down/w", "linear", (dim, bottleneck), name_seed("adapter/down/w")),
        "down_b": ParamSpec("adapter/down/b", "vector", (bottleneck,), name_seed("adapter/down/b")),
        "up_w": ParamSpec("adapter/up/w", "linear", (bottleneck, dim), name_seed("adapter/up/w")),
        "up_b": ParamSpec("adapter/up/b", "vector", (dim,), name_seed("adapter/up/b")),
        "head_w": ParamSpec("head/w", "linear", (dim, 10), name_seed("head/w")),
        "head_b": ParamSpec("head/b", "vector", (10,), name_seed("head/b")),
    }


def init_adapter_state(dim: int, bottleneck: int, seed: int, head_w: np.ndarray, head_b: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 991)
    # Nonzero down projection with zero up projection gives an identity adapter
    # at initialization while leaving useful random hidden features for up_w.
    down_w = rng.choice(np.array([-1, 1], dtype=np.int8), size=(dim, bottleneck))
    return {
        "down_w": down_w.astype(np.int8),
        "down_b": np.zeros((bottleneck,), dtype=np.int8),
        "up_w": np.zeros((bottleneck, dim), dtype=np.int8),
        "up_b": np.zeros((dim,), dtype=np.int8),
        "head_w": head_w.astype(np.int8),
        "head_b": head_b.astype(np.int8),
    }


def main() -> None:
    cli = parse_args()
    if cli.population_size % 2 != 0:
        raise ValueError("--population-size must be even")
    if cli.population_size // 2 % cli.chunk != 0:
        raise ValueError("--population-size / 2 must be divisible by --chunk")
    if not 0 < cli.update_fraction <= 1:
        raise ValueError("--update-fraction must be in (0, 1]")
    if not 0.0 <= cli.plastic_head_lr:
        raise ValueError("--plastic-head-lr must be non-negative")
    if not 0.0 < cli.plastic_support_fraction < 1.0:
        raise ValueError("--plastic-support-fraction must be in (0, 1)")
    support_size = int(cli.batch_size * cli.plastic_support_fraction)
    if cli.plastic_head_lr > 0.0 and not 0 < support_size < cli.batch_size:
        raise ValueError("--plastic-head-lr requires at least one support and one query sample")

    cfg = preset_config(cli.preset)
    specs = make_specs(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
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
        update_fraction=cli.update_fraction,
        param_step=cli.param_step,
        cifar_train_samples=50000,
        cifar_eval_samples=10000,
        cifar_cache_dir="cached_files/cifar10_jax",
        lif_mode=cli.lif_mode,
        soft_spike_width=cli.soft_spike_width,
        continuous_clip=cli.continuous_clip,
    )
    data = load_cifar10_arrays(args, cfg)
    arrays = {
        "train_images": jnp.asarray(data["train_images"]),
        "train_labels": jnp.asarray(data["train_labels"].astype(int)),
    }
    eval_labels = np.asarray(data["eval_labels"].astype(int))

    head_w = np.asarray(params0["head/w"])
    head_b = np.asarray(params0["head/b"])
    if cli.init_head_state:
        head_state = np.load(cli.init_head_state)
        head_w = np.asarray(head_state["head_w"], dtype=np.int8)
        head_b = np.asarray(head_state["head_b"], dtype=np.int8)

    output_dir = Path(cli.output_dir or f"runs/{cli.preset}_adapter_es")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "adapter_state.npz"
    results_path = output_dir / "results.json"
    spec = adapter_specs(cfg.dim, cli.bottleneck)

    def load_adapter_state(path: str) -> dict[str, np.ndarray]:
        loaded = np.load(path)
        missing = [name for name in spec if name not in loaded]
        if missing:
            raise ValueError(f"adapter state {path} is missing keys: {missing}")
        state_loaded = {name: loaded[name].astype(np.int8) for name in spec}
        for name, value in state_loaded.items():
            if value.shape != spec[name].shape:
                raise ValueError(
                    f"adapter state shape mismatch for {name}: "
                    f"{value.shape} != {spec[name].shape}"
                )
        return state_loaded

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

    def adapter_features_np(state: dict[str, np.ndarray], feats: np.ndarray) -> np.ndarray:
        down_w = state["down_w"].astype(np.float32) / PARAM_SCALE
        down_b = state["down_b"].astype(np.float32) / PARAM_SCALE
        up_w = state["up_w"].astype(np.float32) / PARAM_SCALE
        up_b = state["up_b"].astype(np.float32) / PARAM_SCALE
        hidden = np.maximum(feats @ down_w + down_b, 0.0)
        return feats + cli.adapter_scale * (hidden @ up_w + up_b)

    def logits_from_adapted_np(state: dict[str, np.ndarray], adapted: np.ndarray) -> np.ndarray:
        head_w_np = state["head_w"].astype(np.float32) / PARAM_SCALE
        head_b_np = state["head_b"].astype(np.float32) / PARAM_SCALE
        return adapted @ head_w_np + head_b_np

    def forward_np(state: dict[str, np.ndarray], feats: np.ndarray) -> np.ndarray:
        return logits_from_adapted_np(state, adapter_features_np(state, feats))

    def plastic_head_np(
        state: dict[str, np.ndarray],
        support_feats: np.ndarray,
        support_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        head_w_np = state["head_w"].astype(np.float32) / PARAM_SCALE
        head_b_np = state["head_b"].astype(np.float32) / PARAM_SCALE
        if cli.plastic_head_lr <= 0.0:
            return head_w_np, head_b_np
        support_adapted = adapter_features_np(state, support_feats)
        logits = support_adapted @ head_w_np + head_b_np
        probs = softmax_np(logits)
        target = np.eye(cfg.num_classes, dtype=np.float32)[support_labels]
        residual = target - probs
        head_w_np = head_w_np + cli.plastic_head_lr * (support_adapted.T @ residual) / max(len(support_labels), 1)
        head_b_np = head_b_np + cli.plastic_head_lr * residual.mean(axis=0)
        return head_w_np, head_b_np

    def eval_state_plastic(
        state: dict[str, np.ndarray],
        support_feats: np.ndarray,
        support_labels: np.ndarray,
        query_feats: np.ndarray,
        query_labels: np.ndarray,
    ) -> tuple[float, float]:
        head_w_np, head_b_np = plastic_head_np(state, support_feats, support_labels)
        query_adapted = adapter_features_np(state, query_feats)
        logits = query_adapted @ head_w_np + head_b_np
        losses = ce_vector(logits, query_labels)
        return float(losses.mean()), float((logits.argmax(axis=-1) == query_labels).mean() * 100.0)

    def eval_state(state: dict[str, np.ndarray], feats: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        logits = forward_np(state, feats)
        losses = ce_vector(logits, labels)
        return float(losses.mean()), float((logits.argmax(axis=-1) == labels).mean() * 100.0)

    train_head = bool(cli.train_head)

    @jax.jit
    def loss_chunk(feats, labels, down_w_q, down_b_q, up_w_q, up_b_q, head_w_q, head_b_q, epoch, start):
        idx = start + jnp.arange(cli.chunk, dtype=jnp.int32)
        keys = {name: jax.vmap(lambda i, s=ps: noise_key(cli.seed, epoch, i, s))(idx) for name, ps in spec.items()}
        n_down_w = jax.vmap(lambda key: matrix_noise(spec["down_w"], key, 1))(keys["down_w"])
        n_down_b = jax.vmap(lambda key: matrix_noise(spec["down_b"], key, 1))(keys["down_b"])
        n_up_w = jax.vmap(lambda key: matrix_noise(spec["up_w"], key, 1))(keys["up_w"])
        n_up_b = jax.vmap(lambda key: matrix_noise(spec["up_b"], key, 1))(keys["up_b"])
        n_head_w = jax.vmap(lambda key: matrix_noise(spec["head_w"], key, 1))(keys["head_w"])
        n_head_b = jax.vmap(lambda key: matrix_noise(spec["head_b"], key, 1))(keys["head_b"])

        def losses_for(sign):
            down_w = down_w_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_down_w
            down_b = down_b_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_down_b
            up_w = up_w_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_up_w
            up_b = up_b_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_up_b
            if train_head:
                head_w = head_w_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_head_w
                head_b = head_b_q.astype(jnp.float32) / PARAM_SCALE + sign * cli.sigma * n_head_b
            else:
                head_w = jnp.broadcast_to(head_w_q.astype(jnp.float32) / PARAM_SCALE, (cli.chunk,) + spec["head_w"].shape)
                head_b = jnp.broadcast_to(head_b_q.astype(jnp.float32) / PARAM_SCALE, (cli.chunk,) + spec["head_b"].shape)

            hidden = jnp.maximum(jnp.einsum("bd,pdh->pbh", feats, down_w) + down_b[:, None, :], 0.0)
            adapted = feats[None, :, :] + cli.adapter_scale * (
                jnp.einsum("pbh,phd->pbd", hidden, up_w) + up_b[:, None, :]
            )
            labels_for_loss = labels
            if cli.plastic_head_lr > 0.0:
                support_adapted = adapted[:, :support_size, :]
                query_adapted = adapted[:, support_size:, :]
                support_labels = labels[:support_size]
                labels_for_loss = labels[support_size:]
                support_logits = jnp.einsum("pbd,pdc->pbc", support_adapted, head_w) + head_b[:, None, :]
                probs = jax.nn.softmax(support_logits, axis=-1)
                target = jax.nn.one_hot(support_labels, cfg.num_classes, dtype=jnp.float32)[None, :, :]
                residual = target - probs
                head_w = head_w + cli.plastic_head_lr * (
                    jnp.einsum("pbd,pbc->pdc", support_adapted, residual) / float(support_size)
                )
                head_b = head_b + cli.plastic_head_lr * jnp.mean(residual, axis=1)
                adapted = query_adapted
            logits = jnp.einsum("pbd,pdc->pbc", adapted, head_w) + head_b[:, None, :]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            picked = jnp.take_along_axis(log_probs, labels_for_loss[None, :, None], axis=2).squeeze(-1)
            return -jnp.mean(picked, axis=1)

        return losses_for(1.0), losses_for(-1.0)

    @jax.jit
    def z_chunk(weights, epoch, start):
        idx = start + jnp.arange(cli.chunk, dtype=jnp.int32)

        def z_for(ps):
            keys = jax.vmap(lambda i: noise_key(cli.seed, epoch, i, ps))(idx)
            noises = jax.vmap(lambda key: matrix_noise(ps, key, 1))(keys)
            return (weights.reshape((-1,) + (1,) * len(ps.shape)) * noises).sum(axis=0)

        z = {name: z_for(ps) for name, ps in spec.items()}
        if not train_head:
            z["head_w"] = jnp.zeros(spec["head_w"].shape, dtype=jnp.float32)
            z["head_b"] = jnp.zeros(spec["head_b"].shape, dtype=jnp.float32)
        return z

    print(
        f"preset={cli.preset} adapter_rank bottleneck={cli.bottleneck} "
        f"population={cli.population_size} batch={cli.batch_size} sigma={cli.sigma} "
        f"update_fraction={cli.update_fraction} train_head={cli.train_head} "
        f"lif_mode={cli.lif_mode} plastic_head_lr={cli.plastic_head_lr}",
        flush=True,
    )
    print("Precomputing eval features...", flush=True)
    eval_features = np.concatenate(
        [features(jnp.asarray(data["eval_images"][start : start + 256])) for start in range(0, len(data["eval_images"]), 256)],
        axis=0,
    )
    print(f"eval_features={eval_features.shape}", flush=True)

    if state_path.exists() and results_path.exists():
        loaded = np.load(state_path)
        state = {name: loaded[name].astype(np.int8) for name in spec}
        results = json.loads(results_path.read_text())
        start_epoch = int(results[-1]["epoch"]) + 1 if results and results[-1]["epoch"] >= 0 else 0
        best_acc = max(float(row["test_acc"]) for row in results)
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)
    else:
        if cli.init_adapter_state:
            state = load_adapter_state(cli.init_adapter_state)
        else:
            state = init_adapter_state(cfg.dim, cli.bottleneck, cli.seed, head_w, head_b)
        results = []
        test_loss, test_acc = eval_state(state, eval_features, eval_labels)
        initial_row = {"epoch": -1, "test_loss": test_loss, "test_acc": test_acc}
        if cli.plastic_head_lr > 0.0:
            support_images, support_labels = train_batch(args, cfg, arrays, -1)
            support_features = features(support_images)
            support_labels_np = np.asarray(support_labels).astype(np.int32)
            test_plastic_loss, test_plastic_acc = eval_state_plastic(
                state,
                support_features[:support_size],
                support_labels_np[:support_size],
                eval_features,
                eval_labels,
            )
            initial_row["test_plastic_loss"] = test_plastic_loss
            initial_row["test_plastic_acc"] = test_plastic_acc
        results.append(initial_row)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(state_path, **state)
        np.savez_compressed(output_dir / "best_state.npz", **state)
        best_acc = float(initial_row.get("test_plastic_acc", test_acc))
        start_epoch = 0
        if cli.plastic_head_lr > 0.0:
            print(
                f"Epoch -1: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}% "
                f"plastic_test_loss={initial_row['test_plastic_loss']:.4f} "
                f"plastic_test_acc={initial_row['test_plastic_acc']:.2f}%",
                flush=True,
            )
        else:
            print(f"Epoch -1: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%", flush=True)

    for epoch in range(start_epoch, cli.epochs):
        start_time = time.time()
        train_images, train_labels = train_batch(args, cfg, arrays, epoch)
        train_features = features(train_images)
        train_labels_np = np.asarray(train_labels).astype(np.int32)
        train_loss_before, train_acc_before = eval_state(state, train_features, train_labels_np)

        feats_j = jnp.asarray(train_features)
        labels_j = jnp.asarray(train_labels_np)
        q = {name: jnp.asarray(value) for name, value in state.items()}
        n_pairs = cli.population_size // 2
        plus = np.empty(n_pairs, dtype=np.float32)
        minus = np.empty(n_pairs, dtype=np.float32)
        for start in range(0, n_pairs, cli.chunk):
            p, m = loss_chunk(
                feats_j,
                labels_j,
                q["down_w"],
                q["down_b"],
                q["up_w"],
                q["up_b"],
                q["head_w"],
                q["head_b"],
                jnp.int32(epoch),
                jnp.int32(start),
            )
            plus[start : start + cli.chunk] = np.asarray(p)
            minus[start : start + cli.chunk] = np.asarray(m)

        pair_weights = centered_rank_pair_weights(plus, minus).astype(np.float32)
        z_total = {name: np.zeros(ps.shape, dtype=np.float32) for name, ps in spec.items()}
        for start in range(0, n_pairs, cli.chunk):
            z = z_chunk(jnp.asarray(pair_weights[start : start + cli.chunk]), jnp.int32(epoch), jnp.int32(start))
            for name in z_total:
                z_total[name] += np.asarray(z[name])

        changed = {}
        for name, value in state.items():
            z = z_total[name]
            if cli.update_fraction < 1.0:
                threshold = np.percentile(np.abs(z).reshape(-1), 100.0 * (1.0 - cli.update_fraction))
                mask = np.abs(z) >= threshold
            else:
                mask = np.abs(z) > 0
            updated = np.clip(
                value.astype(np.int16) + np.where(mask, np.sign(z).astype(np.int16) * cli.param_step, 0),
                PARAM_MIN,
                PARAM_MAX,
            ).astype(np.int8)
            changed[name] = float(np.mean(updated != value))
            state[name] = updated

        train_loss_after, train_acc_after = eval_state(state, train_features, train_labels_np)
        test_loss, test_acc = eval_state(state, eval_features, eval_labels)
        plastic_metrics = {}
        if cli.plastic_head_lr > 0.0:
            train_plastic_loss, train_plastic_acc = eval_state_plastic(
                state,
                train_features[:support_size],
                train_labels_np[:support_size],
                train_features[support_size:],
                train_labels_np[support_size:],
            )
            test_plastic_loss, test_plastic_acc = eval_state_plastic(
                state,
                train_features[:support_size],
                train_labels_np[:support_size],
                eval_features,
                eval_labels,
            )
            plastic_metrics = {
                "train_plastic_loss": train_plastic_loss,
                "train_plastic_acc": train_plastic_acc,
                "test_plastic_loss": test_plastic_loss,
                "test_plastic_acc": test_plastic_acc,
            }
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
            "changed": changed,
        }
        row.update(plastic_metrics)
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(state_path, **state)
        selection_acc = float(plastic_metrics.get("test_plastic_acc", test_acc))
        if selection_acc >= best_acc:
            best_acc = selection_acc
            np.savez_compressed(output_dir / "best_state.npz", **state)
        print(
            f"Epoch {epoch:>2}: train {train_loss_before:.4f}->{train_loss_after:.4f} "
            f"acc {train_acc_before:.2f}->{train_acc_after:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% time={seconds:.1f}s",
            flush=True,
        )
        if cli.plastic_head_lr > 0.0:
            print(
                f"          plastic train_query_loss={plastic_metrics['train_plastic_loss']:.4f} "
                f"train_query_acc={plastic_metrics['train_plastic_acc']:.2f}% "
                f"test_loss={plastic_metrics['test_plastic_loss']:.4f} "
                f"test_acc={plastic_metrics['test_plastic_acc']:.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
