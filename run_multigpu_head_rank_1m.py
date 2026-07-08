"""Multi-GPU head-only centered-rank ES for calibrated Spikformer presets."""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

CUDA_NVCC = "/home/zhanshaoji/miniforge3/envs/hyperscalees/lib/python3.12/site-packages/nvidia/cuda_nvcc"
os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={CUDA_NVCC}")
os.environ.setdefault("CUDA_HOME", CUDA_NVCC)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

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


PRESET = os.environ.get("PRESET", "spikformer_4_256")
POP_SIZE = int(os.environ.get("POP_SIZE", "1048576"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
SEED = int(os.environ.get("SEED", "0"))
SIGMA = float(os.environ.get("SIGMA", "0.05"))
CHUNK = int(os.environ.get("CHUNK", "2048"))
PARAM_STEP = int(os.environ.get("PARAM_STEP", "1"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs/spikformer4_head_rank_1m_mgpu"))
FEATURE_CHUNK = int(os.environ.get("FEATURE_CHUNK", "256"))


def ce_vector(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - logsum
    return -log_probs[np.arange(labels.shape[0]), labels]


def eval_head(weight_q: np.ndarray, bias_q: np.ndarray, feats: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    logits = feats @ (weight_q.astype(np.float32) / 16.0) + (bias_q.astype(np.float32) / 16.0)
    losses = ce_vector(logits, labels)
    pred = logits.argmax(axis=-1)
    return float(losses.mean()), float((pred == labels).mean() * 100.0)


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
    if POP_SIZE % 2 != 0:
        raise ValueError("POP_SIZE must be even")
    if CHUNK <= 0:
        raise ValueError("CHUNK must be positive")

    devices = jax.devices()
    n_devices = len(devices)
    if n_devices < 1:
        raise RuntimeError("No JAX devices visible")

    cfg = preset_config(PRESET)
    specs = make_specs(cfg)
    spec_by_name = {spec.name: spec for spec in specs}
    head_w_spec = spec_by_name["head/w"]
    head_b_spec = spec_by_name["head/b"]
    params0 = init_params(specs, SEED)

    args = Args(
        preset=PRESET,
        data_source="cifar10",
        seed=SEED,
        epochs=EPOCHS,
        population_size=POP_SIZE,
        batch_size=BATCH_SIZE,
        sigma=SIGMA,
        noise_rank=1,
        update_fraction=1.0,
        param_step=PARAM_STEP,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(
            {
                "preset": PRESET,
                "pop_size": POP_SIZE,
                "n_pairs": POP_SIZE // 2,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "sigma": SIGMA,
                "chunk": CHUNK,
                "seed": SEED,
                "param_step": PARAM_STEP,
                "n_devices": n_devices,
                "devices": [str(d) for d in devices],
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
        )
        + "\n"
    )

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

    n_pairs = POP_SIZE // 2
    pairs_per_device = int(math.ceil(n_pairs / (n_devices * CHUNK)) * CHUNK)
    padded_pairs = pairs_per_device * n_devices
    all_pair_indices = np.arange(padded_pairs, dtype=np.int32)
    valid_mask = all_pair_indices < n_pairs
    all_pair_indices = np.where(valid_mask, all_pair_indices, 0)
    device_pair_indices = all_pair_indices.reshape(n_devices, pairs_per_device)
    device_valid_mask = valid_mask.reshape(n_devices, pairs_per_device).astype(np.float32)

    device_pair_indices_jax = [jax.device_put(jnp.asarray(device_pair_indices[i]), devices[i]) for i in range(n_devices)]
    device_valid_mask_jax = [jax.device_put(jnp.asarray(device_valid_mask[i]), devices[i]) for i in range(n_devices)]

    def make_loss_fn():
        n_local = pairs_per_device

        def local_losses(feats, labels, weight_q, bias_q, epoch, pair_indices, mask):
            weight = weight_q.astype(jnp.float32) / 16.0
            bias = bias_q.astype(jnp.float32) / 16.0
            base_logits = feats @ weight + bias
            epoch_arr = jnp.int32(epoch)

            def ce_population(logits):
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                picked = jnp.take_along_axis(log_probs, labels[None, :, None], axis=2).squeeze(-1)
                return -jnp.mean(picked, axis=1)

            def chunk_step(carry, start):
                plus_acc, minus_acc = carry
                idx = jax.lax.dynamic_slice(pair_indices, (start,), (CHUNK,))
                valid = jax.lax.dynamic_slice(mask, (start,), (CHUNK,))
                keys_w = jax.vmap(lambda i: noise_key(SEED, epoch_arr, i, head_w_spec))(idx)
                keys_b = jax.vmap(lambda i: noise_key(SEED, epoch_arr, i, head_b_spec))(idx)
                noise_w = jax.vmap(lambda key: matrix_noise(head_w_spec, key, 1))(keys_w)
                noise_b = jax.vmap(lambda key: matrix_noise(head_b_spec, key, 1))(keys_b)
                noise_logits = jnp.einsum("bd,pdc->pbc", feats, noise_w) + noise_b[:, None, :]
                plus = ce_population(base_logits[None, :, :] + SIGMA * noise_logits)
                minus = ce_population(base_logits[None, :, :] - SIGMA * noise_logits)
                plus = plus * valid
                minus = minus * valid
                plus_acc = jax.lax.dynamic_update_slice(plus_acc, plus, (start,))
                minus_acc = jax.lax.dynamic_update_slice(minus_acc, minus, (start,))
                return (plus_acc, minus_acc), None

            starts = jnp.arange(0, n_local, CHUNK, dtype=jnp.int32)
            init = (jnp.zeros(n_local, dtype=jnp.float32), jnp.zeros(n_local, dtype=jnp.float32))
            return jax.lax.scan(chunk_step, init, starts)[0]

        return jax.jit(local_losses)

    def make_z_fn():
        n_local = pairs_per_device

        def local_z(pair_weights, epoch, pair_indices, mask):
            epoch_arr = jnp.int32(epoch)

            def chunk_step(carry, start):
                z_w, z_b = carry
                idx = jax.lax.dynamic_slice(pair_indices, (start,), (CHUNK,))
                weights = jax.lax.dynamic_slice(pair_weights, (start,), (CHUNK,))
                valid = jax.lax.dynamic_slice(mask, (start,), (CHUNK,))
                weights = weights * valid
                keys_w = jax.vmap(lambda i: noise_key(SEED, epoch_arr, i, head_w_spec))(idx)
                keys_b = jax.vmap(lambda i: noise_key(SEED, epoch_arr, i, head_b_spec))(idx)
                noise_w = jax.vmap(lambda key: matrix_noise(head_w_spec, key, 1))(keys_w)
                noise_b = jax.vmap(lambda key: matrix_noise(head_b_spec, key, 1))(keys_b)
                z_w = z_w + (weights[:, None, None] * noise_w).sum(axis=0)
                z_b = z_b + (weights[:, None] * noise_b).sum(axis=0)
                return (z_w, z_b), None

            starts = jnp.arange(0, n_local, CHUNK, dtype=jnp.int32)
            init = (
                jnp.zeros(head_w_spec.shape, dtype=jnp.float32),
                jnp.zeros(head_b_spec.shape, dtype=jnp.float32),
            )
            return jax.lax.scan(chunk_step, init, starts)[0]

        return jax.jit(local_z)

    loss_fn = make_loss_fn()
    z_fn = make_z_fn()

    print(
        f"preset={PRESET} rule=rank_full_mgpu pop={POP_SIZE} batch={BATCH_SIZE} "
        f"sigma={SIGMA} chunk={CHUNK} devices={n_devices}",
        flush=True,
    )
    print("Precomputing eval features...", flush=True)
    eval_features = np.concatenate(
        [
            features(jnp.asarray(data["eval_images"][start : start + FEATURE_CHUNK]))
            for start in range(0, len(data["eval_images"]), FEATURE_CHUNK)
        ],
        axis=0,
    )
    print(f"eval_features={eval_features.shape}", flush=True)

    weight_q = np.asarray(params0["head/w"])
    bias_q = np.asarray(params0["head/b"])
    state_path = OUTPUT_DIR / "head_state.npz"
    results_path = OUTPUT_DIR / "results.json"
    if state_path.exists() and results_path.exists():
        state = np.load(state_path)
        weight_q = state["head_w"].astype(DTYPE)
        bias_q = state["head_b"].astype(DTYPE)
        results = json.loads(results_path.read_text())
        start_epoch = int(results[-1]["epoch"]) + 1 if results and results[-1]["epoch"] >= 0 else 0
        print(f"Resumed from epoch {start_epoch - 1}", flush=True)
    else:
        results = []
        test_loss, test_acc = eval_head(weight_q, bias_q, eval_features, eval_labels)
        results.append({"epoch": -1, "test_loss": test_loss, "test_acc": test_acc})
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Epoch -1: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%", flush=True)
        start_epoch = 0

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        train_images, train_labels = train_batch(args, cfg, arrays, epoch)
        train_features = features(train_images)
        train_labels_np = np.asarray(train_labels)
        train_loss_before, train_acc_before = eval_head(weight_q, bias_q, train_features, train_labels_np)

        feats_d = [jax.device_put(jnp.asarray(train_features), d) for d in devices]
        labels_d = [jax.device_put(jnp.asarray(train_labels_np), d) for d in devices]
        weight_d = [jax.device_put(jnp.asarray(weight_q), d) for d in devices]
        bias_d = [jax.device_put(jnp.asarray(bias_q), d) for d in devices]
        epoch_d = [jax.device_put(jnp.int32(epoch), d) for d in devices]

        local_losses = [
            loss_fn(feats_d[i], labels_d[i], weight_d[i], bias_d[i], epoch_d[i], device_pair_indices_jax[i], device_valid_mask_jax[i])
            for i in range(n_devices)
        ]
        plus = np.concatenate([np.asarray(x[0]) for x in local_losses], axis=0)[:n_pairs]
        minus = np.concatenate([np.asarray(x[1]) for x in local_losses], axis=0)[:n_pairs]
        pair_weights = centered_rank_pair_weights(plus, minus).astype(np.float32)
        padded_weights = np.zeros(padded_pairs, dtype=np.float32)
        padded_weights[:n_pairs] = pair_weights
        device_weights = padded_weights.reshape(n_devices, pairs_per_device)
        device_weights_jax = [jax.device_put(jnp.asarray(device_weights[i]), devices[i]) for i in range(n_devices)]

        local_z = [
            z_fn(device_weights_jax[i], epoch_d[i], device_pair_indices_jax[i], device_valid_mask_jax[i])
            for i in range(n_devices)
        ]
        z_w = sum(np.asarray(x[0]) for x in local_z)
        z_b = sum(np.asarray(x[1]) for x in local_z)

        weight_next = np.clip(
            weight_q.astype(np.int16) + np.sign(z_w).astype(np.int16) * PARAM_STEP,
            PARAM_MIN,
            PARAM_MAX,
        ).astype(DTYPE)
        bias_next = np.clip(
            bias_q.astype(np.int16) + np.sign(z_b).astype(np.int16) * PARAM_STEP,
            PARAM_MIN,
            PARAM_MAX,
        ).astype(DTYPE)
        weight_q, bias_q = weight_next, bias_next

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
            "pop": POP_SIZE,
            "n_devices": n_devices,
        }
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        np.savez_compressed(state_path, head_w=weight_q, head_b=bias_q)
        print(
            f"Epoch {epoch:>2}: train {train_loss_before:.4f}->{train_loss_after:.4f} "
            f"acc {train_acc_before:.2f}->{train_acc_after:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% time={seconds:.1f}s",
            flush=True,
        )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
