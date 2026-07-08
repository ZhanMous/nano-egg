"""Multi-GPU calibrated Spikformer ES run with population ~= 1M.

This is a clean successor to the older multigpu_v3/pop4m scripts:
- uses the calibrated init/head-centering in experiments.spikformer_es_smoke
- does not touch GPU 0 when launched with CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
- pads pair shards so exact POP_SIZE=1,048,576 works on 7 visible GPUs
- writes to an isolated output directory and resumes only from that directory
"""
import json
import math
import os
import pickle
import sys
import threading
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
    cross_entropy,
    forward,
    init_params,
    load_cifar10_arrays,
    loss_core,
    make_specs,
    matrix_noise,
    noise_key,
    preset_config,
    spec_in_update_scope,
    train_batch,
)


POP_SIZE = int(os.environ.get("POP_SIZE", "1048576"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
SEED = int(os.environ.get("SEED", "0"))
CHUNK = int(os.environ.get("CHUNK", "64"))
PRESET = os.environ.get("PRESET", "spikformer_4_256")
SIGMA = float(os.environ.get("SIGMA", "0.05"))
UPDATE_FRACTION = float(os.environ.get("UPDATE_FRACTION", "0.01"))
PARAM_STEP = int(os.environ.get("PARAM_STEP", "1"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "runs/spikformer4_1m_calibrated_top1"))
EVAL_CHUNK = int(os.environ.get("EVAL_CHUNK", "64"))
UPDATE_SCOPE = os.environ.get("UPDATE_SCOPE", "all")
INIT_HEAD_STATE = os.environ.get("INIT_HEAD_STATE", "")


def main() -> None:
    if POP_SIZE % 2 != 0:
        raise ValueError("POP_SIZE must be even for antithetic pairs")
    if CHUNK <= 0:
        raise ValueError("CHUNK must be positive")

    n_devices = jax.device_count()
    devices = jax.devices()
    print(f"visible_devices={n_devices}: {devices}", flush=True)
    if n_devices < 1:
        raise RuntimeError("No JAX devices visible")

    cfg = preset_config(PRESET)
    specs = make_specs(cfg)
    active_specs = [spec for spec in specs if spec_in_update_scope(spec.name, UPDATE_SCOPE)]
    if not active_specs:
        raise ValueError(f"UPDATE_SCOPE matched no parameters: {UPDATE_SCOPE}")
    args = Args(
        preset=PRESET,
        data_source="cifar10",
        seed=SEED,
        epochs=EPOCHS,
        population_size=POP_SIZE,
        batch_size=BATCH_SIZE,
        sigma=SIGMA,
        noise_rank=1,
        update_fraction=UPDATE_FRACTION,
        param_step=PARAM_STEP,
        cifar_train_samples=50000,
        cifar_eval_samples=10000,
        use_jit=True,
        eval_mode="vmap",
        vmap_chunk=CHUNK,
        profile_only=False,
        output_dir=str(OUTPUT_DIR),
        save_every=1,
        save_initial=True,
        cifar_cache_dir="cached_files/cifar10_jax",
        update_scope=UPDATE_SCOPE,
    )
    arrays = load_cifar10_arrays(args, cfg)
    params = init_params(specs, SEED)
    if INIT_HEAD_STATE:
        state = np.load(INIT_HEAD_STATE)
        params["head/w"] = jnp.asarray(state["head_w"], dtype=DTYPE)
        params["head/b"] = jnp.asarray(state["head_b"], dtype=DTYPE)
        print(f"Loaded initial head state from {INIT_HEAD_STATE}", flush=True)
    spec_by_name = {spec.name: spec for spec in specs}

    n_pairs = POP_SIZE // 2
    pairs_per_device = int(math.ceil(n_pairs / (n_devices * CHUNK)) * CHUNK)
    padded_pairs = pairs_per_device * n_devices
    all_pair_indices = np.arange(padded_pairs, dtype=np.int32)
    valid_mask = all_pair_indices < n_pairs
    all_pair_indices = np.where(valid_mask, all_pair_indices, 0)
    device_pair_indices = all_pair_indices.reshape(n_devices, pairs_per_device)
    device_valid_mask = valid_mask.reshape(n_devices, pairs_per_device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(
            {
                "preset": PRESET,
                "pop_size": POP_SIZE,
                "n_pairs": n_pairs,
                "n_devices": n_devices,
                "pairs_per_device_padded": pairs_per_device,
                "padded_pairs": padded_pairs,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "sigma": SIGMA,
                "update_fraction": UPDATE_FRACTION,
                "param_step": PARAM_STEP,
                "chunk": CHUNK,
                "seed": SEED,
                "output_dir": str(OUTPUT_DIR),
                "update_scope": UPDATE_SCOPE,
                "n_active_specs": len(active_specs),
                "init_head_state": INIT_HEAD_STATE,
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"config preset={PRESET} pop={POP_SIZE} batch={BATCH_SIZE} epochs={EPOCHS} "
        f"sigma={SIGMA} update_fraction={UPDATE_FRACTION} scope={UPDATE_SCOPE} chunk={CHUNK}",
        flush=True,
    )
    print(
        f"pairs={n_pairs}, padded_pairs={padded_pairs}, "
        f"pairs/device padded={pairs_per_device}",
        flush=True,
    )

    def make_local_gradient_fn():
        n_local = pairs_per_device

        def local_gradient(params_d, images, labels, epoch, pair_indices, mask):
            epoch_arr = jnp.int32(epoch)

            def chunk_losses(carry, chunk_start):
                plus_acc, minus_acc = carry
                chunk_idx = jax.lax.dynamic_slice(pair_indices, (chunk_start,), (CHUNK,))

                def eval_pair(idx):
                    plus_loss = loss_core(params_d, spec_by_name, images, labels, cfg, args, epoch_arr, idx, 1)
                    minus_loss = loss_core(params_d, spec_by_name, images, labels, cfg, args, epoch_arr, idx, -1)
                    return plus_loss, minus_loss

                plus_losses, minus_losses = jax.vmap(eval_pair)(chunk_idx)
                plus_acc = jax.lax.dynamic_update_slice(plus_acc, plus_losses, (chunk_start,))
                minus_acc = jax.lax.dynamic_update_slice(minus_acc, minus_losses, (chunk_start,))
                return (plus_acc, minus_acc), None

            starts = jnp.arange(0, n_local, CHUNK, dtype=jnp.int32)
            plus_init = jnp.zeros(n_local, dtype=jnp.float32)
            minus_init = jnp.zeros(n_local, dtype=jnp.float32)
            (plus, minus), _ = jax.lax.scan(chunk_losses, (plus_init, minus_init), starts)
            advantages = jnp.sign(minus - plus) * mask.astype(jnp.float32)

            def compute_local_z(spec):
                def pair_z(carry, chunk_start):
                    z_local = carry
                    adv = jax.lax.dynamic_slice(advantages, (chunk_start,), (CHUNK,))
                    idx = jax.lax.dynamic_slice(pair_indices, (chunk_start,), (CHUNK,))
                    keys = jax.vmap(lambda i: noise_key(args.seed, epoch_arr, i, spec))(idx)
                    noises = jax.vmap(lambda key: matrix_noise(spec, key, args.noise_rank))(keys)
                    bshape = (-1,) + (1,) * len(spec.shape)
                    return z_local + (adv.reshape(bshape) * noises).sum(axis=0), None

                z0 = jnp.zeros(spec.shape, dtype=jnp.float32)
                z_final, _ = jax.lax.scan(pair_z, z0, starts)
                return z_final

            local_zs = {spec.name: compute_local_z(spec) for spec in active_specs}
            adv_nonzero = jnp.mean(advantages != 0)
            return local_zs, adv_nonzero

        return jax.jit(local_gradient)

    local_gradient_fn = make_local_gradient_fn()

    device_pair_indices_jax = [
        jax.device_put(jnp.asarray(device_pair_indices[d]), devices[d])
        for d in range(n_devices)
    ]
    device_valid_mask_jax = [
        jax.device_put(jnp.asarray(device_valid_mask[d]), devices[d])
        for d in range(n_devices)
    ]

    def eval_chunked(p, n_eval=10000, chunk=EVAL_CHUNK):
        eval_images = arrays["eval_images"]
        eval_labels = arrays["eval_labels"]
        correct = 0
        total = 0
        total_loss = 0.0
        for start in range(0, n_eval, chunk):
            end = min(start + chunk, n_eval)
            images = jnp.asarray(eval_images[start:end])
            labels = jnp.asarray(eval_labels[start:end])
            logits = forward(p, specs, images, cfg, args)
            count = end - start
            correct += int(jnp.sum(jnp.argmax(logits, axis=-1) == labels))
            total += count
            total_loss += float(cross_entropy(logits, labels)) * count
        return total_loss / total, 100.0 * correct / total

    print("Running initial eval...", flush=True)
    init_loss, init_acc = eval_chunked(params)
    print(f"Initial: loss={init_loss:.4f}, acc={init_acc:.2f}%", flush=True)

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    results_path = OUTPUT_DIR / "results.json"
    start_epoch = 0
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if results and results[-1]["epoch"] >= 0:
            last_epoch = int(results[-1]["epoch"])
            ckpt_path = ckpt_dir / f"epoch_{last_epoch:04d}.pkl"
            if ckpt_path.exists():
                with ckpt_path.open("rb") as f:
                    params = pickle.load(f)["params"]
                start_epoch = last_epoch + 1
                print(f"Resumed from epoch {last_epoch}, starting at {start_epoch}", flush=True)
    else:
        results = [{"epoch": -1, "test_loss": init_loss, "test_acc": init_acc, "n_devices": n_devices, "pop": POP_SIZE}]
        results_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"Training epochs {start_epoch}-{EPOCHS - 1}", flush=True)
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        train_images, train_labels = train_batch(args, cfg, arrays, epoch)
        images_np = np.asarray(train_images)
        labels_np = np.asarray(train_labels)
        local_results = [None] * n_devices
        local_adv = [None] * n_devices

        def compute_on_device(device_idx: int) -> None:
            device = devices[device_idx]
            params_d = jax.device_put(params, device)
            images_d = jax.device_put(jnp.asarray(images_np), device)
            labels_d = jax.device_put(jnp.asarray(labels_np), device)
            epoch_d = jax.device_put(jnp.int32(epoch), device)
            zs, adv_nonzero = local_gradient_fn(
                params_d,
                images_d,
                labels_d,
                epoch_d,
                device_pair_indices_jax[device_idx],
                device_valid_mask_jax[device_idx],
            )
            local_results[device_idx] = zs
            local_adv[device_idx] = adv_nonzero

        threads = []
        for device_idx in range(n_devices):
            thread = threading.Thread(target=compute_on_device, args=(device_idx,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        global_zs = {}
        for spec in active_specs:
            shards_np = [np.asarray(local_results[d][spec.name]) for d in range(n_devices)]
            global_zs[spec.name] = jnp.asarray(sum(shards_np))

        new_params = dict(params)
        changed = []
        for spec in active_specs:
            z = global_zs[spec.name]
            abs_z = jnp.abs(z)
            if UPDATE_FRACTION < 1.0:
                threshold = jnp.percentile(abs_z.reshape(-1), 100.0 * (1.0 - UPDATE_FRACTION))
                mask = abs_z >= threshold
            else:
                mask = abs_z > 0
            delta = jnp.where(mask, jnp.sign(z).astype(jnp.int16) * PARAM_STEP, 0)
            updated = jnp.clip(params[spec.name].astype(jnp.int16) + delta, PARAM_MIN, PARAM_MAX).astype(DTYPE)
            new_params[spec.name] = updated
            changed.append(jnp.mean(updated != params[spec.name]))
        params = new_params
        changed_fraction = float(jnp.mean(jnp.asarray(changed)))

        jax.clear_caches()
        test_loss, test_acc = eval_chunked(params)
        seconds = time.time() - start_time
        adv_nonzero = float(np.mean([float(np.asarray(x)) for x in local_adv if x is not None]))
        row = {
            "epoch": epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time": seconds,
            "n_devices": n_devices,
            "pop": POP_SIZE,
            "adv_nonzero": adv_nonzero,
            "changed_fraction": changed_fraction,
        }
        results.append(row)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / f"epoch_{epoch:04d}.pkl").open("wb") as f:
            pickle.dump({"params": params, "epoch": epoch}, f)
        print(
            f"Epoch {epoch}: loss={test_loss:.4f}, acc={test_acc:.2f}%, "
            f"changed={changed_fraction:.5f}, adv_nonzero={adv_nonzero:.3f}, time={seconds:.0f}s",
            flush=True,
        )

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
