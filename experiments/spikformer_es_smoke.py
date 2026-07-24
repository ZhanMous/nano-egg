import json
import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro


FIXED_POINT = 4
PARAM_SCALE = float(2**FIXED_POINT)
DTYPE = jnp.int8
PARAM_MIN = -127
PARAM_MAX = 127


ParamKind = Literal["linear", "conv", "vector"]


@dataclass(frozen=True)
class ModelConfig:
    image_size: int
    in_channels: int
    num_classes: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    patch_size: int
    time_steps: int
    lif_tau: float
    lif_threshold: float
    pool_after: tuple[int, ...]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    shape: tuple[int, ...]
    seed: int


@dataclass
class Args:
    preset: Literal[
        "tiny",
        "smoke",
        "spikformer_2_128",
        "spikformer_4_256",
        "spikformer_dvs_2_256",
    ] = "smoke"
    data_source: Literal["synthetic", "cifar10"] = "synthetic"
    seed: int = 0
    epochs: int = 2
    population_size: int = 4
    batch_size: int = 4
    sigma: float = 0.05
    noise_rank: int = 1
    update_fraction: float = 0.05
    param_step: int = 1
    cifar_cache_dir: str = "cached_files/cifar10_jax"
    cifar_train_samples: int = 512
    cifar_eval_samples: int = 128
    use_jit: bool = True
    eval_mode: Literal["loop", "vmap"] = "loop"
    vmap_chunk: int = 256
    profile_only: bool = False
    output_dir: str = ""
    save_every: int = 1
    save_initial: bool = False
    head_feature_centering: Literal["none", "batch"] = "batch"
    update_scope: str = "all"
    lif_mode: Literal["hard_spike", "soft_spike", "leaky_clip", "leaky_tanh"] = "hard_spike"
    soft_spike_width: float = 0.25
    continuous_clip: float = 1.0


def preset_config(name: str) -> ModelConfig:
    if name == "spikformer_dvs_2_256":
        return ModelConfig(
            image_size=128,
            in_channels=2,
            num_classes=10,
            dim=256,
            depth=2,
            num_heads=16,
            mlp_ratio=4,
            patch_size=16,
            time_steps=10,
            lif_tau=2.0,
            lif_threshold=0.5,
            pool_after=(0, 1, 2, 3),
        )
    if name == "spikformer_4_256":
        return ModelConfig(
            image_size=32,
            in_channels=3,
            num_classes=10,
            dim=256,
            depth=4,
            num_heads=8,
            mlp_ratio=4,
            patch_size=4,
            time_steps=4,
            lif_tau=2.0,
            lif_threshold=0.5,
            pool_after=(2, 3),
        )
    if name == "tiny":
        return ModelConfig(
            image_size=32,
            in_channels=3,
            num_classes=10,
            dim=64,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            patch_size=4,
            time_steps=1,
            lif_tau=2.0,
            lif_threshold=0.5,
            pool_after=(2, 3),
        )
    if name == "spikformer_2_128":
        return ModelConfig(
            image_size=32,
            in_channels=3,
            num_classes=10,
            dim=128,
            depth=2,
            num_heads=4,
            mlp_ratio=4,
            patch_size=4,
            time_steps=2,
            lif_tau=2.0,
            lif_threshold=0.5,
            pool_after=(2, 3),
        )
    if name == "smoke":
        return ModelConfig(
            image_size=32,
            in_channels=3,
            num_classes=10,
            dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            patch_size=4,
            time_steps=2,
            lif_tau=2.0,
            lif_threshold=0.5,
            pool_after=(2, 3),
        )
    raise ValueError(f"unknown preset: {name}")


def name_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def make_specs(cfg: ModelConfig) -> list[ParamSpec]:
    specs: list[ParamSpec] = []

    def add(name: str, kind: ParamKind, shape: tuple[int, ...]) -> None:
        specs.append(ParamSpec(name=name, kind=kind, shape=shape, seed=name_seed(name)))

    d = cfg.dim
    c0 = d // 8
    c1 = d // 4
    c2 = d // 2

    add("sps/proj0/w", "conv", (3, 3, cfg.in_channels, c0))
    add("sps/proj0/gamma", "vector", (c0,))
    add("sps/proj0/beta", "vector", (c0,))
    add("sps/proj1/w", "conv", (3, 3, c0, c1))
    add("sps/proj1/gamma", "vector", (c1,))
    add("sps/proj1/beta", "vector", (c1,))
    add("sps/proj2/w", "conv", (3, 3, c1, c2))
    add("sps/proj2/gamma", "vector", (c2,))
    add("sps/proj2/beta", "vector", (c2,))
    add("sps/proj3/w", "conv", (3, 3, c2, d))
    add("sps/proj3/gamma", "vector", (d,))
    add("sps/proj3/beta", "vector", (d,))
    add("sps/rpe/w", "conv", (3, 3, d, d))
    add("sps/rpe/gamma", "vector", (d,))
    add("sps/rpe/beta", "vector", (d,))

    hidden = d * cfg.mlp_ratio
    for i in range(cfg.depth):
        prefix = f"blocks/{i}"
        for tag in ("q", "k", "v", "proj"):
            add(f"{prefix}/ssa/{tag}/w", "linear", (d, d))
            add(f"{prefix}/ssa/{tag}/b", "vector", (d,))
            add(f"{prefix}/ssa/{tag}/gamma", "vector", (d,))
            add(f"{prefix}/ssa/{tag}/beta", "vector", (d,))
        add(f"{prefix}/mlp/fc1/w", "linear", (d, hidden))
        add(f"{prefix}/mlp/fc1/b", "vector", (hidden,))
        add(f"{prefix}/mlp/fc1/gamma", "vector", (hidden,))
        add(f"{prefix}/mlp/fc1/beta", "vector", (hidden,))
        add(f"{prefix}/mlp/fc2/w", "linear", (hidden, d))
        add(f"{prefix}/mlp/fc2/b", "vector", (d,))
        add(f"{prefix}/mlp/fc2/gamma", "vector", (d,))
        add(f"{prefix}/mlp/fc2/beta", "vector", (d,))
        # The official CIFAR block allocates LayerNorm parameters but does not
        # call them in forward. Keeping them preserves the published count.
        add(f"{prefix}/norm1/gamma", "vector", (d,))
        add(f"{prefix}/norm1/beta", "vector", (d,))
        add(f"{prefix}/norm2/gamma", "vector", (d,))
        add(f"{prefix}/norm2/beta", "vector", (d,))

    add("head/norm/gamma", "vector", (d,))
    add("head/norm/beta", "vector", (d,))
    add("head/w", "linear", (d, cfg.num_classes))
    add("head/b", "vector", (cfg.num_classes,))
    return specs


def init_params(specs: list[ParamSpec], seed: int) -> dict[str, jax.Array]:
    params = {}
    base_key = jax.random.PRNGKey(seed)
    for spec in specs:
        key = jax.random.fold_in(base_key, spec.seed)
        if spec.name.endswith("/gamma"):
            value = jnp.ones(spec.shape, dtype=DTYPE) * int(PARAM_SCALE)
        elif spec.name.endswith("/beta") or spec.name.endswith("/b"):
            value = jnp.zeros(spec.shape, dtype=DTYPE)
        elif spec.name == "head/w":
            # Small random init in the quantized grid. With fixed-point scale 16,
            # +/-2 gives dequant std 0.125; batch feature centering below keeps
            # logits calibrated while avoiding an under-dispersed head.
            value = jax.random.choice(key, jnp.array([-2, 2], dtype=DTYPE), shape=spec.shape)
        else:
            value = jnp.clip(
                jnp.rint(jax.random.normal(key, spec.shape) * PARAM_SCALE),
                PARAM_MIN,
                PARAM_MAX,
            ).astype(DTYPE)
        params[spec.name] = value
    return params


def dequant(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32) / PARAM_SCALE


def noise_key(seed: int, epoch: int, pair_idx: int, spec: ParamSpec) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, epoch)
    key = jax.random.fold_in(key, pair_idx)
    return jax.random.fold_in(key, spec.seed)


def matrix_noise(spec: ParamSpec, key: jax.Array, rank: int) -> jax.Array:
    if spec.kind == "linear":
        fan_in, fan_out = spec.shape
        a_key, b_key = jax.random.split(key)
        a = jax.random.normal(a_key, (fan_in, rank))
        b = jax.random.normal(b_key, (fan_out, rank))
        return (a @ b.T) / math.sqrt(float(fan_in * rank))

    if spec.kind == "conv":
        kh, kw, fan_in_ch, fan_out = spec.shape
        fan_in = kh * kw * fan_in_ch
        a_key, b_key = jax.random.split(key)
        a = jax.random.normal(a_key, (fan_in, rank))
        b = jax.random.normal(b_key, (fan_out, rank))
        return ((a @ b.T) / math.sqrt(float(fan_in * rank))).reshape(spec.shape)

    size = int(np.prod(spec.shape))
    return jax.random.normal(key, spec.shape) / math.sqrt(float(max(size, 1)))


def maybe_noisy(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    name: str,
    seed: int,
    epoch: int,
    pair_idx: int | None,
    sign: int,
    sigma: float,
    rank: int,
) -> jax.Array:
    p = dequant(params[name])
    if pair_idx is None:
        return p
    spec = spec_by_name[name]
    return p + sign * sigma * matrix_noise(spec, noise_key(seed, epoch, pair_idx, spec), rank)


def spec_in_update_scope(name: str, scope: str) -> bool:
    """Return whether a parameter participates in ES perturbation/update."""
    tokens = {token.strip() for token in scope.split(",") if token.strip()}
    if not tokens or "all" in tokens:
        return True

    for token in tokens:
        if token == "head" and name.startswith("head/"):
            return True
        if token == "head_linear" and name in ("head/w", "head/b"):
            return True
        if token == "head_norm" and name.startswith("head/norm/"):
            return True
        if token == "sps" and name.startswith("sps/"):
            return True
        if token == "blocks" and name.startswith("blocks/"):
            return True
        if token.startswith("block") and token[5:].isdigit():
            if name.startswith(f"blocks/{int(token[5:])}/"):
                return True
        if name == token or name.startswith(token.rstrip("/") + "/"):
            return True
    return False


def conv2d(x: jax.Array, w: jax.Array) -> jax.Array:
    tb, h, w_in, c = x.shape
    del tb, h, w_in, c
    return jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def maxpool2d(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 3, 3, 1),
        window_strides=(1, 2, 2, 1),
        padding="SAME",
    )


def affine_channels(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    return x * gamma + beta


def layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    """Applied over the last dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return x_norm * gamma + beta


def lif(x: jax.Array, cfg: ModelConfig, args: Args) -> jax.Array:
    def step(v, current):
        v = v + (current - v) / cfg.lif_tau
        if args.lif_mode == "hard_spike":
            out = (v >= cfg.lif_threshold).astype(jnp.float32)
            v_next = v * (1.0 - out)
        elif args.lif_mode == "soft_spike":
            width = jnp.asarray(max(args.soft_spike_width, 1e-6), dtype=jnp.float32)
            out = jnp.clip(0.5 + (v - cfg.lif_threshold) / width, 0.0, 1.0)
            v_next = v * (1.0 - out)
        elif args.lif_mode == "leaky_clip":
            scale = jnp.asarray(max(cfg.lif_threshold, 1e-6), dtype=jnp.float32)
            out = jnp.clip(v / scale, -args.continuous_clip, args.continuous_clip)
            v_next = v
        elif args.lif_mode == "leaky_tanh":
            scale = jnp.asarray(max(cfg.lif_threshold, 1e-6), dtype=jnp.float32)
            out = jnp.tanh(v / scale)
            v_next = v
        else:
            raise ValueError(f"unknown lif_mode: {args.lif_mode}")
        return v_next, out

    _, spikes = jax.lax.scan(step, jnp.zeros_like(x[0]), x)
    return spikes


def fetch(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    name: str,
    args: Args,
    epoch: int,
    pair_idx: int | None,
    sign: int,
) -> jax.Array:
    if not spec_in_update_scope(name, args.update_scope):
        return dequant(params[name])
    return maybe_noisy(
        params,
        spec_by_name,
        name,
        args.seed,
        epoch,
        pair_idx,
        sign,
        args.sigma,
        args.noise_rank,
    )


def sps_forward(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    images: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch: int,
    pair_idx: int | None,
    sign: int,
) -> jax.Array:
    if images.ndim == 4:
        x = jnp.repeat(images[None], cfg.time_steps, axis=0)
    elif images.ndim == 5:
        if images.shape[1] != cfg.time_steps:
            raise ValueError(
                f"event input has T={images.shape[1]}, expected cfg.time_steps={cfg.time_steps}"
            )
        if images.shape[-1] != cfg.in_channels:
            raise ValueError(
                f"event input has C={images.shape[-1]}, expected cfg.in_channels={cfg.in_channels}"
            )
        x = jnp.transpose(images, (1, 0, 2, 3, 4))
    else:
        raise ValueError(
            "images must be [B,H,W,C] static frames or [B,T,H,W,C] event frames, "
            f"got {images.shape}"
        )
    t, b, h, w, c = x.shape
    x = x.reshape((t * b, h, w, c))

    for idx in range(4):
        prefix = f"sps/proj{idx}"
        x = conv2d(x, fetch(params, spec_by_name, f"{prefix}/w", args, epoch, pair_idx, sign))
        gamma = fetch(params, spec_by_name, f"{prefix}/gamma", args, epoch, pair_idx, sign)
        beta = fetch(params, spec_by_name, f"{prefix}/beta", args, epoch, pair_idx, sign)
        _, h_now, w_now, ch_now = x.shape
        x = affine_channels(x, gamma, beta).reshape((t, b, h_now, w_now, ch_now))
        x = lif(x, cfg, args).reshape((t * b, h_now, w_now, ch_now))
        if idx in cfg.pool_after:
            x = maxpool2d(x)

    _, h_now, w_now, ch_now = x.shape
    x_feat = x.reshape((t, b, h_now, w_now, ch_now))
    rpe = conv2d(x, fetch(params, spec_by_name, "sps/rpe/w", args, epoch, pair_idx, sign))
    rpe = affine_channels(
        rpe,
        fetch(params, spec_by_name, "sps/rpe/gamma", args, epoch, pair_idx, sign),
        fetch(params, spec_by_name, "sps/rpe/beta", args, epoch, pair_idx, sign),
    ).reshape((t, b, h_now, w_now, ch_now))
    x = lif(rpe, cfg, args) + x_feat
    return x.reshape((t, b, h_now * w_now, ch_now))


def linear_tokens(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.einsum("tbnc,co->tbno", x, w) + b


def ssa_forward(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    x: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch: int,
    pair_idx: int | None,
    sign: int,
    block_idx: int,
) -> jax.Array:
    prefix = f"blocks/{block_idx}/ssa"
    qkv = []
    for tag in ("q", "k", "v"):
        pfx = f"{prefix}/{tag}"
        y = linear_tokens(
            x,
            fetch(params, spec_by_name, f"{pfx}/w", args, epoch, pair_idx, sign),
            fetch(params, spec_by_name, f"{pfx}/b", args, epoch, pair_idx, sign),
        )
        y = affine_channels(
            y,
            fetch(params, spec_by_name, f"{pfx}/gamma", args, epoch, pair_idx, sign),
            fetch(params, spec_by_name, f"{pfx}/beta", args, epoch, pair_idx, sign),
        )
        qkv.append(lif(y, cfg, args))

    q, k, v = qkv
    t, b, n, c = q.shape
    head_dim = c // cfg.num_heads
    q = q.reshape((t, b, n, cfg.num_heads, head_dim)).transpose((0, 1, 3, 2, 4))
    k = k.reshape((t, b, n, cfg.num_heads, head_dim)).transpose((0, 1, 3, 2, 4))
    v = v.reshape((t, b, n, cfg.num_heads, head_dim)).transpose((0, 1, 3, 2, 4))
    attn = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * 0.125
    y = jnp.matmul(attn, v).transpose((0, 1, 3, 2, 4)).reshape((t, b, n, c))
    y = lif(y, cfg, args)

    pfx = f"{prefix}/proj"
    y = linear_tokens(
        y,
        fetch(params, spec_by_name, f"{pfx}/w", args, epoch, pair_idx, sign),
        fetch(params, spec_by_name, f"{pfx}/b", args, epoch, pair_idx, sign),
    )
    y = affine_channels(
        y,
        fetch(params, spec_by_name, f"{pfx}/gamma", args, epoch, pair_idx, sign),
        fetch(params, spec_by_name, f"{pfx}/beta", args, epoch, pair_idx, sign),
    )
    # Apply norm1 before LIF
    block_prefix = "/".join(prefix.split("/")[:-1])  # blocks/{i}/ssa -> blocks/{i}
    norm1_g = fetch(params, spec_by_name, f"{block_prefix}/norm1/gamma", args, epoch, pair_idx, sign)
    norm1_b = fetch(params, spec_by_name, f"{block_prefix}/norm1/beta", args, epoch, pair_idx, sign)
    y = layer_norm(y, norm1_g, norm1_b)
    return lif(y, cfg, args)


def mlp_forward(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    x: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch: int,
    pair_idx: int | None,
    sign: int,
    block_idx: int,
) -> jax.Array:
    prefix = f"blocks/{block_idx}/mlp"
    for tag in ("fc1", "fc2"):
        pfx = f"{prefix}/{tag}"
        x = linear_tokens(
            x,
            fetch(params, spec_by_name, f"{pfx}/w", args, epoch, pair_idx, sign),
            fetch(params, spec_by_name, f"{pfx}/b", args, epoch, pair_idx, sign),
        )
        x = affine_channels(
            x,
            fetch(params, spec_by_name, f"{pfx}/gamma", args, epoch, pair_idx, sign),
            fetch(params, spec_by_name, f"{pfx}/beta", args, epoch, pair_idx, sign),
        )
        x = lif(x, cfg, args)
    return x


def forward_core(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    images: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch,
    pair_idx,
    sign: int = 1,
) -> jax.Array:
    x = sps_forward(params, spec_by_name, images, cfg, args, epoch, pair_idx, sign)
    for block_idx in range(cfg.depth):
        x = x + ssa_forward(params, spec_by_name, x, cfg, args, epoch, pair_idx, sign, block_idx)
        x = x + mlp_forward(params, spec_by_name, x, cfg, args, epoch, pair_idx, sign, block_idx)
    pooled = x.mean(axis=(0, 2))
    # Apply LayerNorm before head
    norm_gamma = fetch(params, spec_by_name, "head/norm/gamma", args, epoch, pair_idx, sign)
    norm_beta = fetch(params, spec_by_name, "head/norm/beta", args, epoch, pair_idx, sign)
    pooled = layer_norm(pooled, norm_gamma, norm_beta)
    if args.head_feature_centering == "batch" and pooled.shape[0] > 1:
        pooled = pooled - jnp.mean(pooled, axis=0, keepdims=True)
    logits = pooled @ fetch(params, spec_by_name, "head/w", args, epoch, pair_idx, sign)
    logits = logits + fetch(params, spec_by_name, "head/b", args, epoch, pair_idx, sign)
    return logits


def forward(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    images: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch=0,
    pair_idx=None,
    sign: int = 1,
) -> jax.Array:
    spec_by_name = {spec.name: spec for spec in specs}
    return forward_core(params, spec_by_name, images, cfg, args, epoch, pair_idx, sign)


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def loss_core(
    params: dict[str, jax.Array],
    spec_by_name: dict[str, ParamSpec],
    images: jax.Array,
    labels: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch,
    pair_idx,
    sign: int = 1,
) -> jax.Array:
    return cross_entropy(forward_core(params, spec_by_name, images, cfg, args, epoch, pair_idx, sign), labels)


def loss_for_member(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    images: jax.Array,
    labels: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch,
    pair_idx,
    sign: int = 1,
) -> jax.Array:
    spec_by_name = {spec.name: spec for spec in specs}
    return loss_core(params, spec_by_name, images, labels, cfg, args, epoch, pair_idx, sign)


def make_loss_functions(specs: list[ParamSpec], cfg: ModelConfig, args: Args):
    static_specs = tuple(specs)

    @jax.jit
    def clean_loss_fn(params, images, labels):
        return loss_for_member(params, static_specs, images, labels, cfg, args, 0, None, 1)

    @jax.jit
    def noisy_loss_fn(params, images, labels, epoch, pair_idx, sign):
        return loss_for_member(params, static_specs, images, labels, cfg, args, epoch, pair_idx, sign)

    return clean_loss_fn, noisy_loss_fn


def synthetic_batch(cfg: ModelConfig, seed: int, epoch: int, batch_size: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.fold_in(jax.random.PRNGKey(seed + 1009), epoch + 1_000_000)
    image_key, teacher_key = jax.random.split(key)
    images = jax.random.uniform(
        image_key,
        (batch_size, cfg.image_size, cfg.image_size, cfg.in_channels),
        minval=-1.0,
        maxval=1.0,
    )
    teacher = jax.random.normal(
        teacher_key,
        (cfg.image_size * cfg.image_size * cfg.in_channels, cfg.num_classes),
    ) / math.sqrt(float(cfg.image_size * cfg.image_size * cfg.in_channels))
    labels = jnp.argmax(images.reshape((batch_size, -1)) @ teacher, axis=-1).astype(jnp.int32)
    return images, labels


def normalize_cifar_images(images: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)
    images = images.astype(np.float32) / 255.0
    return (images - mean[None, None, None, :]) / std[None, None, None, :]


def decode_cifar_rows(dataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for idx in indices.tolist():
        row = dataset[int(idx)]
        images.append(np.asarray(row["img"].convert("RGB"), dtype=np.uint8))
        labels.append(int(row["label"]))
    return normalize_cifar_images(np.stack(images)), np.asarray(labels, dtype=np.int32)


def cifar_cache_path(args: Args) -> Path:
    return (
        Path(args.cifar_cache_dir)
        / f"cifar10_train{args.cifar_train_samples}_eval{args.cifar_eval_samples}_seed{args.seed}.npz"
    )


def load_cifar10_arrays(args: Args, cfg: ModelConfig) -> dict[str, np.ndarray]:
    if (cfg.image_size, cfg.in_channels, cfg.num_classes) != (32, 3, 10):
        raise ValueError("cifar10 data_source requires 32x32 RGB images and 10 classes")

    path = cifar_cache_path(args)
    if path.is_file():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    from datasets import load_dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    train_ds = load_dataset("uoft-cs/cifar10", split="train")
    eval_ds = load_dataset("uoft-cs/cifar10", split="test")
    rng = np.random.default_rng(args.seed)
    train_count = min(args.cifar_train_samples, len(train_ds))
    eval_count = min(args.cifar_eval_samples, len(eval_ds))
    train_indices = rng.permutation(len(train_ds))[:train_count]
    eval_indices = rng.permutation(len(eval_ds))[:eval_count]
    train_images, train_labels = decode_cifar_rows(train_ds, train_indices)
    eval_images, eval_labels = decode_cifar_rows(eval_ds, eval_indices)
    np.savez_compressed(
        path,
        train_images=train_images,
        train_labels=train_labels,
        eval_images=eval_images,
        eval_labels=eval_labels,
    )
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "eval_images": eval_images,
        "eval_labels": eval_labels,
    }


def array_batch(
    arrays: dict[str, np.ndarray],
    split: Literal["train", "eval"],
    seed: int,
    epoch: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    images = arrays[f"{split}_images"]
    labels = arrays[f"{split}_labels"]
    if batch_size > len(images):
        raise ValueError(f"batch_size={batch_size} exceeds {split} cache size={len(images)}")
    rng = np.random.default_rng(seed + 10_000 + epoch + (0 if split == "train" else 1_000_000))
    indices = rng.choice(len(images), size=batch_size, replace=False)
    return jnp.asarray(images[indices]), jnp.asarray(labels[indices])


def make_data_source(
    args: Args,
    cfg: ModelConfig,
) -> tuple[tuple[jax.Array, jax.Array], object | None]:
    if args.data_source == "synthetic":
        return synthetic_batch(cfg, args.seed, -1, args.batch_size), None
    arrays = load_cifar10_arrays(args, cfg)
    print(f"cifar_cache={cifar_cache_path(args)}")
    return array_batch(arrays, "eval", args.seed, -1, args.batch_size), arrays


def train_batch(
    args: Args,
    cfg: ModelConfig,
    arrays: object | None,
    epoch: int,
) -> tuple[jax.Array, jax.Array]:
    if args.data_source == "synthetic":
        return synthetic_batch(cfg, args.seed, epoch, args.batch_size)
    if not isinstance(arrays, dict):
        raise TypeError("cifar10 arrays were not initialized")
    return array_batch(arrays, "train", args.seed, epoch, args.batch_size)


def make_vmap_loss_fns(specs: list[ParamSpec], cfg: ModelConfig, args: Args):
    spec_by_name = {spec.name: spec for spec in specs}

    def plus_fn(params, images, labels, epoch, idx_arr):
        return jax.vmap(lambda idx: loss_core(params, spec_by_name, images, labels, cfg, args, epoch, idx, 1))(idx_arr)

    def minus_fn(params, images, labels, epoch, idx_arr):
        return jax.vmap(lambda idx: loss_core(params, spec_by_name, images, labels, cfg, args, epoch, idx, -1))(idx_arr)

    return jax.jit(plus_fn), jax.jit(minus_fn)


def population_losses_vmap(
    params: dict[str, jax.Array],
    images: jax.Array,
    labels: jax.Array,
    args: Args,
    epoch: int,
    plus_fn,
    minus_fn,
    chunk: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if args.population_size % 2 != 0:
        raise ValueError("population_size must be even for antithetic pairs")
    n_pairs = args.population_size // 2
    pair_indices = jnp.arange(n_pairs)
    epoch_arr = jnp.int32(epoch)
    if chunk <= 0 or chunk >= n_pairs:
        plus = plus_fn(params, images, labels, epoch_arr, pair_indices)
        minus = minus_fn(params, images, labels, epoch_arr, pair_indices)
    else:
        plus = jnp.concatenate([plus_fn(params, images, labels, epoch_arr, pair_indices[i:i + chunk])
                                for i in range(0, n_pairs, chunk)])
        minus = jnp.concatenate([minus_fn(params, images, labels, epoch_arr, pair_indices[i:i + chunk])
                                for i in range(0, n_pairs, chunk)])
    advantages = jnp.sign(minus - plus)
    return plus, minus, advantages


def make_vmap_update_fns(specs: list[ParamSpec], args: Args):
    fns = {}
    for spec in specs:
        def step(carry, adv_chunk, idx_chunk, epoch, _spec=spec):
            keys = jax.vmap(lambda idx: noise_key(args.seed, epoch, idx, _spec))(idx_chunk)
            noises = jax.vmap(lambda k: matrix_noise(_spec, k, args.noise_rank))(keys)
            bshape = (-1,) + (1,) * len(_spec.shape)
            per = adv_chunk.reshape(bshape) * noises
            new_carry, _ = jax.lax.scan(lambda c, x: (c + x, None), carry, per)
            return new_carry
        fns[spec.name] = jax.jit(step)
    return fns


def update_params_vmap(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    advantages: jax.Array,
    args: Args,
    epoch: int,
    update_fns: dict,
    chunk: int = 0,
) -> tuple[dict[str, jax.Array], float]:
    new_params = {}
    changed = []
    n_pairs = advantages.shape[0]
    if chunk <= 0:
        chunk = n_pairs
    pair_indices = jnp.arange(n_pairs)
    epoch_arr = jnp.int32(epoch)
    for spec in specs:
        z = jnp.zeros(spec.shape, dtype=jnp.float32)
        for cs in range(0, n_pairs, chunk):
            idx_chunk = pair_indices[cs:cs + chunk]
            adv_chunk = advantages[cs:cs + chunk]
            z = update_fns[spec.name](z, adv_chunk, idx_chunk, epoch_arr)
        abs_z = jnp.abs(z)
        if args.update_fraction < 1.0:
            threshold = jnp.percentile(abs_z.reshape(-1), 100.0 * (1.0 - args.update_fraction))
            mask = abs_z >= threshold
        else:
            mask = abs_z > 0
        delta = jnp.where(mask, jnp.sign(z).astype(jnp.int16) * args.param_step, 0)
        updated = jnp.clip(params[spec.name].astype(jnp.int16) + delta, PARAM_MIN, PARAM_MAX).astype(DTYPE)
        new_params[spec.name] = updated
        changed.append(jnp.mean(updated != params[spec.name]))
    changed_fraction = float(jnp.mean(jnp.asarray(changed)))
    return new_params, changed_fraction


def population_losses(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    images: jax.Array,
    labels: jax.Array,
    cfg: ModelConfig,
    args: Args,
    epoch: int,
    noisy_loss_fn=None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if args.population_size % 2 != 0:
        raise ValueError("population_size must be even for antithetic pairs")
    plus_losses = []
    minus_losses = []
    for pair_idx in range(args.population_size // 2):
        if noisy_loss_fn is None:
            plus_loss = loss_for_member(params, specs, images, labels, cfg, args, epoch, pair_idx, 1)
            minus_loss = loss_for_member(params, specs, images, labels, cfg, args, epoch, pair_idx, -1)
        else:
            plus_loss = noisy_loss_fn(params, images, labels, epoch, pair_idx, 1)
            minus_loss = noisy_loss_fn(params, images, labels, epoch, pair_idx, -1)
        plus_losses.append(plus_loss)
        minus_losses.append(minus_loss)
    plus = jnp.asarray(plus_losses)
    minus = jnp.asarray(minus_losses)
    advantages = jnp.sign(minus - plus)
    return plus, minus, advantages


def update_params(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    advantages: jax.Array,
    args: Args,
    epoch: int,
) -> tuple[dict[str, jax.Array], float]:
    new_params = {}
    changed = []
    for spec in specs:
        z = jnp.zeros(spec.shape, dtype=jnp.float32)
        for pair_idx in range(args.population_size // 2):
            adv = advantages[pair_idx]
            z = z + adv * matrix_noise(spec, noise_key(args.seed, epoch, pair_idx, spec), args.noise_rank)
        abs_z = jnp.abs(z)
        if args.update_fraction < 1.0:
            threshold = jnp.percentile(abs_z.reshape(-1), 100.0 * (1.0 - args.update_fraction))
            mask = abs_z >= threshold
        else:
            mask = abs_z > 0
        delta = jnp.where(mask, jnp.sign(z).astype(jnp.int16) * args.param_step, 0)
        updated = jnp.clip(params[spec.name].astype(jnp.int16) + delta, PARAM_MIN, PARAM_MAX).astype(DTYPE)
        new_params[spec.name] = updated
        changed.append(jnp.mean(updated != params[spec.name]))
    changed_fraction = float(jnp.mean(jnp.asarray(changed)))
    return new_params, changed_fraction


def spec_summary(specs: list[ParamSpec]) -> dict[str, int]:
    total = sum(int(np.prod(spec.shape)) for spec in specs)
    sps = sum(int(np.prod(spec.shape)) for spec in specs if spec.name.startswith("sps/"))
    blocks = sum(int(np.prod(spec.shape)) for spec in specs if spec.name.startswith("blocks/"))
    head = sum(int(np.prod(spec.shape)) for spec in specs if spec.name.startswith("head/"))
    return {
        "params_total": total,
        "params_sps": sps,
        "params_blocks": blocks,
        "params_head": head,
    }


def summarize_specs(specs: list[ParamSpec]) -> dict[str, int]:
    summary = spec_summary(specs)
    print(f"params_total={summary['params_total']:,}")
    print(f"params_sps={summary['params_sps']:,}")
    print(f"params_blocks={summary['params_blocks']:,}")
    print(f"params_head={summary['params_head']:,}")
    return summary


def encoded_param_key(index: int) -> str:
    return f"p{index:05d}"


def save_checkpoint(
    params: dict[str, jax.Array],
    specs: list[ParamSpec],
    output_dir: Path,
    epoch: int,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / ("initial.npz" if epoch < 0 else f"epoch_{epoch:04d}.npz")
    arrays = {
        encoded_param_key(i): np.asarray(params[spec.name])
        for i, spec in enumerate(specs)
    }
    arrays["param_names"] = np.asarray([spec.name for spec in specs])
    arrays["param_kinds"] = np.asarray([spec.kind for spec in specs])
    arrays["param_shapes"] = np.asarray([json.dumps(spec.shape) for spec in specs])
    np.savez_compressed(path, **arrays)
    return path


def setup_output_dir(args: Args, cfg: ModelConfig, specs: list[ParamSpec], summary: dict[str, int]) -> Path | None:
    if not args.output_dir:
        return None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    config = {
        "args": asdict(args),
        "model_config": asdict(cfg),
        "param_summary": summary,
        "specs": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "shape": list(spec.shape),
                "seed": spec.seed,
            }
            for spec in specs
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"output_dir={output_dir}")
    return output_dir


def append_metrics(output_dir: Path | None, metrics: dict[str, float | int | str]) -> None:
    if output_dir is None:
        return
    path = output_dir / "metrics.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")


def main() -> None:
    args = tyro.cli(Args)
    cfg = preset_config(args.preset)
    specs = make_specs(cfg)
    print(
        "config "
        f"preset={args.preset} dim={cfg.dim} depth={cfg.depth} "
        f"heads={cfg.num_heads} T={cfg.time_steps} population={args.population_size} "
        f"data={args.data_source} lif_mode={args.lif_mode}"
    )
    summary = summarize_specs(specs)
    if args.profile_only:
        return
    output_dir = setup_output_dir(args, cfg, specs, summary)

    params = init_params(specs, args.seed)
    (clean_images, clean_labels), arrays = make_data_source(args, cfg)
    clean_loss_fn, noisy_loss_fn = (make_loss_functions(specs, cfg, args) if args.use_jit else (None, None))
    vmap_plus_fn = vmap_minus_fn = vmap_update_fns = None
    if args.eval_mode == "vmap":
        vmap_plus_fn, vmap_minus_fn = make_vmap_loss_fns(specs, cfg, args)
        vmap_update_fns = make_vmap_update_fns(specs, args)
    if clean_loss_fn is None:
        clean_loss = float(loss_for_member(params, specs, clean_images, clean_labels, cfg, args, 0, None, 1))
    else:
        clean_loss = float(clean_loss_fn(params, clean_images, clean_labels))
    print(f"initial_clean_loss={clean_loss:.6f}")
    append_metrics(
        output_dir,
        {
            "epoch": -1,
            "phase": "initial",
            "clean_loss": clean_loss,
            "population_size": args.population_size,
            "batch_size": args.batch_size,
        },
    )
    if output_dir is not None and args.save_initial:
        checkpoint_path = save_checkpoint(params, specs, output_dir, -1)
        print(f"checkpoint={checkpoint_path}")

    for epoch in range(args.epochs):
        start = time.time()
        images, labels = train_batch(args, cfg, arrays, epoch)
        if args.eval_mode == "vmap":
            plus, minus, advantages = population_losses_vmap(
                params, images, labels, args, epoch, vmap_plus_fn, vmap_minus_fn, args.vmap_chunk
            )
            params, changed_fraction = update_params_vmap(
                params, specs, advantages, args, epoch, vmap_update_fns, args.vmap_chunk
            )
        else:
            plus, minus, advantages = population_losses(params, specs, images, labels, cfg, args, epoch, noisy_loss_fn)
            params, changed_fraction = update_params(params, specs, advantages, args, epoch)
        losses = jnp.concatenate([plus, minus])
        if clean_loss_fn is None:
            clean_loss = float(loss_for_member(params, specs, clean_images, clean_labels, cfg, args, epoch, None, 1))
        else:
            clean_loss = float(clean_loss_fn(params, clean_images, clean_labels))
        seconds = time.time() - start
        metrics = {
            "epoch": epoch,
            "phase": "train",
            "loss_mean": float(jnp.mean(losses)),
            "loss_min": float(jnp.min(losses)),
            "loss_max": float(jnp.max(losses)),
            "plus_loss_mean": float(jnp.mean(plus)),
            "minus_loss_mean": float(jnp.mean(minus)),
            "adv_nonzero": float(jnp.mean(advantages != 0)),
            "changed_fraction": changed_fraction,
            "clean_loss": clean_loss,
            "seconds": seconds,
            "population_size": args.population_size,
            "batch_size": args.batch_size,
        }
        append_metrics(output_dir, metrics)
        checkpoint_path = None
        if output_dir is not None and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = save_checkpoint(params, specs, output_dir, epoch)
        print(
            f"epoch={epoch} "
            f"loss_mean={metrics['loss_mean']:.6f} "
            f"loss_min={metrics['loss_min']:.6f} "
            f"loss_max={metrics['loss_max']:.6f} "
            f"adv_nonzero={metrics['adv_nonzero']:.3f} "
            f"changed={changed_fraction:.6f} "
            f"clean_loss={clean_loss:.6f} "
            f"seconds={seconds:.2f}"
        )
        if checkpoint_path is not None:
            print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
