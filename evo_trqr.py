"""Core primitives for identity-gated evolutionary temporal redundancy control.

The module is deliberately independent of a particular SNN backbone.  It
expects stage activations in ``[time, batch, token, channel]`` order and keeps
the low-dimensional controller separate from any later EGGROLL weight update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np


MAX_LAYERS = 4
CONTINUOUS_DIM = 16

ReferenceKind = Literal["t0", "previous", "multi_lag", "ema"]
SpatialKind = Literal["token", "channel", "block"]
RecalibrationKind = Literal["zero", "redistribute", "renorm"]


@dataclass(frozen=True)
class DiscreteGenome:
    reference: ReferenceKind = "previous"
    lag_set: tuple[int, ...] = (1, 2)
    spatial_scale: SpatialKind = "token"
    recalibration: RecalibrationKind = "zero"

    def __post_init__(self) -> None:
        if self.reference not in ("t0", "previous", "multi_lag", "ema"):
            raise ValueError(f"unsupported reference: {self.reference}")
        if self.spatial_scale not in ("token", "channel", "block"):
            raise ValueError(f"unsupported spatial_scale: {self.spatial_scale}")
        if self.recalibration not in ("zero", "redistribute", "renorm"):
            raise ValueError(f"unsupported recalibration: {self.recalibration}")
        if not self.lag_set or any(lag <= 0 for lag in self.lag_set):
            raise ValueError("lag_set must contain positive integers")


@dataclass(frozen=True)
class DecodedContinuousGenome:
    layer_gates: jax.Array
    layer_budgets: jax.Array
    fusion_weights: jax.Array
    threshold: jax.Array
    temperature: jax.Array
    mapping_exponent: jax.Array
    warmup: jax.Array
    schedule_end: jax.Array


@dataclass(frozen=True)
class CandidateMetrics:
    accuracy: float
    loss: float
    energy_ratio: float

    def is_finite(self) -> bool:
        return bool(np.isfinite([self.accuracy, self.loss, self.energy_ratio]).all())


@dataclass(frozen=True)
class ControllerDiagnostics:
    local_mi_mean: jax.Array
    global_mi_mean: jax.Array
    mask_probability_mean: jax.Array
    mask_fraction: jax.Array
    removed_activity_fraction: jax.Array
    input_activity: jax.Array
    output_activity: jax.Array


def initial_continuous_genome() -> np.ndarray:
    """Return a conservative controller center with 20--30% layer budgets."""
    genome = np.zeros((CONTINUOUS_DIM,), dtype=np.float32)
    genome[0:4] = -1.5  # layer gate ~= 0.18
    genome[4:8] = -1.0  # layer budget ~= 0.27 * budget ceiling
    genome[8:11] = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    genome[11] = 0.0  # MI threshold 0.5
    genome[12] = -1.5  # mapping temperature ~= 0.25
    genome[13] = 0.5  # mapping exponent ~= 1.2
    genome[14] = -2.0  # short warmup
    genome[15] = 1.5  # schedule finishes late
    return genome


def decode_continuous(
    genome: jax.Array | np.ndarray,
    *,
    budget_ceiling: float = 0.8,
) -> DecodedContinuousGenome:
    raw = jnp.asarray(genome, dtype=jnp.float32)
    if raw.shape != (CONTINUOUS_DIM,):
        raise ValueError(f"continuous genome must have shape {(CONTINUOUS_DIM,)}, got {raw.shape}")
    if not 0.0 < budget_ceiling <= 1.0:
        raise ValueError("budget_ceiling must be in (0, 1]")
    gates = jax.nn.sigmoid(raw[0:4])
    budgets = budget_ceiling * jax.nn.sigmoid(raw[4:8])
    fusion = jnp.tanh(raw[8:11])
    threshold = jax.nn.sigmoid(raw[11])
    temperature = jax.nn.softplus(raw[12]) + 0.05
    mapping_exponent = jax.nn.softplus(raw[13]) + 0.25
    warmup = 0.75 * jax.nn.sigmoid(raw[14])
    end_fraction = jax.nn.sigmoid(raw[15])
    schedule_end = warmup + (1.0 - warmup) * end_fraction
    return DecodedContinuousGenome(
        layer_gates=gates,
        layer_budgets=budgets,
        fusion_weights=fusion,
        threshold=threshold,
        temperature=temperature,
        mapping_exponent=mapping_exponent,
        warmup=warmup,
        schedule_end=schedule_end,
    )


def schedule_value(progress: float | jax.Array, decoded: DecodedContinuousGenome) -> jax.Array:
    progress_arr = jnp.clip(jnp.asarray(progress, dtype=jnp.float32), 0.0, 1.0)
    width = jnp.maximum(decoded.schedule_end - decoded.warmup, 1e-6)
    unit = jnp.clip((progress_arr - decoded.warmup) / width, 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def _lagged_reference(binary_x: jax.Array, lag: int) -> jax.Array:
    indices = jnp.maximum(jnp.arange(binary_x.shape[0]) - int(lag), 0)
    return binary_x[indices]


def build_reference(binary_x: jax.Array, genome: DiscreteGenome) -> jax.Array:
    if genome.reference == "t0":
        return jnp.broadcast_to(binary_x[:1], binary_x.shape)
    if genome.reference == "previous":
        return _lagged_reference(binary_x, 1)
    if genome.reference == "multi_lag":
        references = jnp.stack([_lagged_reference(binary_x, lag) for lag in genome.lag_set])
        return jnp.mean(references, axis=0)

    # EMA prototype.  The current step is compared with the prototype built
    # only from preceding steps, so it cannot reference itself.
    refs = [binary_x[0]]
    prototype = binary_x[0]
    for time_idx in range(1, binary_x.shape[0]):
        refs.append(prototype)
        prototype = 0.75 * prototype + 0.25 * binary_x[time_idx]
    return jnp.stack(refs, axis=0)


def binary_mutual_information(
    binary_x: jax.Array,
    binary_reference: jax.Array,
    *,
    axis: int | tuple[int, ...],
) -> jax.Array:
    """Estimate normalized binary MI over one or more feature axes."""
    x = jnp.clip(jnp.asarray(binary_x, dtype=jnp.float32), 0.0, 1.0)
    y = jnp.clip(jnp.asarray(binary_reference, dtype=jnp.float32), 0.0, 1.0)
    if x.shape != y.shape:
        raise ValueError(f"MI operands must have the same shape, got {x.shape} and {y.shape}")
    p11 = jnp.mean(x * y, axis=axis, keepdims=True)
    p10 = jnp.mean(x * (1.0 - y), axis=axis, keepdims=True)
    p01 = jnp.mean((1.0 - x) * y, axis=axis, keepdims=True)
    p00 = jnp.mean((1.0 - x) * (1.0 - y), axis=axis, keepdims=True)
    joint = jnp.stack((p00, p01, p10, p11), axis=0)
    px0 = p00 + p01
    px1 = p10 + p11
    py0 = p00 + p10
    py1 = p01 + p11
    independent = jnp.stack((px0 * py0, px0 * py1, px1 * py0, px1 * py1), axis=0)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)
    terms = jnp.where(
        joint > eps,
        joint * jnp.log((joint + eps) / (independent + eps)),
        0.0,
    )
    return jnp.clip(jnp.sum(terms, axis=0) / jnp.log(2.0), 0.0, 1.0)


def _mi_scores(binary_x: jax.Array, reference: jax.Array, spatial_scale: SpatialKind) -> tuple[jax.Array, jax.Array]:
    if spatial_scale == "token":
        local = binary_mutual_information(binary_x, reference, axis=3)
    elif spatial_scale == "channel":
        local = binary_mutual_information(binary_x, reference, axis=2)
    else:
        local = binary_mutual_information(binary_x, reference, axis=(2, 3))
    global_score = binary_mutual_information(binary_x, reference, axis=(2, 3))
    return local, global_score


def _recalibrate(x: jax.Array, mask: jax.Array, mode: RecalibrationKind, spatial_scale: SpatialKind) -> jax.Array:
    kept = 1.0 - mask
    zeroed = x * kept
    if mode == "zero" or spatial_scale == "block":
        return zeroed
    reduce_axis = 2 if spatial_scale == "token" else 3 if spatial_scale == "channel" else (2, 3)
    total_count = float(x.shape[2] if spatial_scale == "token" else x.shape[3])
    kept_count = jnp.sum(kept, axis=reduce_axis, keepdims=True)
    if mode == "renorm":
        scale = jnp.where(kept_count > 0, total_count / kept_count, 0.0)
        return zeroed * scale
    removed_sum = jnp.sum(x * mask, axis=reduce_axis, keepdims=True)
    transfer = jnp.where(kept_count > 0, removed_sum / jnp.maximum(kept_count, 1.0), 0.0)
    return zeroed + kept * transfer


def apply_evo_trqr(
    x: jax.Array,
    continuous_genome: jax.Array | np.ndarray,
    discrete_genome: DiscreteGenome,
    *,
    layer_index: int,
    progress: float,
    random_key: jax.Array,
    master_strength: float = 1.0,
    budget_ceiling: float = 0.8,
    uniforms: jax.Array | None = None,
) -> tuple[jax.Array, ControllerDiagnostics]:
    """Apply one controller stage while preserving exact identity at strength 0."""
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    if x_arr.ndim != 4:
        raise ValueError(f"expected [T, B, N, C] activations, got {x_arr.shape}")
    if not 0 <= layer_index < MAX_LAYERS:
        raise ValueError(f"layer_index must be in [0, {MAX_LAYERS})")
    decoded = decode_continuous(continuous_genome, budget_ceiling=budget_ceiling)
    binary_x = (x_arr > 0).astype(jnp.float32)
    reference = build_reference(binary_x, discrete_genome)
    local_mi, global_mi = _mi_scores(binary_x, reference, discrete_genome.spatial_scale)
    w_local, w_global, w_interaction = decoded.fusion_weights
    redundancy = (
        w_local * local_mi
        + w_global * global_mi
        + w_interaction * local_mi * global_mi
    )
    mapped = jax.nn.sigmoid((redundancy - decoded.threshold) / decoded.temperature)
    mapped = jnp.power(jnp.clip(mapped, 1e-6, 1.0), decoded.mapping_exponent)
    probability = (
        decoded.layer_budgets[layer_index]
        * schedule_value(progress, decoded)
        * mapped
    )
    probability = jnp.clip(probability, 0.0, 1.0)
    if uniforms is None:
        uniforms = jax.random.uniform(random_key, probability.shape)
    elif uniforms.shape != probability.shape:
        raise ValueError(f"uniforms must have shape {probability.shape}, got {uniforms.shape}")
    mask = (uniforms < probability).astype(jnp.float32)
    transformed = _recalibrate(x_arr, mask, discrete_genome.recalibration, discrete_genome.spatial_scale)

    # This form is intentionally used instead of scaling the mask probability:
    # master_strength == 0 is an algebraic identity regardless of mask/RNG state.
    strength = jnp.asarray(master_strength, dtype=jnp.float32) * decoded.layer_gates[layer_index]
    output = x_arr + strength * (transformed - x_arr)
    input_activity_sum = jnp.sum(jnp.abs(x_arr))
    masked_activity_sum = jnp.sum(jnp.abs(x_arr) * mask)
    removed_activity_fraction = jnp.where(
        input_activity_sum > 0,
        strength * masked_activity_sum / input_activity_sum,
        0.0,
    )
    diagnostics = ControllerDiagnostics(
        local_mi_mean=jnp.mean(local_mi),
        global_mi_mean=jnp.mean(global_mi),
        mask_probability_mean=jnp.mean(probability),
        mask_fraction=jnp.mean(mask),
        removed_activity_fraction=removed_activity_fraction,
        input_activity=jnp.mean(jnp.abs(x_arr)),
        output_activity=jnp.mean(jnp.abs(output)),
    )
    return output, diagnostics


def candidate_order_key(metrics: CandidateMetrics, energy_budget: float) -> tuple[float, ...]:
    """Lexicographic epsilon-constraint ordering; larger tuples are better."""
    if not metrics.is_finite():
        return (-2.0, -np.inf, -np.inf, -np.inf)
    violation = max(metrics.energy_ratio - energy_budget, 0.0)
    if violation == 0.0:
        return (1.0, metrics.accuracy, -metrics.loss, -metrics.energy_ratio)
    return (0.0, -violation, metrics.accuracy, -metrics.loss)


def better_candidate(candidate: CandidateMetrics, incumbent: CandidateMetrics, energy_budget: float) -> bool:
    return candidate_order_key(candidate, energy_budget) > candidate_order_key(incumbent, energy_budget)


def centered_constraint_ranks(
    metrics: Sequence[CandidateMetrics],
    energy_budget: float,
) -> np.ndarray:
    order = sorted(range(len(metrics)), key=lambda idx: candidate_order_key(metrics[idx], energy_budget))
    ranks = np.empty((len(metrics),), dtype=np.float32)
    ranks[np.asarray(order)] = np.arange(len(metrics), dtype=np.float32)
    if len(metrics) <= 1:
        return np.zeros_like(ranks)
    ranks = ranks / float(len(metrics) - 1) - 0.5
    return ranks / (ranks.std() + 1e-8)


def ask_antithetic(
    center: np.ndarray,
    *,
    pairs: int,
    sigma: float,
    rng: np.random.Generator,
    active_dimensions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairs <= 0 or sigma <= 0:
        raise ValueError("pairs and sigma must be positive")
    center_arr = np.asarray(center, dtype=np.float32)
    noise = rng.standard_normal((pairs, center_arr.size), dtype=np.float32)
    if active_dimensions is not None:
        active = np.asarray(active_dimensions, dtype=np.float32)
        if active.shape != center_arr.shape:
            raise ValueError("active_dimensions must match center shape")
        noise *= active[None, :]
    return center_arr[None, :] + sigma * noise, center_arr[None, :] - sigma * noise, noise


def tell_antithetic(
    center: np.ndarray,
    noise: np.ndarray,
    plus_metrics: Sequence[CandidateMetrics],
    minus_metrics: Sequence[CandidateMetrics],
    *,
    energy_budget: float,
    sigma: float,
    learning_rate: float,
) -> np.ndarray:
    if len(plus_metrics) != len(minus_metrics) or len(plus_metrics) != noise.shape[0]:
        raise ValueError("metrics and noise must contain the same number of pairs")
    all_metrics = list(plus_metrics) + list(minus_metrics)
    ranks = centered_constraint_ranks(all_metrics, energy_budget)
    pair_signal = ranks[: noise.shape[0]] - ranks[noise.shape[0] :]
    gradient = np.mean(pair_signal[:, None] * noise, axis=0) / max(sigma, 1e-8)
    return np.asarray(center, dtype=np.float32) + learning_rate * gradient.astype(np.float32)


def mutate_discrete(genome: DiscreteGenome, rng: np.random.Generator) -> DiscreteGenome:
    field = int(rng.integers(0, 4))
    values = asdict(genome)
    if field == 0:
        choices = [value for value in ("t0", "previous", "multi_lag", "ema") if value != genome.reference]
        values["reference"] = choices[int(rng.integers(len(choices)))]
    elif field == 1:
        lag_options = [(1,), (2,), (4,), (1, 2), (1, 2, 4)]
        choices = [value for value in lag_options if value != genome.lag_set]
        values["lag_set"] = choices[int(rng.integers(len(choices)))]
    elif field == 2:
        choices = [value for value in ("token", "channel", "block") if value != genome.spatial_scale]
        values["spatial_scale"] = choices[int(rng.integers(len(choices)))]
    else:
        choices = [value for value in ("zero", "redistribute", "renorm") if value != genome.recalibration]
        values["recalibration"] = choices[int(rng.integers(len(choices)))]
    return DiscreteGenome(**values)
