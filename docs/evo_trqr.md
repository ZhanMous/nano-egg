# Evo-TRQR frozen-backbone mechanism search

This implementation covers the discovery stage only:

1. keep a trained SNN backbone frozen;
2. insert exact identity-gated temporal redundancy controllers after candidate
   transformer blocks;
3. evolve a 16-dimensional continuous controller with antithetic ES;
4. mutate the small discrete mechanism genome outside the continuous update;
5. rank candidates with an epsilon-constrained energy budget.

It does **not** update backbone weights with EGGROLL yet. That belongs after a
controller survives discovery and inherited-adaptation validation.

## Files

- `evo_trqr.py`: controller, MI estimator, genome decoder, constrained ranking,
  antithetic ask/tell, and discrete mutation.
- `experiments/evo_trqr_frozen_search.py`: frozen-backbone search runner.
- `tests/test_evo_trqr.py`: neutrality, MI, common-random-number, constraint,
  and antithetic tests.
- `tests/test_spikformer_event_input.py`: event-frame input contract.

## Genome

The continuous genome has 16 values:

```text
0:4    layer interpolation-strength logits
4:8    layer mask-budget logits
8:11   local/global/interaction MI fusion weights
11     mask-mapping threshold
12     mask-mapping temperature
13     mask-mapping exponent
14     schedule warmup
15     schedule end
```

The discrete genome contains:

```text
reference     = t0 | previous | multi_lag | ema
lag_set       = subset of {1, 2, 4}
spatial_scale = token | channel | block
recalibration = zero | redistribute | renorm
```

With a final-policy-only `--progress-grid 1.0`, the two schedule dimensions are
frozen because they are not identifiable. They become active only when the
runner evaluates multiple schedule positions. Layer gate/budget dimensions
beyond the selected backbone depth are also frozen.

## Strict inherited neutrality

Each stage uses:

```text
y = x + master_strength * (TRQR(x) - x)
```

The runner compares clean logits with controller logits at
`master_strength=0` and aborts unless the maximum absolute error is exactly
zero.

## Local smoke

This checks the code path only. Static images are repeated across time, and
the randomly initialized backbone is not scientific evidence for temporal MI:

```bash
JAX_PLATFORMS=cpu conda run -n nanoegg python \
  experiments/evo_trqr_frozen_search.py \
  --preset tiny \
  --data-source synthetic \
  --batch-size 2 \
  --eval-batch-size 2 \
  --generations 1 \
  --pairs 1 \
  --mask-seeds 0 \
  --structure-interval 0 \
  --output-dir /tmp/evo_trqr_frozen_smoke
```

## CIFAR10-DVS event-cache contract

Real mechanism discovery uses the event preset and a trained, frozen nano-egg
checkpoint:

```bash
conda run -n nanoegg python experiments/evo_trqr_frozen_search.py \
  --preset spikformer_dvs_2_256 \
  --data-source event_npz \
  --event-cache cached_files/cifar10dvs_frames_t10.npz \
  --backbone-state runs/cifar10dvs_backbone/checkpoints/best.npz \
  --batch-size 64 \
  --eval-batch-size 128 \
  --generations 50 \
  --pairs 32 \
  --energy-budget 0.90 \
  --mask-seeds 0,1,2,3 \
  --output-dir runs/evo_trqr/cifar10dvs_spikformer_rho090_seed0
```

The event cache must contain:

```text
train_frames [B_train, 10, 128, 128, 2]
train_labels [B_train]
eval_frames  [B_eval, 10, 128, 128, 2]
eval_labels  [B_eval]
```

Frames are batch-major and must use the same preprocessing as the backbone
checkpoint. The checkpoint uses the named nano-egg format written by
`experiments/spikformer_es_smoke.py`: `param_names` plus `p00000`, `p00001`,
and so on.

The runner refuses event-data discovery without `--backbone-state`; a random
event SNN would make the mechanism ranking uninterpretable.

## Outputs and interpretation

The output directory contains:

- `run_config.json`: protocol, model shape, clean baseline, and neutrality
  result;
- `metrics.jsonl`: center genome, discrete mechanism, constraint status and
  diagnostics for each generation;
- `best_genome.json`: best search candidate and its held-out batch metrics.

The runner refuses to replace existing search artifacts unless `--overwrite`
is passed.

`energy_ratio` is currently the retained pre-recalibration stage-activity
ratio. This lets zeroing, redistribution, and renormalization compete under the
same deletion budget, but it remains a search proxy rather than measured
hardware energy or latency. A later confirmatory stage must compile hard gates
that actually skip work and record device-level latency/energy.

Run the dependency-free test suite with:

```bash
JAX_PLATFORMS=cpu conda run -n nanoegg python -m unittest discover -s tests -v
```
