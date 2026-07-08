# Spikformer ES Smoke

This repo now has a standalone JAX smoke path for trying EGGROLL-style
evolution on a Spikformer-like SNN without changing the original minGRU
`run.py` training entrypoint.

Reference implementation:

- Official Spikformer repo cloned at `/home/zhan_shaoji/code/spikformer_reference`
- Official CIFAR model definition: `/home/zhan_shaoji/code/spikformer_reference/cifar10/model.py`
- Target architecture: `Spikformer-4-256`, `T=4`, `num_heads=8`, CIFAR-style `32x32` input

The target preset matches the published CIFAR parameter scale:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset spikformer_4_256 --profile-only
```

Expected count:

```text
params_total=4,159,274
params_sps=979,232
params_blocks=3,177,472
params_head=2,570
```

Minimal smoke training:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset smoke --epochs 2 --population-size 4 --batch-size 4
```

Minimal CIFAR-10 smoke training:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset smoke --data-source cifar10 --cifar-train-samples 16 \
  --cifar-eval-samples 8 --epochs 1 --population-size 4 --batch-size 4
```

Minimal CIFAR-10 smoke training with metrics and checkpoints:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset smoke --data-source cifar10 --cifar-train-samples 32 \
  --cifar-eval-samples 16 --epochs 3 --population-size 4 --batch-size 4 \
  --output-dir runs/spikformer_es_smoke/cifar_smoke_named_params
```

Minimal target-shape update check:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset spikformer_4_256 --epochs 1 --population-size 2 \
  --batch-size 1 --update-fraction 0.01
```

Minimal target-shape CIFAR-10 update check:

```bash
conda run -n nanoegg python experiments/spikformer_es_smoke.py \
  --preset spikformer_4_256 --data-source cifar10 \
  --cifar-train-samples 4 --cifar-eval-samples 4 \
  --epochs 1 --population-size 2 --batch-size 1 --update-fraction 0.01
```

Current status:

- Synthetic-data smoke path runs end to end.
- CIFAR-10 subset caching runs through Hugging Face `uoft-cs/cifar10`; `pillow` is required for image decoding.
- The `spikformer_4_256` preset completes a clean forward pass.
- The `spikformer_4_256` preset completes one minimal antithetic ES update.
- The `spikformer_4_256` preset completes one minimal antithetic ES update on real CIFAR-10 images.
- Initialization calibration is now required before any ES run is interpretable:
  head LayerNorm, batch feature centering, small quantized head init, and mean
  pooling keep random logits near CE `log(10)` instead of saturated predictions.
- Full-model calibrated ES at `pop=1,048,576`, `update_fraction=0.05`,
  `batch_size=16`, `sigma=0.05` on `10.28.2.47` reached a peak
  `test_acc=18.61%` and final `18.47%` after 20 epochs:
  `runs/spikformer4_1m_calibrated_top5/results.json`.
- Head-only centered-rank ES at `pop=1,048,576`, `batch_size=256`,
  `sigma=0.05` on the same random `spikformer_4_256` trunk is stronger:
  it peaked at epoch 15 with `test_acc=28.16%`, `test_loss=1.9947`
  and finished at `27.95%` in `runs/spikformer4_head_rank_1m/results.json`.
- A scoped full-model smoke with `INIT_HEAD_STATE` and
  `UPDATE_SCOPE=block3,head_linear` reproduced the trained-head initial eval
  (`24.50%` at the epoch-4 head state), proving that head-only state can be
  handed to the full runner. Its update step was too slow in the current
  full-forward implementation, so do not launch a 1M scoped run until that path
  is optimized or narrowed.
- A frozen-trunk feature-adapter runner now trains a residual bottleneck adapter
  above the Spikformer pooled features. With the epoch-8 head fixed, bottleneck
  `64`, `pop=1,048,576`, and `update_fraction=0.05`, adapter-only ES reached
  `test_acc=28.44%`, `test_loss=1.9732` after 20 epochs:
  `runs/spikformer4_adapter_only_1m_b64_top5/results.json`. Head parameters
  were unchanged (`changed.head_w=0`, `changed.head_b=0`), so this is evidence
  for ES training non-head structural parameters. Bottleneck `16` reached
  `27.71%`; bottleneck `128` reached `27.78%`, so `64` is the current best
  adapter width.
- With the stronger epoch-15 head fixed, `bottleneck=64`, `batch_size=1024`,
  and `pop=1,048,576`, adapter-only ES reached `30.20%`:
  `runs/spikformer4_adapter_only_1m_b64_e15_b1024_top5/results.json`.
  Continuing from that best state with smaller sparse updates improved the
  result on `10.28.2.47`:
  - stage2, `sigma=0.05`, `update_fraction=0.01`: best `31.23%`,
    `test_loss=1.9059`, `runs/spikformer4_adapter_b64_b1024_stage2_top1/results.json`;
  - stage3, `sigma=0.05`, `update_fraction=0.005`: best `31.64%`,
    `test_loss=1.8955`, `runs/spikformer4_adapter_b64_b1024_stage3_top05/results.json`;
  - stage3, `sigma=0.025`, `update_fraction=0.01`: best `31.79%`,
    `test_loss=1.8912`, `runs/spikformer4_adapter_b64_b1024_stage3_sigma025_top1/results.json`;
  - stage4, `sigma=0.0125`, `update_fraction=0.01`: best `32.18%`,
    `test_loss=1.8853`, `runs/spikformer4_adapter_b64_b1024_stage4_sigma0125_top1/results.json`;
  - stage4, `sigma=0.025`, `update_fraction=0.005`: current best
    `32.36%`, `test_loss=1.8862`,
    `runs/spikformer4_adapter_b64_b1024_stage4_sigma025_top05/results.json`.
  - stage5, continued from the stage4 best state on shared GPUs with
    `chunk=128` and JAX preallocation disabled, `sigma=0.025`,
    `update_fraction=0.0025`: best `32.50%`, `test_loss=1.8802`,
    `runs/spikformer4_adapter_b64_b1024_stage5_sigma025_top025_c128/results.json`;
  - stage5, same continuation but keeping `update_fraction=0.005` for
    60 epochs: current best `32.60%`, `test_loss=1.8782`,
    `runs/spikformer4_adapter_b64_b1024_stage5_sigma025_top05_c128/results.json`.
  - stage6, continued from the stage5 best with `sigma=0.025`,
    `update_fraction=0.005`: no accuracy gain over the initial checkpoint
    (`32.60%`), but final loss dropped to `1.8753`:
    `runs/spikformer4_adapter_b64_b1024_stage6_sigma025_top05_c128/results.json`;
  - stage6, `sigma=0.0125`, `update_fraction=0.005`: no accuracy gain
    and worse final accuracy (`31.58%`):
    `runs/spikformer4_adapter_b64_b1024_stage6_sigma0125_top05_c128/results.json`.
  - stage7, continued from the stage5 best but enabling sparse head updates
    (`train_head=True`), `sigma=0.025`, `update_fraction=0.0025`: best
    `32.64%`, `test_loss=1.8752`,
    `runs/spikformer4_adapter_b64_b1024_stage7_trainhead_sigma025_top025_c128/results.json`;
  - stage7, `train_head=True`, `sigma=0.025`, `update_fraction=0.001`:
    best `32.69%`, `test_loss=1.8732`,
    `runs/spikformer4_adapter_b64_b1024_stage7_trainhead_sigma025_top01_c128/results.json`.
  - stage8, continued from the stage7 best with `train_head=True`,
    `sigma=0.025`, `update_fraction=0.001`: current best `32.83%`,
    `test_loss=1.8743`, with final loss `1.8727`:
    `runs/spikformer4_adapter_b64_b1024_stage8_trainhead_sigma025_top01_c128/results.json`;
  - stage8, `sigma=0.0125`, `update_fraction=0.001`: best `32.74%`,
    worse than keeping `sigma=0.025`:
    `runs/spikformer4_adapter_b64_b1024_stage8_trainhead_sigma0125_top01_c128/results.json`.
  The best frozen-head run proves non-head adapter gains beyond the head-only
  baseline; the later sparse head updates add a small extra improvement. Final
  epoch accuracy can trail the best checkpoint, so best-state selection matters.
- The current evidence says population scale and calibrated random features are
  sufficient for a learnable ES signal. Full-model sparse sign updates dilute
  the signal; adapter-only sparse updates are currently the best structured
  direction.
- `--use-jit` exists but is off by default for the original smoke script. The
  server runners use targeted JIT where it helped the large population path.
- Current decision: keep the named parameter dictionary for interpretability and architecture alignment; do not flatten parameters for now.
- `--output-dir` writes `run_config.json`, `metrics.jsonl`, and named-parameter checkpoints under `checkpoints/`.
  The calibrated target preset has 131 named int8 parameter arrays and was validated on `spikformer_4_256`.

Next engineering step:

- Preserve `runs/spikformer4_adapter_b64_b1024_stage8_trainhead_sigma025_top01_c128/best_state.npz`
  as the current best adapter/head state (`32.83%`).
- Further continuation with the same feature-adapter readout still gives small
  gains, but the curve is flattening. The next meaningful engineering step is
  to add a stronger non-head parameterization, such as late-block adapters or a
  deeper feature readout, then rerun the same 1M ES ladder.
- Avoid all-parameter full-sign updates near 25% accuracy; use sparse updates,
  fixed heads, or smaller steps before returning to full-model perturbations.
- Treat head-only results as a diagnostic baseline, not as the final paper-level
  claim: the real goal still requires training more than the classifier head and
  validating beyond one CIFAR-10 configuration.
