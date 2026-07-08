"""Sanity check: verify random init gives reasonable logits.
Run this BEFORE any ES training."""
import sys, os, math
sys.path.insert(0, os.getcwd())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax
import jax.numpy as jnp
import numpy as np
from experiments.spikformer_es_smoke import (
    Args, preset_config, make_specs, init_params,
    load_cifar10_arrays, forward, cross_entropy,
)

def sanity_check(preset_name, num_samples=256):
    cfg = preset_config(preset_name)
    specs = make_specs(cfg)
    params = init_params(specs, 0)
    total_params = sum(int(np.prod(s.shape)) for s in specs)
    
    args = Args(
        preset=preset_name, data_source="cifar10", seed=0,
        epochs=1, population_size=64, batch_size=num_samples,
        sigma=0.05, noise_rank=1, update_fraction=0.01,
        param_step=1, cifar_train_samples=50000, cifar_eval_samples=10000,
        use_jit=True, eval_mode="loop", vmap_chunk=64,
        profile_only=False, output_dir="/tmp/sanity",
        save_every=1, save_initial=True,
        cifar_cache_dir="cached_files/cifar10_jax",
    )
    
    data = load_cifar10_arrays(args, cfg)
    X = jnp.asarray(data["eval_images"][:num_samples])
    Y = jnp.asarray(data["eval_labels"][:num_samples].astype(int))
    
    logits = forward(params, specs, X, cfg, args)
    logits_np = np.array(logits)
    
    probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    ce = float(cross_entropy(logits, Y))
    pred = logits_np.argmax(axis=-1)
    hist = np.bincount(pred, minlength=10)
    nonzero_classes = int((hist > 0).sum())
    
    print(f"\n{'='*50}")
    print(f"Sanity Check: {preset_name}")
    print(f"  Params: {total_params:,}")
    print(f"{'='*50}")
    print(f"Logits mean:   {logits_np.mean():>8.4f}  (target: ~0)")
    print(f"Logits std:    {logits_np.std():>8.4f}  (target: 0.1-2.0)")
    print(f"Logits min:    {logits_np.min():>8.4f}  (target: >-10)")
    print(f"Logits max:    {logits_np.max():>8.4f}  (target: <10)")
    print(f"CE loss:       {ce:>8.4f}  (target: ~{math.log(10):.2f})")
    print(f"Max prob mean: {probs.max(axis=-1).mean():>8.4f}  (target: 0.1-0.3)")
    print(f"Accuracy:      {(pred == np.array(Y)).mean()*100:>7.2f}%  (target: ~10%)")
    print(f"Pred hist:     {hist}")
    print(f"  (should be roughly uniform across 10 classes)")
    
    # Verdict
    ok = True
    checks = []
    if logits_np.std() < 0.1 or logits_np.std() > 2.0:
        checks.append(f"FAIL: logits std={logits_np.std():.2f} outside [0.1, 2.0]")
        ok = False
    if abs(logits_np.mean()) > 5.0:
        checks.append(f"FAIL: logits mean={logits_np.mean():.2f}")
        ok = False
    if logits_np.max() > 10 or logits_np.min() < -10:
        checks.append(f"FAIL: logits range [{logits_np.min():.1f}, {logits_np.max():.1f}]")
        ok = False
    if abs(ce - math.log(10)) > 0.35:
        checks.append(f"FAIL: CE={ce:.2f} not close to log(10)={math.log(10):.2f}")
        ok = False
    max_prob_mean = probs.max(axis=-1).mean()
    if max_prob_mean < 0.1 or max_prob_mean > 0.3:
        checks.append(f"FAIL: max prob={max_prob_mean:.3f} outside [0.1, 0.3]")
        ok = False
    if nonzero_classes < 8:
        checks.append(f"FAIL: pred hist covers only {nonzero_classes}/10 classes")
        ok = False
    if hist.max() > int(0.35 * num_samples):
        checks.append(f"FAIL: pred hist collapsed; max bucket={hist.max()}/{num_samples}")
        ok = False
    
    print(f"\n{'='*50}")
    if ok:
        print("  ✅ PASSED - Init is in normal range")
    else:
        print("  ❌ FAILED - Init needs more fixes:")
        for c in checks:
            print(f"    {c}")
    print(f"{'='*50}")
    return ok

# Run for all presets
for preset in ["tiny", "smoke", "spikformer_2_128", "spikformer_4_256"]:
    try:
        sanity_check(preset)
    except Exception as e:
        print(f"\n{preset}: ERROR - {e}")
