import unittest

import jax.numpy as jnp

from experiments.spikformer_es_smoke import (
    Args,
    init_params,
    make_specs,
    preset_config,
    sps_forward,
)


class SpikformerEventInputTests(unittest.TestCase):
    def test_sps_accepts_batch_major_event_frames(self) -> None:
        cfg = preset_config("smoke")
        specs = make_specs(cfg)
        params = init_params(specs, seed=0)
        spec_by_name = {spec.name: spec for spec in specs}
        args = Args(preset="smoke", batch_size=1, lif_mode="hard_spike")
        frames = jnp.zeros(
            (1, cfg.time_steps, cfg.image_size, cfg.image_size, cfg.in_channels),
            dtype=jnp.float32,
        )
        output = sps_forward(params, spec_by_name, frames, cfg, args, 0, None, 1)
        self.assertEqual(output.shape[0], cfg.time_steps)
        self.assertEqual(output.shape[1], 1)
        self.assertEqual(output.shape[-1], cfg.dim)

    def test_dvs_preset_matches_reference_input_contract(self) -> None:
        cfg = preset_config("spikformer_dvs_2_256")
        self.assertEqual(cfg.image_size, 128)
        self.assertEqual(cfg.in_channels, 2)
        self.assertEqual(cfg.time_steps, 10)
        self.assertEqual(cfg.depth, 2)
        self.assertEqual(cfg.dim, 256)
        self.assertEqual(cfg.pool_after, (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
