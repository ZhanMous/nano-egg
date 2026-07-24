import unittest

import jax
import jax.numpy as jnp
import numpy as np

from evo_trqr import (
    CONTINUOUS_DIM,
    CandidateMetrics,
    DiscreteGenome,
    apply_evo_trqr,
    ask_antithetic,
    better_candidate,
    binary_mutual_information,
    candidate_order_key,
    decode_continuous,
    initial_continuous_genome,
)


def balanced_binary_tensor() -> jax.Array:
    return jnp.asarray(
        [
            [[[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]],
            [[[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
        ]
    )


class EvoTRQRTests(unittest.TestCase):
    def test_continuous_genome_is_16d_and_decodes_to_bounded_controls(self) -> None:
        genome = initial_continuous_genome()
        decoded = decode_continuous(genome)
        self.assertEqual(genome.shape, (CONTINUOUS_DIM,))
        self.assertTrue(
            np.all((np.asarray(decoded.layer_gates) > 0.0) & (np.asarray(decoded.layer_gates) < 1.0))
        )
        self.assertTrue(
            np.all((np.asarray(decoded.layer_budgets) > 0.0) & (np.asarray(decoded.layer_budgets) < 0.8))
        )
        self.assertTrue(0.0 <= float(decoded.warmup) < float(decoded.schedule_end) <= 1.0)

    def test_binary_mi_is_one_for_balanced_identical_variables(self) -> None:
        values = balanced_binary_tensor()
        mi = binary_mutual_information(values, values, axis=3)
        np.testing.assert_allclose(np.asarray(mi), np.ones((2, 1, 2, 1)), atol=1e-6)

    def test_master_strength_zero_is_exact_identity_for_every_recalibration(self) -> None:
        x = balanced_binary_tensor()
        genome = initial_continuous_genome()
        for recalibration in ("zero", "redistribute", "renorm"):
            output, diagnostics = apply_evo_trqr(
                x,
                genome,
                DiscreteGenome(reference="previous", recalibration=recalibration),
                layer_index=0,
                progress=1.0,
                random_key=jax.random.PRNGKey(0),
                master_strength=0.0,
            )
            self.assertTrue(np.array_equal(np.asarray(output), np.asarray(x)))
            self.assertTrue(np.isfinite(float(diagnostics.local_mi_mean)))
            self.assertTrue(np.isfinite(float(diagnostics.global_mi_mean)))

    def test_common_uniforms_couple_masks_across_genomes(self) -> None:
        x = balanced_binary_tensor()
        genome = initial_continuous_genome()
        discrete = DiscreteGenome(reference="previous")
        decoded = decode_continuous(genome)
        probability_shape = (x.shape[0], x.shape[1], x.shape[2], 1)
        uniforms = jnp.full(probability_shape, 0.1)
        first, first_diag = apply_evo_trqr(
            x,
            genome,
            discrete,
            layer_index=0,
            progress=1.0,
            random_key=jax.random.PRNGKey(1),
            uniforms=uniforms,
        )
        second, second_diag = apply_evo_trqr(
            x,
            genome,
            discrete,
            layer_index=0,
            progress=1.0,
            random_key=jax.random.PRNGKey(999),
            uniforms=uniforms,
        )
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertEqual(float(first_diag.mask_fraction), float(second_diag.mask_fraction))
        self.assertEqual(decoded.layer_gates.shape, (4,))

    def test_all_spatial_and_recalibration_modes_preserve_shape_and_finiteness(self) -> None:
        x = jnp.tile(balanced_binary_tensor(), (1, 1, 2, 1))
        genome = initial_continuous_genome()
        for spatial_scale in ("token", "channel", "block"):
            for recalibration in ("zero", "redistribute", "renorm"):
                output, diagnostics = apply_evo_trqr(
                    x,
                    genome,
                    DiscreteGenome(
                        reference="multi_lag",
                        lag_set=(1, 2),
                        spatial_scale=spatial_scale,
                        recalibration=recalibration,
                    ),
                    layer_index=0,
                    progress=1.0,
                    random_key=jax.random.PRNGKey(11),
                )
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(np.isfinite(np.asarray(output)).all())
                self.assertTrue(np.isfinite(float(diagnostics.mask_probability_mean)))

    def test_epsilon_constraint_prefers_feasible_candidate_before_accuracy(self) -> None:
        feasible = CandidateMetrics(accuracy=0.70, loss=0.8, energy_ratio=0.80)
        infeasible = CandidateMetrics(accuracy=0.99, loss=0.1, energy_ratio=0.91)
        self.assertTrue(better_candidate(feasible, infeasible, energy_budget=0.90))
        self.assertGreater(candidate_order_key(feasible, 0.90), candidate_order_key(infeasible, 0.90))

    def test_antithetic_ask_is_symmetric_and_can_freeze_schedule_dimensions(self) -> None:
        center = initial_continuous_genome()
        active = np.ones_like(center)
        active[14:] = 0.0
        plus, minus, noise = ask_antithetic(
            center,
            pairs=3,
            sigma=0.2,
            rng=np.random.default_rng(7),
            active_dimensions=active,
        )
        expected = np.broadcast_to(center[None, :], plus.shape)
        np.testing.assert_allclose((plus + minus) / 2.0, expected, atol=1e-7)
        np.testing.assert_array_equal(noise[:, 14:], 0.0)


if __name__ == "__main__":
    unittest.main()
