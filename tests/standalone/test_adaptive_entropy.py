"""CPU-only checks for the opt-in entropy controller and checkpoint state."""

import unittest
from types import SimpleNamespace

from miles.backends.training_utils.adaptive_entropy import (
    next_entropy_coefficient,
    saved_entropy_coefficient,
    validate_adaptive_entropy_args,
)


def make_args(**changes):
    return SimpleNamespace(
        **(
            dict(
                adaptive_entropy_target=0.4,
                adaptive_entropy_rate=0.0001,
                adaptive_entropy_max_coef=0.001,
                entropy_coef=0.0,
                observe_training_entropy=True,
                calculate_per_token_loss=True,
                train_backend="megatron",
            )
            | changes
        )
    )


class AdaptiveEntropyTests(unittest.TestCase):
    def test_controller_ramps_from_zero_and_releases_above_target(self):
        beta = next_entropy_coefficient(0.0, 0.2, 0.4, 0.0001, 0.001)
        self.assertAlmostEqual(beta, 0.00002)
        beta = next_entropy_coefficient(beta, 0.6, 0.4, 0.0001, 0.001)
        self.assertAlmostEqual(beta, 0.0)

    def test_controller_clamps_and_rejects_nonfinite_observations(self):
        self.assertAlmostEqual(next_entropy_coefficient(0.00099, 0.0, 0.4, 0.0001, 0.001), 0.001)
        self.assertEqual(next_entropy_coefficient(0.00001, 2.0, 0.4, 0.0001, 0.001), 0.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            next_entropy_coefficient(0.0, float("nan"), 0.4, 0.0001, 0.001)

    def test_invalid_settings_fail_before_training(self):
        for change in (
            {"adaptive_entropy_rate": None},
            {"adaptive_entropy_rate": 0.0},
            {"adaptive_entropy_target": float("nan")},
            {"adaptive_entropy_max_coef": 0.0},
            {"entropy_coef": 0.002},
            {"observe_training_entropy": False},
            {"calculate_per_token_loss": False},
            {"train_backend": "fsdp"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_adaptive_entropy_args(make_args(**change))

    def test_disabled_controller_leaves_ordinary_entropy_configuration_alone(self):
        validate_adaptive_entropy_args(
            make_args(adaptive_entropy_target=None, adaptive_entropy_rate=None, adaptive_entropy_max_coef=None)
        )
        with self.assertRaisesRegex(ValueError, "require"):
            validate_adaptive_entropy_args(make_args(adaptive_entropy_target=None))

    def test_exact_resume_uses_next_step_coefficient_from_matching_common_state(self):
        args = make_args()
        state = {"iteration": 21, "args": make_args(entropy_coef=0.00042)}
        self.assertAlmostEqual(saved_entropy_coefficient(state, args, 21), 0.00042)
        with self.assertRaisesRegex(ValueError, "iteration"):
            saved_entropy_coefficient(state, args, 20)
        with self.assertRaisesRegex(ValueError, "target"):
            saved_entropy_coefficient(state, make_args(adaptive_entropy_target=0.2), 21)
        with self.assertRaisesRegex(ValueError, "coefficient"):
            saved_entropy_coefficient({"iteration": 21, "args": make_args(entropy_coef=0.002)}, args, 21)


if __name__ == "__main__":
    unittest.main()
