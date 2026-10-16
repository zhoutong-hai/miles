import copy
import unittest
from collections import Counter

from miles.rollout.adaptive_sampling.policy import (
    AdaptivePromptPolicy,
    SamplingConfig,
    coverage_unseen_fraction,
    fingerprint,
)


def policy(size=128, strata=None, **config):
    return AdaptivePromptPolicy(
        row_fingerprints=[str(row) for row in range(size)],
        strata=strata or ["all"] * size,
        config=SamplingConfig(**config),
        seed=42,
    )


class TestAdaptiveSamplingPolicy(unittest.TestCase):
    def test_exact_budget_unique_and_deterministic(self):
        left, right = policy(), policy()
        for _ in range(20):
            rows, selected = left.select(48)
            self.assertEqual((rows, selected), right.select(48))
            self.assertEqual(len(set(rows)), 48)
            self.assertEqual(sum(selected.values()), 48)

    def test_expected_mix_when_all_pools_available(self):
        sampler = policy(200, warmup_rollouts=0)
        for row in range(50):
            sampler.observe(row, [0, 1] * 8)
        for row in range(50, 100):
            sampler.observe(row, [1] * 16)
        for row in range(100, 150):
            sampler.observe(row, [0] * 16)
        _, mix = sampler.select(48)
        self.assertEqual(mix, dict(informative=24, unseen=12, solved=6, hard=6))

    def test_stratum_proportions_over_rolling_window(self):
        strata = ["large"] * 85 + ["small"] * 14 + ["rare"]
        sampler = policy(100, strata=strata)
        counts = Counter()
        for _ in range(100):
            rows, _ = sampler.select(13)
            counts.update(strata[row] for row in rows)
        for key, expected in (("large", 1105), ("small", 182), ("rare", 13)):
            self.assertLessEqual(abs(counts[key] - expected), 1)

    def test_missing_pools_backfill_without_extra_rows(self):
        sampler = policy(60)
        for row in range(60):
            sampler.observe(row, [1] * 16)
        for _ in range(4):
            rows, mix = sampler.select(48)
            self.assertEqual(len(rows), 48)
            self.assertEqual(mix["solved"], 48)
        self.assertEqual(sampler.metrics()["unique_rows_seen"], 60)

    def test_unreliable_rewards_do_not_change_history(self):
        sampler = policy()
        sampler.observe(0, [0, 1])
        before = sampler.state_dict()
        for rewards in (None, [], [float("nan")], [float("inf")]):
            self.assertFalse(sampler.observe(0, rewards))
        self.assertEqual(sampler.state_dict(), before)

    def test_hard_and_solved_are_reversible(self):
        sampler = policy()
        for _ in range(100):
            sampler.observe(0, [0] * 16)
        sampler.observe(0, [0, 1])
        self.assertEqual(sampler.metrics()["pool_informative"], 1)
        sampler.observe(0, [1] * 16)
        sampler.observe(0, [0] * 16)
        self.assertEqual(sampler.metrics()["pool_hard"], 1)
        sampler.observe(0, [0.3] * 16)
        self.assertEqual(sampler.metrics()["pool_hard"], 1)

    def test_checkpoint_exact_next_selection(self):
        original = policy()
        rows, _ = original.select(48)
        for row in rows:
            original.observe(row, [row % 2, 1])
        restored = policy()
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        for _ in range(10):
            self.assertEqual(original.select(48), restored.select(48))

    def test_checkpoint_rejects_scoring_or_dataset_change(self):
        state = policy().state_dict()
        for sampler in (policy(129), policy(reward_version="new"), policy(weights=(1, 1, 1, 1))):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                sampler.load_state_dict(state)

    def test_small_dataset_and_invalid_config(self):
        with self.assertRaises(ValueError):
            policy(47).select(48)
        for values in ({"weights": (1, 0, 1, 1)}, {"weights": (1, 2)}, {"warmup_rollouts": -1}):
            with self.assertRaises(ValueError):
                SamplingConfig(**values)
        with self.assertRaises(TypeError):
            SamplingConfig.from_mapping({"typo": True})

    def test_soft_targets_do_not_starve_observed_rows(self):
        sampler = policy(128, warmup_rollouts=0)
        for row in range(128):
            sampler.observe(row, [0, 1] if row < 126 else [0, 0])
        seen = set()
        for _ in range(20):
            rows, _ = sampler.select(48)
            seen.update(rows)
        self.assertEqual(len(seen), 128)


class TestCoverageSampling(unittest.TestCase):
    def test_formula_endpoints_and_monotonicity(self):
        values = [coverage_unseen_fraction(unseen=n, total=100, start=0.75, end=0.25) for n in range(101)]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[50], 0.5)
        self.assertEqual(values[100], 0.75)
        self.assertGreater(values[1], 0.25)
        self.assertEqual(values, sorted(values))
        for unseen, total in ((-1, 100), (101, 100), (0, 0)):
            with self.assertRaises(ValueError):
                coverage_unseen_fraction(unseen=unseen, total=total, start=0.75, end=0.25)

    def test_config_validation_and_old_identity(self):
        for settings in (
            {"unseen_schedule": "typo"},
            {"unseen_end_fraction": 0},
            {"unseen_start_fraction": 1.1},
            {"unseen_start_fraction": 0.2},
            {"unseen_end_fraction": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SamplingConfig(**settings)
        expected = fingerprint(
            {
                "rows": [str(row) for row in range(128)],
                "strata": ["all"] * 128,
                "config": {
                    "weights": (4, 2, 1, 1),
                    "strata_keys": (),
                    "reward_key": None,
                    "error_key": None,
                    "success_reward": 1.0,
                    "variation_tolerance": 1e-6,
                    "warmup_rollouts": 1,
                    "reward_version": "default",
                },
                "seed": 42,
            }
        )
        self.assertEqual(policy().identity, expected)
        self.assertEqual(policy().state_dict()["version"], 1)

    def test_schedule_is_local_to_scenario_and_pool_availability(self):
        sampler = policy(
            200, strata=["done"] * 100 + ["new"] * 100, unseen_schedule="coverage_linear", warmup_rollouts=0
        )
        for row in range(100):
            sampler._history[row]["visits"] = 1
            sampler.observe(row, [1] * 16)
        rows, _ = sampler.select(48)
        self.assertEqual(sum(row < 100 for row in rows), 24)
        self.assertEqual(sampler.metrics()["selected_new_rows"], 24)
        self.assertEqual(sampler.metrics()["selected_revisit_rows"], 24)
        self.assertEqual(sampler.metrics()["unseen_rows_remaining"], 76)
        # Exhausted strata contribute zero, not the configured 25% endpoint.
        self.assertGreater(sampler.metrics()["scheduled_unseen_fraction"], 0.3)
        self.assertLess(sampler.metrics()["scheduled_unseen_fraction"], 0.4)

    def test_presented_without_reward_is_not_unseen_or_hard(self):
        sampler = policy(48, unseen_schedule="coverage_linear")
        rows, _ = sampler.select(48)
        for row in rows:
            self.assertFalse(sampler.observe(row, None))
        rows, _ = sampler.select(48)
        self.assertEqual(len(set(rows)), 48)
        self.assertEqual(sampler.metrics()["selected_new_rows"], 0)
        self.assertEqual(sampler.metrics()["scheduled_unseen_fraction"], 0)
        self.assertEqual(sampler.metrics()["pool_hard"], 0)
        self.assertEqual(sampler.metrics()["pool_unseen"], 48)  # Unrated history, not coverage.

    def test_all_seen_revisit_relative_weights(self):
        sampler = policy(600, unseen_schedule="coverage_linear", warmup_rollouts=0)
        for row in range(600):
            sampler._history[row]["visits"] = 1
            sampler.observe(row, ([0, 1], [1, 1], [0, 0])[row // 200])
        totals = Counter()
        for _ in range(100):
            rows, mix = sampler.select(48)
            self.assertEqual(len(set(rows)), 48)
            totals.update(mix)
            self.assertEqual(sampler.metrics()["scheduled_unseen_fraction"], 0)
        for key, fraction in (("informative", 4 / 6), ("solved", 1 / 6), ("hard", 1 / 6)):
            self.assertAlmostEqual(totals[key] / 4800, fraction, delta=0.025)

    def test_full_coverage_no_retirement_and_proportional_slots(self):
        strata = ["large"] * 850 + ["small"] * 140 + ["rare"] * 10
        sampler = policy(1000, strata=strata, unseen_schedule="coverage_linear")
        counts = Counter()
        for _ in range(120):
            rows, _ = sampler.select(48)
            counts.update(strata[row] for row in rows)
            for row in rows:
                sampler.observe(row, ([0, 1], [1, 1], [0, 0])[row % 3])
        self.assertEqual(sampler.metrics()["unique_rows_seen"], 1000)
        self.assertEqual(sampler.metrics()["unseen_rows_remaining"], 0)
        self.assertEqual(sampler.metrics()["scheduled_unseen_fraction"], 0)
        for key, fraction in (("large", 0.85), ("small", 0.14), ("rare", 0.01)):
            self.assertLessEqual(abs(counts[key] - 5760 * fraction), 1)

    def test_determinism_restore_and_schedule_mismatch(self):
        sampler = policy(unseen_schedule="coverage_linear")
        rows, _ = sampler.select(48)
        for row in rows:
            sampler.observe(row, [row % 2, 1])
        saved = copy.deepcopy(sampler.state_dict())
        restored = policy(unseen_schedule="coverage_linear")
        restored.load_state_dict(saved)
        for _ in range(10):
            self.assertEqual(sampler.select(48), restored.select(48))
            self.assertEqual(sampler.metrics(), restored.metrics())
        for other in (policy(), policy(unseen_schedule="coverage_linear", unseen_start_fraction=0.8)):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                other.load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
