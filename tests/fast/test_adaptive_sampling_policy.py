import copy
import unittest
from collections import Counter

from miles.rollout.adaptive_sampling.policy import AdaptivePromptPolicy, SamplingConfig


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


if __name__ == "__main__":
    unittest.main()
