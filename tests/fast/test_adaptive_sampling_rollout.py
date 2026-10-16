"""CPU contracts with in-memory dataset/generation boundaries (no model download)."""

import asyncio
import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def _load_adapter():
    """Keep mocks scoped; do not replace loaded Miles modules in other test files."""
    checkpoints = {}

    class Source:
        def __init__(self, args):
            self.args = args
            self.dataset = SimpleNamespace(origin_samples=copy.deepcopy(args.rows))
            self.metadata = {}
            self.sample_group_index = self.sample_index = 0

        def save(self, checkpoint):
            checkpoints[checkpoint] = copy.deepcopy((self.metadata, self.sample_group_index, self.sample_index))

        def load(self, checkpoint):
            self.metadata, self.sample_group_index, self.sample_index = copy.deepcopy(checkpoints[checkpoint])

    class Inference:
        def __init__(self, input):
            self.data_source = input.data_source
            self.state = SimpleNamespace(args=input.args)

        async def _call_train(self, input):
            samples = []
            while len(samples) < self.state.args.rollout_batch_size:
                samples.extend(self.data_source.get_samples(7))
            if getattr(input, "fail", False):
                raise RuntimeError("synthetic generation failure")
            for responses in samples:
                for sample in responses:
                    sample.status = SimpleNamespace(value="completed")
                    sample.reward = {"scenario_score": sample.index % 2, "score": 0.3, "is_error": 0}
            self.data_source.add_samples([])
            return SimpleNamespace(samples=samples[::-1], metrics={"native_metric": 1})

        async def _call_eval(self, input):
            return "native evaluation"

        async def __call__(self, input):
            return await (self._call_eval(input) if input.evaluation else self._call_train(input))

    modules = {}
    for name, attr, value in (
        ("miles.rollout.data_source", "RolloutDataSource", Source),
        ("miles.rollout.inference_rollout.inference_rollout_common", "InferenceRolloutFn", Inference),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, value)
        modules[name] = module
    spec = importlib.util.spec_from_file_location(
        "_adaptive_rollout_test", ROOT / "miles/rollout/adaptive_sampling/rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def _args(**overrides):
    rows = [
        SimpleNamespace(
            prompt=f"Synthetic task {row}",
            label="answer",
            metadata={"theme": str(row % 3)},
            multimodal_inputs=None,
            reward=None,
            response="",
            tokens=[],
            status=SimpleNamespace(value="pending"),
        )
        for row in range(100)
    ]
    return SimpleNamespace(
        **{
            "rows": rows,
            "rollout_global_dataset": True,
            "num_rollout": 20,
            "n_samples_per_prompt": 16,
            "rollout_batch_size": 48,
            "rollout_seed": 123,
            "rollout_function_path": "miles.rollout.adaptive_sampling.rollout.AdaptiveRolloutFn",
            "adaptive_sampling": {"strata_keys": ["theme"], "reward_key": "scenario_score", "error_key": "is_error"},
            **overrides,
        }
    )


class TestAdaptiveRollout(unittest.TestCase):
    def setUp(self):
        self.module = _load_adapter()
        self.args = _args()
        self.source = self.module.AdaptiveDataSource(self.args)

    def _driver(self, source=None):
        return self.module.AdaptiveRolloutFn(SimpleNamespace(args=self.args, data_source=source or self.source))

    def _run(self, source=None, **kwargs):
        return asyncio.run(self._driver(source)(SimpleNamespace(evaluation=False, **kwargs)))

    def test_fresh_complete_rollout_and_native_metrics(self):
        self.source.load(-1)
        result = self._run()
        self.assertEqual(len(result.samples), 48)
        self.assertEqual(sum(len(responses) for responses in result.samples), 768)
        self.assertEqual(result.metrics["native_metric"], 1)
        self.assertEqual(result.metrics["adaptive_sampling/pool_informative"], 48)
        self.assertEqual(result.metrics["adaptive_sampling/prompt_presentations"], 48)

    def test_budget_exhaustion_and_no_completion_reuse(self):
        self._run()
        self.source.begin_rollout()
        samples = self.source.get_samples(100)
        self.assertEqual(len(samples), 48)
        self.assertTrue(all(s.reward is None and not s.response and not s.tokens for ss in samples for s in ss))
        samples[0][0].metadata["theme"] = "changed"
        self.assertNotEqual(samples[0][0].metadata, samples[0][1].metadata)
        with self.assertRaisesRegex(RuntimeError, "fixed prompt budget"):
            self.source.get_samples(1)
        with self.assertRaises(RuntimeError):
            self.source.add_samples(samples)

    def test_invalid_metadata_and_multimodal_fail_at_startup(self):
        for field, value in (("metadata", {}), ("prompt", [{"role": "user", "content": "text"}])):
            args = _args()
            setattr(args.rows[0], field, value)
            with self.assertRaises(ValueError):
                self.module.AdaptiveDataSource(args)

    def test_failure_rolls_back_history_rng_and_indices(self):
        state = self.source.policy.state_dict()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self._run(fail=True)
        self.assertEqual(self.source.policy.state_dict(), state)
        self.assertEqual(self.source.sample_index, 0)
        self.assertEqual(self.source.sample_group_index, 0)
        self._run()

    def test_evaluation_leaves_curriculum_untouched(self):
        before = self.source.policy.state_dict()
        result = asyncio.run(self._driver()(SimpleNamespace(evaluation=True)))
        self.assertEqual(result, "native evaluation")
        self.assertEqual(self.source.policy.state_dict(), before)

    def test_native_metadata_save_restore_next_rollout(self):
        self._run()
        self.source.save("checkpoint")
        other = self.module.AdaptiveDataSource(self.args)
        other.load("checkpoint")
        expected, actual = self._run(), self._run(other)
        self.assertEqual(expected.metrics, actual.metrics)
        self.assertEqual(
            [[s.metadata for s in responses] for responses in expected.samples],
            [[s.metadata for s in responses] for responses in actual.samples],
        )
        self.assertEqual(self.source.sample_index, other.sample_index)

    def test_restore_rejects_different_batch_or_reward_revision(self):
        self._run()
        self.source.save("checkpoint")
        other = self.module.AdaptiveDataSource(_args(n_samples_per_prompt=8))
        with self.assertRaisesRegex(ValueError, "batch shape"):
            other.load("checkpoint")
        other = self.module.AdaptiveDataSource(_args(adaptive_sampling={"reward_version": "different"}))
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            other.load("checkpoint")

    def test_missing_or_unfinished_checkpoint_is_not_exact_resume(self):
        self.source.begin_rollout()
        with self.assertRaises(RuntimeError):
            self.source.save("checkpoint")
        with self.assertRaises(RuntimeError):
            self.source.load("checkpoint")

    def test_reject_filter_and_async_modes(self):
        for option in ("fully_async", "partial_rollout", "dynamic_sampling_filter_path", "rollout_sample_filter_path"):
            with self.assertRaises(ValueError):
                self.module.AdaptiveDataSource(_args(**{option: True}))
        with self.assertRaises(ValueError):
            self.module.AdaptiveDataSource(_args(rollout_submission_granularity="sample"))

    def test_unreliable_judgments_do_not_erase_valid_history(self):
        self._run()
        self.source.begin_rollout()
        samples = self.source.get_samples(48)
        before = self.source.policy.state_dict()["history"]
        for responses in samples:
            for sample in responses:
                sample.status = SimpleNamespace(value="completed")
                sample.reward = {"scenario_score": 0, "is_error": 1}
        metrics = self.source.finish_rollout(samples)
        self.assertEqual(metrics["adaptive_sampling/ignored_reward_observations"], 48)
        self.assertEqual(self.source.policy.state_dict()["history"], before)

    def test_missing_key_nan_and_aborted_are_unreliable(self):
        response = SimpleNamespace(status=SimpleNamespace(value="completed"))
        for reward in ({"scenario_score": 0}, {"scenario_score": float("nan"), "is_error": 0}, {"is_error": 0}):
            response.reward = reward
            self.assertIsNone(self.module._trusted_rewards([response], self.source.config))
        response.reward = {"scenario_score": 1, "is_error": 0}
        response.status.value = "aborted"
        self.assertIsNone(self.module._trusted_rewards([response], self.source.config))
        response.status.value = "truncated"
        self.assertEqual(self.module._trusted_rewards([response], self.source.config), [1])

    def test_duplicate_or_missing_responses_rejected(self):
        self.source.begin_rollout()
        samples = self.source.get_samples(48)
        samples[-1] = samples[0]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.source.finish_rollout(samples)
        self.source.abort_rollout()


class TestCoverageRollout(TestAdaptiveRollout):
    """Run the same native boundary/rollback/resume contracts with scheduling."""

    def setUp(self):
        super().setUp()
        self.args.adaptive_sampling["unseen_schedule"] = "coverage_linear"
        self.source = self.module.AdaptiveDataSource(self.args)


if __name__ == "__main__":
    unittest.main()
