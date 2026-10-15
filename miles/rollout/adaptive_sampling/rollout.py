"""Native synchronous rollout with a fixed, preselected prompt budget."""

import copy
import math

from miles.rollout.adaptive_sampling.policy import AdaptivePromptPolicy, SamplingConfig, fingerprint
from miles.rollout.data_source import RolloutDataSource
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn

_METADATA_KEY = "miles_adaptive_sampling"
_ROLLOUT_PATH = "miles.rollout.adaptive_sampling.rollout.AdaptiveRolloutFn"


def _validate_args(args):
    if not args.rollout_global_dataset or not args.num_rollout:
        raise ValueError("adaptive sampling requires a global dataset and an explicit num_rollout")
    if args.n_samples_per_prompt < 2:
        raise ValueError("adaptive sampling requires at least two responses per prompt")
    if args.rollout_function_path != _ROLLOUT_PATH:
        raise ValueError(f"adaptive data source requires rollout-function-path={_ROLLOUT_PATH}")
    for name in (
        "fully_async",
        "partial_rollout",
        "dynamic_sampling_filter_path",
        "rollout_sample_filter_path",
        "rollout_all_samples_process_path",
        "buffer_filter_path",
        "rollout_data_postprocess_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"adaptive sampling does not support {name}")
    if getattr(args, "rollout_submission_granularity", None) not in (None, "group"):
        raise ValueError("adaptive sampling requires group-level submission to bound generation")


def _row_identity(sample):
    if not isinstance(sample.prompt, str) or sample.multimodal_inputs:
        raise ValueError("adaptive sampling currently supports text-only rendered prompts")
    if _METADATA_KEY in sample.metadata:
        raise ValueError("dataset uses reserved adaptive sampling metadata")
    # Hash all reward-relevant metadata, not just the prompt. Never persist text.
    return fingerprint({"prompt": sample.prompt, "label": sample.label, "metadata": sample.metadata})


def _stratum(sample, keys):
    values = [sample.metadata.get(key) for key in keys]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("each configured stratum key must name nonempty string metadata")
    return fingerprint(values)


def _trusted_rewards(samples, config):
    rewards = []
    for sample in samples:
        if sample.status.value not in ("completed", "truncated"):
            return None
        reward = sample.reward
        if config.error_key is not None:
            if not isinstance(reward, dict) or reward.get(config.error_key) != 0:
                return None
        if config.reward_key is not None:
            if not isinstance(reward, dict) or config.reward_key not in reward:
                return None
            reward = reward[config.reward_key]
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            return None
        rewards.append(float(reward))
    return rewards


class AdaptiveDataSource(RolloutDataSource):
    """Checkpoint through the native data-source metadata, with no completion cache."""

    def __init__(self, args):
        _validate_args(args)
        self.config = SamplingConfig.from_mapping(getattr(args, "adaptive_sampling", {}))
        super().__init__(args)
        # Stable original order, independent of Dataset.shuffle's mutable view.
        self._rows = tuple(self.dataset.origin_samples)
        self.policy = AdaptivePromptPolicy(
            row_fingerprints=[_row_identity(sample) for sample in self._rows],
            strata=[_stratum(sample, self.config.strata_keys) for sample in self._rows],
            config=self.config,
            seed=args.rollout_seed,
        )
        if not 0 < args.rollout_batch_size <= len(self._rows):
            raise ValueError("rollout_batch_size must fit distinct dataset rows")
        self._shape = (args.rollout_batch_size, args.n_samples_per_prompt)
        self._pending = None

    def begin_rollout(self):
        if self._pending is not None:
            raise RuntimeError("adaptive rollout is already active")
        before = (self.policy.state_dict(), self.sample_group_index, self.sample_index)
        rows, selected = self.policy.select(self.args.rollout_batch_size)
        self._pending = {"rows": rows, "selected": selected, "offset": 0, "before": before, "issued": {}}

    def get_samples(self, num_samples):
        if self._pending is None:
            raise RuntimeError("adaptive source must be driven by AdaptiveRolloutFn")
        offset = self._pending["offset"]
        rows = self._pending["rows"][offset : offset + num_samples]
        if num_samples <= 0 or not rows:
            # Native rollout may request a replacement after a failed generation/RM.
            # Surface that failure instead of silently replenishing the fixed budget.
            raise RuntimeError("adaptive rollout exhausted its fixed prompt budget; no replacement generation")
        result = []
        for row in rows:
            samples = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(self._rows[row])
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                sample.metadata[_METADATA_KEY] = {"row": row, "batch": self.policy.batches}
                self.sample_index += 1
                samples.append(sample)
            self._pending["issued"][self.sample_group_index] = (row, [sample.index for sample in samples])
            self.sample_group_index += 1
            result.append(samples)
        self._pending["offset"] += len(rows)
        return result

    def add_samples(self, samples):
        # Older native synchronous rollout calls this even with an empty abort list.
        if samples:
            raise RuntimeError("adaptive sampling does not replay buffered completions")

    def finish_rollout(self, samples):
        pending = self._pending
        if pending is None or pending["offset"] != self.args.rollout_batch_size:
            raise RuntimeError("adaptive rollout did not consume its exact prompt budget")
        if len(samples) != self.args.rollout_batch_size:
            raise RuntimeError("adaptive rollout returned an incomplete batch")
        seen, observations = set(), []
        for responses in samples:
            if len(responses) != self.args.n_samples_per_prompt:
                raise RuntimeError("adaptive rollout returned an incomplete response set")
            index = responses[0].group_index
            if index in seen or index not in pending["issued"]:
                raise RuntimeError("adaptive rollout returned a duplicate or unknown prompt")
            seen.add(index)
            row, expected = pending["issued"][index]
            if sorted(sample.index for sample in responses) != expected or any(
                sample.group_index != index
                or sample.metadata.get(_METADATA_KEY) != {"row": row, "batch": self.policy.batches}
                for sample in responses
            ):
                raise RuntimeError("adaptive rollout response identity mismatch")
            observations.append((row, _trusted_rewards(responses, self.config)))
        # Validate the entire returned batch before committing any observations.
        accepted = sum(self.policy.observe(row, rewards) for row, rewards in observations)
        metrics = {
            **self.policy.metrics(),
            **{f"selected_{key}": value for key, value in pending["selected"].items()},
            "ignored_reward_observations": len(observations) - accepted,
        }
        self._pending = None
        return {f"adaptive_sampling/{key}": value for key, value in metrics.items()}

    def abort_rollout(self):
        if self._pending is not None:
            state, self.sample_group_index, self.sample_index = self._pending["before"]
            self.policy.load_state_dict(state)
            self._pending = None

    def save(self, checkpoint):
        if self._pending is not None:
            raise RuntimeError("cannot checkpoint an unfinished adaptive rollout")
        self.metadata[_METADATA_KEY] = {"policy": self.policy.state_dict(), "shape": self._shape}
        super().save(checkpoint)

    def load(self, checkpoint=None):
        if self._pending is not None:
            raise RuntimeError("cannot restore during an adaptive rollout")
        # Earlier Miles drivers call load(-1) even for a fresh initialization.
        if checkpoint == -1:
            if self.policy.batches:
                raise ValueError("cannot reset an already-used adaptive sampler")
            return
        super().load(checkpoint)
        state = self.metadata.get(_METADATA_KEY)
        if state is None or tuple(state["shape"]) != self._shape:
            raise ValueError("exact adaptive resume requires matching saved sampler state and batch shape")
        self.policy.load_state_dict(state["policy"])

    def get_buffer_length(self):
        return 0


class AdaptiveRolloutFn(InferenceRolloutFn):
    """Keep native generation/evaluation; add transactional curriculum feedback."""

    def __init__(self, input):
        if not isinstance(input.data_source, AdaptiveDataSource):
            raise ValueError("AdaptiveRolloutFn requires AdaptiveDataSource")
        super().__init__(input)

    async def _call_train(self, input):
        self.data_source.begin_rollout()
        try:
            output = await super()._call_train(input)
            metrics = self.data_source.finish_rollout(output.samples)
        except BaseException:
            self.data_source.abort_rollout()
            raise
        output.metrics = {**(output.metrics or {}), **metrics}
        return output
