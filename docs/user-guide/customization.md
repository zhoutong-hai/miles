---
title: Customization
description: The plug-points where you can drop in your own Python without forking Miles.
---
Most of Miles's behavior can be replaced with user-supplied Python by passing a
`--*-path` flag. This page lists every such hook, the function signature it expects,
and the default it replaces.

## At a glance

| Stage | Flag | Replaces |
|---|---|---|
| **Rollout** | `--rollout-function-path` | The whole rollout loop |
| | `--custom-agent-function-path` | The agent-environment loop inside a TITO session |
| | `--custom-generate-function-path` | A single sample's generation |
| | `--data-source-path` | How prompts are loaded |
| | `--eval-function-path` | The eval rollout |
| **Session** | `--session-message-matcher` | Prefix-replay equivalence |
| **Reward** | `--custom-rm-path` | Reward computation |
| | `--custom-reward-post-process-path` | Reward normalization |
| **Filtering** | `--dynamic-sampling-filter-path` | Per-group filter (DAPO) |
| | `--buffer-filter-path` | Buffer dequeue filter |
| | `--rollout-sample-filter-path` | Per-sample loss filter |
| | `--rollout-all-samples-process-path` | Inspect all samples post-rollout |
| | `--rollout-data-postprocess-path` | Mutate samples post-logprob |
| **Training** | `--custom-loss-function-path` | The loss formula |
| | `--custom-tis-function-path` | Importance sampling correction |
| | `--custom-pg-loss-reducer-function-path` | Loss reduction (Dr.GRPO) |
| | `--custom-convert-samples-to-train-data-path` | Sample to tensor batch |
| **Megatron hooks** | `--custom-megatron-init-path` | After Megatron init |
| | `--custom-megatron-before-log-prob-hook-path` | Before logprob compute |
| | `--custom-megatron-before-train-step-hook-path` | Before each train step |
| | `--custom-megatron-post-save-hook-path` | After each checkpoint save |
| **Logging** | `--custom-rollout-log-function-path` | Train-rollout logging |
| | `--custom-eval-rollout-log-function-path` | Eval-rollout logging |
| **Model** | `--custom-model-provider-path` | Megatron model factory |

---

## Rollout

### `--rollout-function-path`

Replace the entire rollout function. Use this only for fundamentally different flows
such as multi-agent co-evolution.

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) \
        -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
```

**Default:** `miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn`; use `miles.rollout.sglang_rollout.generate_rollout` under `MILES_USE_LEGACY_ROLLOUT_V1=1`.

Plain functions with the signature above are wrapped in a legacy adapter. A class-based rollout
function needs to subclass `miles.rollout.base_types.BaseRolloutFn`.

**Reference:** [`examples/experimental/multi_agent/rollout_with_multi_agents.py`](https://github.com/radixark/miles/blob/main/examples/experimental/multi_agent/rollout_with_multi_agents.py).

### `--custom-generate-function-path`

Replace just the generation step inside the default rollout. Most tool-use, RAG, and
multi-turn workflows live here.

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample:
    ...
```

The hook also accepts the `GenerateFnInput -> GenerateFnOutput` form; both
signatures load through the same adapter. See
[Generate Endpoint](/user-guide/generate-endpoint) for the full contract.

**Reference:** [`examples/experimental/search-r1/generate_with_search.py`](https://github.com/radixark/miles/blob/main/examples/experimental/search-r1/generate_with_search.py).


### `--custom-agent-function-path`

Enabled when you set `--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate`.
Use `--custom-agent-function-path` to specify the async agent or environment loop
that sends OpenAI-compatible chat requests through Miles' TITO session server.


```python
async def run_agent(
    base_url: str,
    prompt,
    request_kwargs: dict,
    metadata: dict,
    **kwargs,
) -> dict | None:
    ...
```

See [Agentic Rollout (TITO)](/user-guide/agentic-rollout) for the full wiring and
message/token ownership contract.


### `--data-source-path`

```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples) -> list[list[Sample]]: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...
```

**Default:** `miles.rollout.data_source.RolloutDataSource`.

#### Fixed-budget reward-adaptive sampling (experimental)

To choose prompts using prior rollout rewards without rejection/refill sampling,
select these two hooks together:

```text
--data-source-path miles.rollout.adaptive_sampling.rollout.AdaptiveDataSource
--rollout-function-path miles.rollout.adaptive_sampling.rollout.AdaptiveRolloutFn
--rollout-submission-granularity group
--custom-config-path curriculum.yaml
--num-rollout 20
```

The custom config owns sampler settings; merge this mapping into an existing
custom config if one is already used:

```yaml
adaptive_sampling:
  weights: [4, 2, 1, 1]  # informative, unseen, solved, hard
  strata_keys: [theme, scenario]  # optional dataset metadata fields
  reward_key: scenario_score  # dictionary field; null for scalar rewards
  error_key: is_error  # require this field to equal zero; null if RM raises on failure
  success_reward: 1.0
  variation_tolerance: 1.0e-6
  warmup_rollouts: 1
  reward_version: my-scorer-v1
```

Set `strata_keys: []` for one pool. Every requested metadata key must be a
nonempty string on every row. `reward_key` selects the **curriculum signal only**;
it does not replace `--reward-key`, rewards, advantages or the training objective.
The RM must expose reliability explicitly or raise on infrastructure errors;
the sampler cannot distinguish a fabricated zero from a genuine failure score.

The first rollout samples within strata without reward preference. Thereafter:

- **Informative:** observed reward range exceeds the tolerance, excluding all-success sets.
- **Unseen:** no trusted observation yet. After exhaustion, this lane explores
  least-recently selected rows across all buckets.
- **Solved:** every observed response reaches `success_reward`.
- **Hard:** other uniform sets, including flat partial credit. No permanent retirement.

The weights are soft targets: with all pools available, a 48-prompt batch aims
for 24/12/6/6. Dataset-proportional stratum credit takes precedence, carrying
rounding credit across batches. Empty buckets backfill without additional
generation; least-recent selection within buckets encourages coverage. History
is updated after a complete rollout, using the latest trusted observation.
Missing/nonfinite/error-marked rewards leave history unchanged and increment
`adaptive_sampling/ignored_reward_observations`. Truncation alone is not an error.

Each rollout preselects exactly `--rollout-batch-size` distinct **rows**, creates
`--n-samples-per-prompt` fresh responses per row, and trains on all returned
responses. Duplicate prompt text in distinct dataset rows is not deduplicated.
No pilot, oversampling, reward-based rejection, old-completion replay or entropy
controller is added. Generation/RM failures surface rather than silently drawing
replacement prompts. This changes the training distribution, without importance
weighting back to the original dataset distribution.

Supported scope: synchronous class-based inference rollout, rendered text-only
prompts, at least two responses per prompt, explicit rollout count. Partial/fully
asynchronous rollout, sample-backfill submission, buffering and custom filters/
postprocessors are rejected. Evaluation remains the native, unchanged path.

Native data-source checkpoints include RNG, stratum credit, row history, sample
IDs and batch shape. Resume rejects a different dataset, sampler configuration
or `reward_version`, or a missing curriculum state; it is not interchangeable
with a sequential sampler checkpoint. Sampler state alone is **not** a resumable
model checkpoint. A failed rollout rolls back sampler state; native generation
recovery remains the caller's responsibility.

Aggregate `adaptive_sampling/*` metrics report pool/selected counts, unique rows
seen and total prompt presentations. Per-response private metadata identifies
the source row and selection batch. Custom loggers must retain returned metrics
if they replace native logging. No prompt text or verifier output is logged by
the sampler. Repeated prompts mean rollout count is not unique-coverage epochs.

Keep entropy regularization and sampling temperature as independent recipe
settings. This sampler does not guarantee an entropy floor or learning gain.
Interpret rewards/entropy on comparable task cohorts: changing the selected
mixture can move batch averages even without learning.

##### Optional coverage schedule

To schedule discovery from remaining unseen rows instead of a fixed bucket
weight, add these fields inside `adaptive_sampling`:

```yaml
unseen_schedule: coverage_linear
unseen_start_fraction: 0.75
unseen_end_fraction: 0.25
```

For each stratum, discovery probability is `end + (start - end) * U / N`,
where `U` counts rows never presented and `N` is the stratum size. It becomes
zero at `U == 0`. The configured unseen bucket weight is ignored in this mode;
the other weights retain their relative proportions (4:1:1 in the example).
The outer dataset-proportional stratum allocator is unchanged. Selection uses
the saved seeded RNG, with no duplicate rows inside a batch. Percentages are
probabilities, not exact batch quotas or a guaranteed full-coverage deadline.
The existing warmup applies first; unavailable pools fall back to available rows.

Revisits use the latest trusted reward bucket and least-recent selection within
that bucket. A presented row with no trusted reward is not new and is not labeled
hard: it has a neutral refresh lane of weight one alongside available revisit
buckets. Therefore legacy `pool_unseen`/`selected_unseen` (unrated history) need
not equal never-presented coverage. Use the additional metrics:

- `scheduled_unseen_fraction`: average pre-fallback probability over selected slots.
- `selected_new_rows`, `selected_revisit_rows`, `actual_new_fraction`: actual coverage choices.
- `unseen_rows_remaining`: never-presented rows after selection.

Coverage is derived from visit counts and survives native save/restore, together
with RNG and selection metrics. Changing the schedule or endpoints rejects an
exact resume. Omitted `unseen_schedule` retains the original fixed behavior and
checkpoint identity. This schedule changes coverage, not the reward objective,
entropy coefficient, temperature, or the number of generated responses.

### `--eval-function-path`

Same signature as `--rollout-function-path`. Defaults to whatever rollout function is
configured.

---

## Session

### `--session-message-matcher`

Some harnesses do not replay history verbatim — they reserialize tool-call arguments or drop `reasoning_content` — and the default `strict` matcher counts that as divergence (v1 rollback, v2 branching). This flag loosens what "the same message" means during replay: choose a looser built-in selector (see [Agentic Rollout (TITO)](/user-guide/agentic-rollout#choose-replay-matching)) or supply your own matcher via a trusted dotted import path:

```python
def matcher(stored_message: dict[str, Any], replayed_message: dict[str, Any]) -> bool:
    ...
```

`stored_message` is the authoritative stored message; return `True` to accept the replayed message as the same history at that position. Keep the matcher a fast, synchronous, side-effect-free equivalence check — exceptions or non-`bool` results return HTTP 500, and startup fails if the path does not resolve.

---

## Reward

### `--custom-rm-path`

```python
async def custom_rm(args, sample: Sample) -> float:
    ...

# Batched mode (set --group-rm)
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]:
    ...
```

**Built-in `--rm-type` options:** `math`, `dapo`, `deepscaler`, `gemma_math`, `f1`,
`gpqa`, `ifbench`, `remote_rm` (with `--rm-url`), `random`, `deterministic_random`.
Prefixing any of them with `boxed_` (for example `boxed_math`) extracts `\boxed{}`
from the response before grading.

### `--custom-reward-post-process-path`

Hook to normalize rewards differently from the default GRPO normalization.

---

## Filtering

### `--dynamic-sampling-filter-path`

Per-group filter; runs after scoring, before queueing for training.

```python
def filter_function(args, samples: list[Sample], **kwargs) -> FilterOutput:
    return FilterOutput(keep=True, reason=None)
```

**Stock implementation:** `miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.

### `--buffer-filter-path`

Pops samples from the rollout buffer at dequeue time. The default is
`pop_first` in `miles/rollout/data_source.py`.

```python
def buffer_filter(
    args,
    rollout_id: int | None,
    buffer: list[list[Sample]],
    num_samples: int,
) -> list[list[Sample]]:
    ...
```

### `--rollout-sample-filter-path`

Per-sample, in-place. Set `s.remove_sample = True` to exclude a sample from the loss
(advantage normalization still uses it).

The framework passes `data: list[list[Sample]]` — a list of
`n_samples_per_prompt`-size groups — so iterate the outer list once to reach `Sample`
objects:

```python
def filter_function(args, data: list[list[Sample]]) -> None:
    for group in data:
        for s in group:
            if not_good(s):
                s.remove_sample = True
```

### `--rollout-all-samples-process-path`

Runs after rollout completes and can see all samples, including filtered ones.
Useful for logging or analysis.

### `--rollout-data-postprocess-path`

Runs after log probabilities have been computed but before training. Useful for
updating loss masks based on per-token logprobs.

---

## Training

### `--custom-loss-function-path`

Replace the GRPO/PPO loss. Requires `--loss-type custom_loss`. Useful for novel
objectives or multi-objective work.

### `--custom-tis-function-path`

Importance sampling correction for off-policy training when train and inference
diverge.

**Reference:** [`examples/infra_features/train_infer_mismatch_helper/mis.py`](https://github.com/radixark/miles/blob/main/examples/infra_features/train_infer_mismatch_helper/mis.py).

### `--custom-pg-loss-reducer-function-path`

```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    ...
```

Use case: Dr.GRPO divides by a constant instead of effective token count.
**Reference:** [`examples/experimental/DrGRPO/custom_reducer.py`](https://github.com/radixark/miles/blob/main/examples/experimental/DrGRPO/custom_reducer.py).

### `--custom-convert-samples-to-train-data-path`

```python
def convert_samples_to_train_data(args, samples) -> dict:
    return {
        "tokens":           [...],
        "response_lengths": [...],
        "rewards":          [...],
        "raw_reward":       [...],
        "truncated":        [...],
        "sample_indices":   [...],
        "loss_masks":       [...],
        # optional
        "round_number":            [...],
        "rollout_log_probs":       [...],
        "rollout_routed_experts":  [...],
        "metadata":                [...],
        "multimodal_train_inputs": [...],
        "teacher_log_probs":       [...],
    }
```

---

## Megatron hooks

| Flag | Signature |
|---|---|
| `--custom-megatron-init-path` | `def custom_init(args) -> None` |
| `--custom-megatron-before-log-prob-hook-path` | `def custom_hook(args, model, store_prefix) -> None` |
| `--custom-megatron-before-train-step-hook-path` | `def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None` |
| `--custom-megatron-post-save-hook-path` | `def hook(args, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None) -> None` |

The Megatron init, log-prob, and train-step hooks give access to the live model
and optimizer, useful for custom probes, weight clipping, or surgical interventions.
The post-save hook runs on rank 0 after checkpoint save completion and receives
the saved checkpoint paths instead of live model objects.

---

## Logging

```python
# Training rollouts
def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    ...

# Eval rollouts
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    ...
```

Return `True` to suppress Miles's default logging, `False` to layer on top.

---

## Model

### `--custom-model-provider-path`

Replace Megatron's default model factory.

```python
def custom_model_provider(
    pre_process: bool,
    post_process: bool,
    vp_stage: int | None = None,
) -> GPTModel:
    ...
```

---

## Worked example

A custom rollout plus a custom reward in one launch script:

```bash
ROLLOUT_ARGS+=(
   --custom-generate-function-path my_pkg.search_rollout.generate
   --custom-rm-path                my_pkg.rewards.f1_with_grounding
   --metadata-key metadata
   --rollout-max-response-len 4096
)
```

That is the entire delta from the stock GRPO recipe, with no source changes to Miles.

→ Next: [Server arguments reference](/user-guide/cli-reference)
