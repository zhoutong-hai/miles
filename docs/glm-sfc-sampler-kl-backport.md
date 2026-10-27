# GLM SFC + sampler KL on the retained E010 runtime

Branch: `codex/e010-glm-sfc-sampler-kl`, based on
`d951cc5765e2abf1f6d07815ff2ed2f92cd49c69` (i021 coverage sampler).
This is a local source backport for the matched i023/i024 preparations, not
GLM GPU/runtime qualification or permission to launch either experiment.

## Source provenance

Native SFC reference: `209bdbd1c280bf3dffd0a61cb9def25abdd991a5`.
Relevant upstream donors: `2fcf55081` (math/selected-token probabilities),
`9b061baa4` (candidate preservation), `991c17865` (native collection),
`47b793fa6` (training integration), `eb0f5e3da` (current masking/router contract).
Head/tail sampler KL comes from `ee1732e519699678289ac59ab9a1624d881786bb`.
The equation modules are copied from that final donor snapshot with old-layout
imports restored. The retained runtime's response iterator, data conversion,
parser and loss dispatcher are adapted locally instead of importing E011's
new lifecycle, Qwen bridge or dependency stack. The pre-existing long response
iterator is retained to keep the packed/CP alignment change reviewable.

## Selected behavior

Both arms use `--loss-type score_centering --rollout-top-logprobs-num 128
--sampler-kl-coef 0.03 --entropy-coef 0 --observe-training-entropy`.
i023 sets `--score-centering-is none`; i024 sets
`--score-centering-is tis --score-centering-tis-clip 2`.
Generic `--use-tis` stays off. There is no lower floor or PPO surrogate.
The common source contains both selected modes.

Sampler KL is KL(q_sampler || p_trainer), with q detached, an exact transported
head and a proportional modeled tail. Its gradient is independent of advantages
and TIS; zero advantages do not disable it. Only the omitted tail is modeled:
the sampled-token TIS weight always uses its actual generation-time q, including
when that token is outside the saved head. Observed entropy at coefficient zero
is detached. Existing global original-response-token reduction is retained.

Training requests record native `/generate` top-logprobs alongside sampled-token
logprobs. Candidates survive Sample save/restore, strip/retry, tool/merge gaps,
typed-ragged object-store registration, reordered microbatches and response-axis
CP slicing/redistribution. Candidate integrity is validated at conversion.
Evaluation remains greedy and does not request training candidates.
There is no post-generation rescoring fallback.

The backport deliberately supports the selected native, text, unfiltered
sampling path: top_p=1, top_k=-1, min_p=0, matched positive temperature, THD,
group-centered GRPO, native importance none/tis. Filtered/support replay,
speculation, partial/fully asynchronous rollout, custom converters, generic TIS
and the newer session candidate API are not qualified here. Custom generation
must supply valid generation-time candidates or conversion fails.
Rollout servers must use `SGLANG_RETURN_ORIGINAL_LOGPROB=0`; actual server
semantics, top128 availability, request routing and payload overhead still need
verification on the retained SGLang runtime.

## Diagnostics and remaining work

Basic detached diagnostics include the existing guarded mismatch, SFC head/tail
mass and correction, weight moments, raw lower/upper ratio tails crossed with
advantage sign (even when weighting is off), actual cap/floor fractions and
signed/absolute weight-times-advantage changes. Sampler KL raw/weighted loss and
coefficient use the original valid-token denominator. The applied cap fraction
is zero for importance mode none; raw upper-tail diagnostics remain available.

This does not complete E010's additional logging specification. Private aligned
pre-update token sidecars, global quantiles/ESS, theme/position crosses,
counterfactual floor changes, equal-response entropy, sealed fixed-prefix probes,
sparse component-gradient audits and overhead measurements remain to be bound
and qualified before launch. No fixed probe inputs are fabricated locally.

The qualified GLM attention kernels/indexer, router observer/frozen gates,
coverage sampler, actor lifecycle, checkpointing and weight publication sources
are unchanged from the base. Existing Megatron/SGLang pins are retained as
references; no third-party package upgrade is included. Changed paths still
require the actual GLM TP4/PP8/CP4 model-bearing qualification, GPU compiled-path
checks, memory/timing evidence, full-state/sampler save/resume binding and first
ordinary-update evidence. CPU tests cannot establish these properties.

## Local validation

Use `PYTHONPATH=.` with a Python environment containing PyTorch/numpy/pytest.
Run focused tests under `tests/standalone/test_score_centering*.py`,
`tests/standalone/test_glm_sfc*.py`, plus the existing adaptive-sampling,
objective-normalization, router and GLM contract tests.
`--noconftest -o addopts=''` avoids the newer heavy runtime fixtures.
On the local Mac, `TORCH_COMPILE_DISABLE=1` is required because the CPU inductor
SDK/linker path is unavailable. Those results are eager CPU evidence; six CUDA
tests remain skipped. A four-process CPU/Gloo test performs real TP4 and CP4
collectives, including candidate-width tensors and empty response ranks.
