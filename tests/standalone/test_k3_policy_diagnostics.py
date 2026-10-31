"""Dense oracles, exact native TIS boundaries and sealed DP ownership."""

import json
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import policy_metrics as metrics
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss_hub.policy_audit import component_statistics, distribution_statistics
from miles.utils.policy_diagnostics import validate_policy_diagnostics_args


@pytest.mark.parametrize("optimizer_steps", [1, 2])
def test_actor_replay_advantage_and_observer_lifecycle(monkeypatch, optimizer_steps):
    from miles.backends.training_utils.loss_hub import policy_diagnostics, policy_probe

    source = Path(__file__).resolve().parents[2] / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(source.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    function = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "train_actor")
    function.decorator_list = []
    phases = []
    manager = SimpleNamespace(
        name="routing",
        data_key="rollout_routed_experts",
        replays=[],
        register_replay_list_func=None,
        if_sp_region=False,
        replay_indices_are_token_positions=False,
        enabled=True,
    )

    def configure(*args):
        assert phases == ["fill_original_replay", "advantages", "probe"]
        phases.append("configure_observer")

    monkeypatch.setattr(policy_diagnostics, "configure", configure)
    monkeypatch.setattr(policy_probe, "run_fixed_probe", lambda *args, **kwargs: phases.append("probe"))

    class AtTraining(Exception):
        pass

    def train(*args, **kwargs):
        assert phases == ["fill_original_replay", "advantages", "probe", "configure_observer"]
        raise AtTraining

    environment = {
        "RolloutBatch": dict,
        "WitnessInfo": object,
        "TrainStepOutput": object,
        "get_data_iterator": lambda *a: (None, [1] * optimizer_steps),
        "all_replay_managers": [manager],
        "fill_replay_data": lambda **k: phases.append("fill_original_replay"),
        "inverse_timer": lambda *a: nullcontext(),
        "timer": lambda *a: nullcontext(),
        "compute_advantages_and_returns": lambda *a: phases.append("advantages"),
        "log_train_advantage_computation_event": lambda *a: None,
        "log_rollout_data": lambda *a: None,
        "get_num_rollouts": lambda *a: 1,
        "train": train,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), environment)
    args = SimpleNamespace(
        policy_diagnostics_dir="/private",
        skip_actor_forward_only=True,
        compute_advantages_and_returns=True,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        get_mismatch_metrics=True,
        use_critic=False,
    )
    owner = SimpleNamespace(
        args=args,
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        weights_backuper=SimpleNamespace(backup_tags={"actor"}),
        _active_model_tag="actor",
        _switch_model=lambda *a: None,
        _set_replay_stage=lambda *a: None,
        _use_rollout_replay=lambda *a: True,
        rollout_data_postprocess=None,
        weight_updater=SimpleNamespace(weight_version="v1"),
        _ft_test_action_executor=None,
    )
    with pytest.raises(AtTraining if optimizer_steps == 1 else AssertionError):
        environment["train_actor"](owner, 0, {}, witness_info=None, attempt=0)
    if optimizer_steps == 2:
        assert phases == []


@pytest.mark.parametrize("advantage", [-0.5, 0.0, 0.5])
def test_real_native_loss_logging_preserves_gradients(tmp_path, monkeypatch, advantage):
    from miles.backends.training_utils import cp_utils, parallel
    from miles.backends.training_utils.loss_hub import losses, policy_diagnostics

    state = SimpleNamespace(
        cp=SimpleNamespace(size=1, rank=0, group=None),
        tp=SimpleNamespace(size=1, rank=0, group=None),
        is_pp_last_stage=True,
        effective_dp=SimpleNamespace(size=1, rank=0, group=None),
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setenv("MILES_SOURCE_COMMIT", "cpu-native-test")
    for name in ("MILES_INPUT_SHA256", "MILES_TOKENIZER_REVISION", "MILES_RESOLVED_CONFIG_SHA256"):
        monkeypatch.setenv(name, "cpu-identity")

    def dense_scores(logits, **kwargs):
        logp = logits[0].log_softmax(-1)
        return {
            "log_probs": [logp.gather(-1, kwargs["unconcat_tokens"][0][1:, None]).squeeze(-1)],
            "entropy": [-(logp.exp() * logp).sum(-1)],
        }

    monkeypatch.setattr(losses, "get_log_probs_and_entropy", dense_scores)
    args = SimpleNamespace(
        loss_type="reinforce",
        skip_actor_forward_only=True,
        use_rollout_logprobs=False,
        entropy_coef=0.0,
        observe_training_entropy=True,
        use_sampling_support_replay=False,
        qkv_format="thd",
        allgather_cp=False,
        advantage_estimator="grpo",
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=True,
        use_tis=True,
        custom_tis_function_path=None,
        tis_clip_low=0.5,
        tis_clip=2.0,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=True,
        use_kl_loss=False,
        policy_diagnostics_dir=None,
        policy_audit_updates=[1],
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        vocab_size=5,
        wandb_run_id="test",
    )
    initial = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(7))
    tokens = torch.tensor([0, 1, 2, 3, 4, 0])
    selected = initial[0].log_softmax(-1).gather(-1, tokens[1:, None]).squeeze(-1)
    q = selected - torch.tensor([0.1, 0.75, 1.0, 2.0, 3.0]).log()
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    batch = dict(
        advantages=[torch.full((5,), advantage)],
        rollout_log_probs=[q],
        loss_masks=[mask],
        total_lengths=[6],
        response_lengths=[5],
        unconcat_tokens=[tokens],
        rollout_mask_sums=[mask.sum()],
        sample_indices=[10],
        weight_versions=[["v1"]],
        policy_sample_metadata=[dict(valid_tokens=4, first_valid_positions=[0, 1, 3, 4], theme="a", scenario="s")],
    )
    reducer = cp_utils.get_sum_of_sample_mean([6], [5], [mask], calculate_per_token_loss=True)
    plain_logits = initial.clone().requires_grad_()
    plain, native = losses.policy_loss_function(args, batch, plain_logits, reducer)
    (plain_gradient,) = torch.autograd.grad(plain, plain_logits)
    assert not {"ppo_kl", "pg_clipfrac", "ess_ratio", "ois"} & native.keys()
    args.policy_diagnostics_dir = str(tmp_path)
    policy_diagnostics.configure(args, 0, "v1", batch)
    observed_logits = initial.clone().requires_grad_()
    observed, _ = losses.policy_loss_function(args, batch, observed_logits, reducer)
    (observed_gradient,) = torch.autograd.grad(observed, observed_logits)
    torch.testing.assert_close(observed, plain, rtol=0, atol=0)
    torch.testing.assert_close(observed_gradient, plain_gradient, rtol=0, atol=0)
    dense = initial.clone().requires_grad_()
    score = dense[0].log_softmax(-1).gather(-1, tokens[1:, None]).squeeze(-1)
    expected = (-advantage * (score.detach() - q).exp().clamp(0.5, 2.0) * score * mask).sum()
    (expected_gradient,) = torch.autograd.grad(expected, dense)
    torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(observed_gradient, expected_gradient)
    (frame,) = metrics.read_frames(tmp_path / "paired_scores/update-000001/dp-0.bin")
    assert frame["data"]["mask"].sum() == 4
    from miles.utils.tracking_utils import tracking

    monkeypatch.setattr(tracking, "define_step_key_metric_group", lambda *a: None)
    monkeypatch.setattr(tracking, "log", lambda *a, **k: None)
    policy_diagnostics.finish(args, "NORMAL")
    complete = json.loads((tmp_path / "paired_scores/update-000001/complete.json").read_text())
    assert complete["metrics"]["policy/audit/positions"] == 4
    assert complete["artifact_sha256"]


def _distributed_observer_worker(rank, root):
    import torch.distributed as dist
    from miles.backends.training_utils import parallel
    from miles.backends.training_utils.loss_hub import policy_diagnostics
    from miles.utils.tracking_utils import tracking
    import os

    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "rendezvous"), rank=rank, world_size=4)
    parallel.set_parallel_state(
        SimpleNamespace(
            cp=SimpleNamespace(size=1, rank=0, group=None),
            tp=SimpleNamespace(size=1, rank=0, group=None),
            is_pp_last_stage=True,
            effective_dp=SimpleNamespace(size=4, rank=rank, group=dist.group.WORLD),
        )
    )
    os.environ["MILES_SOURCE_COMMIT"] = "distributed-mechanics"
    for name in ("MILES_INPUT_SHA256", "MILES_TOKENIZER_REVISION", "MILES_RESOLVED_CONFIG_SHA256"):
        os.environ[name] = "cpu-identity"
    args = SimpleNamespace(
        policy_diagnostics_dir=root,
        wandb_run_id="test",
        policy_audit_updates=[],
        qkv_format="thd",
        rollout_temperature=1.0,
    )
    data, samples = example()
    ids = ([10, 20], [30], [40, 50], [])[rank]
    rows = [torch.where(data["sample_id"] == identity)[0] for identity in ids]
    batch = dict(
        sample_indices=ids,
        policy_sample_metadata=[samples[i] for i in ids],
        weight_versions=[["v1"] for _ in ids],
        advantages=[data["advantage"][r] for r in rows],
        loss_masks=[torch.ones(len(r)) for r in rows],
        response_lengths=[len(r) for r in rows],
        unconcat_tokens=[torch.cat((torch.tensor([0]), data["token_id"][r])) for r in rows],
    )
    policy_diagnostics.configure(args, 0, "v1", batch)
    selected = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
    policy_diagnostics.observe(
        args,
        batch,
        None,
        logp=data["logp"][selected],
        logq=data["logq"][selected],
        ratio=data["ratio"][selected],
        weight=data["weight"][selected],
        advantage=data["advantage"][selected],
        active=torch.ones(len(selected), dtype=torch.bool),
        entropy=data["entropy"][selected],
    )
    tracking.define_step_key_metric_group = lambda *a: None
    tracking.log = lambda *a, **k: None
    policy_diagnostics.finish(args, "NORMAL")
    from miles.backends.training_utils.loss_hub.policy_probe import _persist_probe

    selected_probe = list(range(rank, 64, 4))
    probe_values = torch.tensor(selected_probe, dtype=torch.float32)
    _persist_probe(
        args,
        {"probe_entropy": [probe_values], "probe_head_mass": [torch.ones(16)], "probe_count": [torch.ones(16)]},
        {"identities": [[i, i] for i in range(64)]},
        "sealed-probe",
        0,
        0.1,
        selected_probe,
    )
    # Same four real processes now check TP vocabulary reduction, including a
    # padded logit with enormous mass that must be excluded.
    global_logits = torch.randn(7, 8, generator=torch.Generator().manual_seed(91))
    global_logits[:, 7] = 100.0
    local = global_logits[:, rank * 2 : (rank + 1) * 2].clone()
    entropy, mass, _, _ = distribution_statistics(local, vocab_size=7, group=dist.group.WORLD)
    dense_logp = global_logits[:, :7].log_softmax(-1)
    torch.testing.assert_close(entropy, -(dense_logp.exp() * dense_logp).sum(-1))
    torch.testing.assert_close(mass, torch.ones(7))
    dist.destroy_process_group()


def test_real_four_rank_gloo_observer(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_observer_worker, args=(str(tmp_path),), nprocs=4, join=True)
    complete = json.loads((tmp_path / "paired_scores/update-000001/complete.json").read_text())
    assert complete["binding"]["dp_size"] == 4
    assert complete["metrics"]["policy/valid_tokens"] == 7
    probe = json.loads((tmp_path / "fixed_probe/completed-000000.json").read_text())
    assert probe["entropy"] == list(range(64))


def example():
    ratio = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 0.25, 3.0])
    data = {
        "ratio": ratio,
        "weight": ratio.clamp(0.5, 2.0),
        "advantage": torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 2.0]),
        "logp": ratio.log(),
        "logq": torch.zeros(7),
        "entropy": torch.arange(1.0, 8.0),
        "sample_id": torch.tensor([10, 10, 20, 20, 30, 30, 40]),
        "position": torch.tensor([0, 1, 0, 1, 0, 1, 0]),
        "token_id": torch.arange(7),
    }
    samples = {
        10: dict(valid_tokens=2, advantage=1.0, first_valid_positions=[0, 1], theme="a", scenario="s"),
        20: dict(valid_tokens=2, advantage=-1.0, first_valid_positions=[0, 1], theme="a", scenario="s"),
        30: dict(valid_tokens=2, advantage=0.0, first_valid_positions=[0, 1], theme="b", scenario="s"),
        40: dict(valid_tokens=1, advantage=2.0, first_valid_positions=[0], theme="b", scenario="s"),
        50: dict(valid_tokens=0, advantage=0.0, first_valid_positions=[], theme="b", scenario="s"),
    }
    return data, samples


def write_shards(root, *, duplicate=False):
    data, samples = example()
    binding = {"run": "test", "completed_update": 1}
    partitions = ([10, 20], [30], [40, 50], [])
    if duplicate:
        partitions = ([10, 20], [30], [40, 50], [10])
    for rank, ids in enumerate(partitions):
        chosen = torch.tensor([int(i) in ids for i in data["sample_id"]])
        path = root / f"dp-{rank}.bin"
        frames = 0
        if chosen.any():
            part = {k: v[chosen] for k, v in data.items()}
            part["mask"] = torch.ones(int(chosen.sum()), dtype=torch.bool)
            metrics.append_frame(path, {"binding": binding, "dp_rank": rank, "data": part})
            frames = 1
        else:
            path.touch()
        metrics.write_json(
            path.with_suffix(".complete.json"),
            {
                "binding": binding,
                "frames": frames,
                "valid_tokens": int(chosen.sum()),
                "samples": {i: samples[i] for i in ids},
                "outcome": "NORMAL",
                "sha256": metrics.sha256(path),
            },
        )
    return data, samples, binding


def test_dp4_unequal_empty_rank_pooling(tmp_path):
    data, samples, binding = write_shards(tmp_path)
    loaded, metadata = metrics.load_shards(tmp_path, dp_size=4, expected_samples=samples, binding=binding)
    result = metrics.summarize_tokens(loaded, metadata)
    assert result["policy/entropy/token_mean"] == 4.0
    assert result["policy/entropy/equal_nonempty_response_mean"] == 4.375
    assert result["policy/empty_responses"] == 1
    assert result["policy/all/tis_floor_tokens"] == 2
    assert result["policy/all/tis_cap_tokens"] == 2
    assert result["policy/adv_positive/tis_floor_conditional_fraction"] == pytest.approx(1 / 3)
    assert result["policy/adv_positive/tis_floor_global_fraction"] == pytest.approx(1 / 7)
    assert result["policy/adv_zero/pg_coefficient_abs_mean"] == 0
    assert result["policy/adv_zero/tokens"] == 2
    for name, weights in (("raw", data["ratio"]), ("applied", data["weight"])):
        assert result[f"policy/ess/{name}"] == pytest.approx(
            float(weights.double().sum().square() / weights.double().square().sum())
        )


@pytest.mark.parametrize("damage", ["duplicate", "truncated", "changed", "wrong_identity", "non_normal"])
def test_incomplete_or_mixed_dp_data_rejected(tmp_path, damage):
    _, samples, binding = write_shards(tmp_path, duplicate=damage == "duplicate")
    if damage in ("truncated", "changed"):
        path = tmp_path / "dp-0.bin"
        path.write_bytes(path.read_bytes()[:-3] if damage == "truncated" else path.read_bytes() + b"x")
    if damage in ("wrong_identity", "non_normal"):
        path = tmp_path / "dp-0.complete.json"
        seal = json.loads(path.read_text())
        if damage == "wrong_identity":
            seal["binding"] = {"run": "other"}
        else:
            seal["outcome"] = "SKIPPED"
        path.write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        metrics.load_shards(tmp_path, dp_size=4, expected_samples=samples, binding=binding)


def test_native_tis_exact_boundaries_and_detached_weights():
    ratio = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
    score = ratio.log().requires_grad_()
    loss, mask, native = vanilla_tis_function(
        SimpleNamespace(tis_clip_low=0.5, tis_clip=2.0),
        pg_loss=-score,
        train_log_probs=[score.detach()],
        rollout_log_probs=[torch.zeros(5)],
        loss_masks=[torch.ones(5)],
    )
    (grad,) = torch.autograd.grad(loss.sum(), score)
    torch.testing.assert_close(grad, -ratio.clamp(0.5, 2.0))
    assert native["tis_floorfrac"].tolist() == [1, 0, 0, 0, 0]
    assert native["tis_capfrac"].tolist() == [0, 0, 0, 0, 1]
    assert mask[0].sum() == 5


def test_entropy_full_true_vocabulary_and_sparse_gradient_oracle():
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(7, 9, generator=generator, requires_grad=True)
    logits.data[:, 7:] = 100  # padded vocabulary must not dominate entropy
    logp = logits[:, :7].log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    entropy, mass, _, _ = distribution_statistics(logits, vocab_size=7)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(mass, torch.ones(7))
    sampled = torch.arange(7)
    advantage = torch.tensor([1.0, -1.0, 0.0, 1.0, -2.0, 0.0, 3.0])
    ratio = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 0.25, 3.0])
    weight = ratio.clamp(0.5, 2.0)
    loss = (-advantage * weight * logits.log_softmax(-1)[torch.arange(7), sampled]).sum() / 19
    (derivative,) = torch.autograd.grad(loss, logits, retain_graph=True)
    (entropy_derivative,) = torch.autograd.grad(expected_entropy.sum(), logits)
    packet = component_statistics(
        logits, sampled, advantage, ratio, weight, vocab_size=7, temperature=1.0, denominator=19
    )
    torch.testing.assert_close(packet["norm2"][:, 0], derivative.square().sum(-1))
    torch.testing.assert_close(packet["entropy_direction"][:, 0], -(derivative * entropy_derivative).sum(-1))
    assert packet["norm2"][advantage == 0, 0].count_nonzero() == 0
    assert not packet["norm2"].requires_grad


def test_audit_selection_is_global_balanced_and_deterministic():
    samples = {i: dict(advantage=1.0 if i % 2 else -1.0, first_valid_positions=list(range(20))) for i in range(8)}
    selected = metrics.select_positions(samples)
    assert len(selected) == len(set(selected)) == 32
    assert sum(samples[i]["advantage"] > 0 for i, _ in selected) == 16
    assert selected == metrics.select_positions(dict(reversed(list(samples.items()))))


def test_invalid_native_weight_not_accepted():
    data, samples = example()
    data["weight"][0] = 0.0  # MIS rejection must never be accepted as TIS
    with pytest.raises(AssertionError):
        metrics.summarize_tokens(data, samples)


def test_tracking_failure_preserves_numeric_spool(tmp_path, monkeypatch):
    from miles.utils.tracking_utils import tracking

    monkeypatch.setattr(tracking, "define_step_key_metric_group", lambda *args: None)

    def unavailable(*args, **kwargs):
        raise ConnectionError("transport unavailable")

    monkeypatch.setattr(tracking, "log", unavailable)
    metrics.write_json(tmp_path / "numeric.json", {"policy/step": 1, "policy/entropy/token_mean": 0.2})
    metrics.publish_metrics(None, {"policy/step": 1}, tmp_path / "tracking.json")
    assert json.loads((tmp_path / "tracking.json").read_text())["state"] == "DEGRADED_DURABLE_METRICS_PRESERVED"
    assert json.loads((tmp_path / "numeric.json").read_text())["policy/entropy/token_mean"] == 0.2


def test_reinforce_argument_binding():
    args = SimpleNamespace(
        loss_type="reinforce",
        advantage_estimator="grpo",
        skip_actor_forward_only=True,
        use_rollout_logprobs=False,
        use_tis=True,
        tis_clip_low=0.5,
        tis_clip=2.0,
        entropy_coef=0.0,
        kl_coef=0.0,
        use_kl_loss=False,
        policy_diagnostics_dir="/private",
        calculate_per_token_loss=True,
        qkv_format="thd",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        policy_audit_updates=[1, 5, 10],
        policy_probe_updates=[0, 1, 5, 10],
        policy_fixed_probe="/private/probe.pt",
        policy_fixed_probe_sha256="sealed",
    )
    validate_policy_diagnostics_args(args)
    args.use_rollout_logprobs = True
    with pytest.raises(ValueError):
        validate_policy_diagnostics_args(args)
