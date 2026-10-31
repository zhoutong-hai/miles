"""Publication ownership, finite version bounds and durable tracking mechanics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location("isolated_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spool_survives_transport_failure_and_excludes_text(tmp_path, monkeypatch):
    module = load("miles/utils/tracking_utils/durable_metrics.py")
    monkeypatch.setenv("MILES_METRICS_SPOOL_ROOT", str(tmp_path))

    def fail(*args, **kwargs):
        assert list(tmp_path.glob("metrics-*.json")), "spool must precede delivery"
        raise ConnectionError("do not persist transport error text")

    module.log_durable({"eval/step": 10, "eval/scenario": .8, "raw_text": "private"},
                       step_key="eval/step", deliver=True, writer=fail)
    rows = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(rows) == 2
    assert all(row["metrics"] == {"eval/step": 10., "eval/scenario": .8} for row in rows)
    assert "private" not in "".join(path.read_text() for path in tmp_path.glob("*.json"))
    with pytest.raises(ValueError):
        module.log_durable({"eval/step": float("nan")}, step_key="eval/step", deliver=False, writer=fail)


def test_online_identity_must_match_selected_project(tmp_path, monkeypatch):
    module = load("miles/utils/tracking_utils/durable_metrics.py")
    monkeypatch.setenv("MILES_METRICS_SPOOL_ROOT", str(tmp_path))
    monkeypatch.setenv("MILES_EXPERIMENT_RUN_ID", "e012-i001-baseline-validation-40")
    args = SimpleNamespace(wandb_run_id="k3-i001-r1", wandb_project="zhoutong-miles-k3",
                           wandb_team="hippocraticai", wandb_mode="online")
    run = SimpleNamespace(id=args.wandb_run_id, project="miles-rl", entity=args.wandb_team)
    with pytest.raises(ValueError):
        module.record_online_identity(run, args)
    run.project = args.wandb_project
    module.record_online_identity(run, args)
    assert len(list(tmp_path.glob("online-identity-*.json"))) == 1


def test_publication_path_is_exact_attempt_owned(monkeypatch):
    module = load("miles/backends/training_utils/k3_publication/publication_paths.py")
    monkeypatch.setenv("MILES_EXPERIMENT_RUN_ID", module.RUN_ID)
    monkeypatch.setenv("MILES_EXPERIMENT_ATTEMPT_ID", "r1")
    assert module.publication_root_matches(module.PREFIX + "/r1/restricted/publication-profile")
    assert not module.publication_root_matches(module.PREFIX + "/r2/restricted/publication-profile")
    monkeypatch.setenv("MILES_EXPERIMENT_ATTEMPT_ID", "../r1")
    assert not module.publication_root_matches(module.PREFIX + "/../r1/restricted/publication-profile")


def test_structural_publication_has_exact_finite_version_bound(monkeypatch):
    from miles.backends.training_utils.k3_publication.publication import publication_mode

    monkeypatch.setenv("MILES_K3_PUBLICATION_MODE", "structural")
    assert all(publication_mode(version) == "structural" for version in range(1, 42))
    for version in (0, 42, True, 1.):
        with pytest.raises(ValueError):
            publication_mode(version)


@pytest.mark.parametrize("interval,accepted", [(10, True), (20, True), (5, False), (40, False)])
def test_full_policy_validator_admits_selected_save_cadence(interval, accepted):
    from miles.utils.policy_diagnostics import validate_policy_diagnostics_args

    args = SimpleNamespace(loss_type="reinforce", advantage_estimator="grpo", skip_actor_forward_only=True,
                           use_rollout_logprobs=False, use_tis=True, tis_clip_low=.5, tis_clip=2.,
                           entropy_coef=0., kl_coef=0., use_kl_loss=False,
                           policy_diagnostics_dir="/private/diagnostics", calculate_per_token_loss=True,
                           qkv_format="thd", rollout_top_p=1., rollout_top_k=-1, rollout_temperature=1.,
                           policy_audit_updates=[1, 5, 10, 20, 30, 40],
                           policy_probe_updates=[0, 1, 5, 10, 20, 30, 40],
                           policy_fixed_probe="/sealed/probe", policy_fixed_probe_sha256="sealed",
                           save_full_lora_state=True, num_rollout=40, save_interval=interval, save="/private/save",
                           no_save_optim=False, no_save_rng=False, async_save=False)
    if accepted:
        validate_policy_diagnostics_args(args)
    else:
        with pytest.raises(ValueError, match="ten/twenty-update"):
            validate_policy_diagnostics_args(args)
