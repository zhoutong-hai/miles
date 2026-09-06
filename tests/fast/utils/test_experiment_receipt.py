from argparse import Namespace

import pytest

from miles.utils.audit_utils.experiment_receipt import resolve_experiment_run_id


def test_resolve_experiment_run_id_does_not_require_parser_field(monkeypatch):
    monkeypatch.delenv("MILES_EXPERIMENT_RUN_ID", raising=False)

    assert (
        resolve_experiment_run_id(Namespace(), "/shared/runs/e006-i001-r21/startup-finite-probe.json")
        == "e006-i001-r21"
    )


def test_resolve_experiment_run_id_prefers_explicit_identity(monkeypatch):
    monkeypatch.setenv("MILES_EXPERIMENT_RUN_ID", "e006-i001-explicit")

    assert (
        resolve_experiment_run_id(Namespace(run_id="from-args"), "/shared/runs/from-path/receipt.json")
        == "from-args"
    )
    assert resolve_experiment_run_id(Namespace(), "/shared/runs/from-path/receipt.json") == "e006-i001-explicit"


def test_resolve_experiment_run_id_fails_closed_when_identity_is_unavailable(monkeypatch):
    monkeypatch.delenv("MILES_EXPERIMENT_RUN_ID", raising=False)

    with pytest.raises(RuntimeError, match="unable to infer experiment run id"):
        resolve_experiment_run_id(Namespace(), "receipt.json")
