import os
from pathlib import Path


def resolve_experiment_run_id(args, receipt_path: str) -> str:
    """Resolve a durable experiment identity without requiring a parser field."""
    run_id = getattr(args, "run_id", None) or os.environ.get("MILES_EXPERIMENT_RUN_ID")
    if run_id:
        return str(run_id)

    inferred = Path(receipt_path).parent.name
    if not inferred:
        raise RuntimeError("unable to infer experiment run id from receipt path")
    return inferred
