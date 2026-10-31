"""Attempt-bound publication evidence paths; no model/transport changes."""

import os
from pathlib import Path
import re

RUN_ID = "e012-i001-baseline-validation-40"
PREFIX = "/hai/zhoutong/miles_rl/aws/e012/i001"


def publication_root_matches(value):
    attempt = os.environ.get("MILES_EXPERIMENT_ATTEMPT_ID", "")
    if os.environ.get("MILES_EXPERIMENT_RUN_ID") != RUN_ID:
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", attempt):
        return False
    expected = f"{PREFIX}/{attempt}/restricted/publication-profile"
    if value != expected:
        return False
    # Do not allow an evidence path to escape its attempt through symlinks.
    path = Path(value)
    return not any(parent.is_symlink() for parent in (path, *path.parents))
