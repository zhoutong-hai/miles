"""One-scalar entropy feedback for Megatron actor training.

The coefficient used for an optimizer step is changed only after that step
succeeds.  Megatron checkpoints serialize ``args`` as common state, so the
next-step coefficient is part of the model/optimizer checkpoint rather than a
separate, potentially stale file.
"""

from __future__ import annotations

import math
from argparse import Namespace
from typing import Any


def validate_adaptive_entropy_args(args: Namespace) -> None:
    """Reject incomplete or incompatible opt-in configurations."""
    target = getattr(args, "adaptive_entropy_target", None)
    rate = getattr(args, "adaptive_entropy_rate", None)
    maximum = getattr(args, "adaptive_entropy_max_coef", None)
    if target is None:
        if rate is not None or maximum is not None:
            raise ValueError("Adaptive entropy rate/max require --adaptive-entropy-target")
        return

    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (target, rate, maximum)):
        raise ValueError("Adaptive entropy target, rate, and max must be finite numbers")
    if target < 0 or rate <= 0 or maximum <= 0:
        raise ValueError("Adaptive entropy requires target >= 0, rate > 0, max > 0")
    initial = args.entropy_coef
    if not isinstance(initial, (int, float)) or not math.isfinite(initial) or not 0 <= initial <= maximum:
        raise ValueError("Initial entropy coefficient must be finite and within [0, adaptive max]")
    if not args.observe_training_entropy or not args.calculate_per_token_loss:
        raise ValueError("Adaptive entropy requires observed, token-mean training entropy")
    if args.train_backend != "megatron":
        raise ValueError("Adaptive entropy currently supports only Megatron training")


def next_entropy_coefficient(coefficient: float, entropy: float, target: float, rate: float, maximum: float) -> float:
    """Apply bounded integral feedback using the completed step's entropy."""
    if not all(math.isfinite(x) for x in (coefficient, entropy, target, rate, maximum)):
        raise ValueError("Adaptive entropy controller received a non-finite value")
    if entropy < 0 or target < 0 or rate <= 0 or maximum <= 0 or not 0 <= coefficient <= maximum:
        raise ValueError("Adaptive entropy controller received an out-of-range value")
    return min(max(coefficient + rate * (target - entropy), 0.0), maximum)


def saved_entropy_coefficient(state_dict: dict[str, Any], args: Namespace, iteration: int) -> float:
    """Read the next-step coefficient from a matching full Megatron checkpoint."""
    if state_dict.get("iteration") != iteration:
        raise ValueError("Adaptive entropy checkpoint iteration does not match loaded model state")
    saved_args = state_dict.get("args")
    if saved_args is None:
        raise ValueError("Adaptive entropy checkpoint is missing saved arguments")
    for name in ("adaptive_entropy_target", "adaptive_entropy_rate", "adaptive_entropy_max_coef"):
        if getattr(saved_args, name, None) != getattr(args, name):
            raise ValueError(f"Adaptive entropy checkpoint {name} does not match the run configuration")
    coefficient = getattr(saved_args, "entropy_coef", None)
    if not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient):
        raise ValueError("Adaptive entropy checkpoint has no finite next-step coefficient")
    if not 0 <= coefficient <= args.adaptive_entropy_max_coef:
        raise ValueError("Adaptive entropy checkpoint coefficient is outside the declared bounds")
    return float(coefficient)
