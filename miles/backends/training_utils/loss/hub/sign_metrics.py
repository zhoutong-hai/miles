"""Additive sign/zero and entropy statistics; no token dumps or extra model forwards."""

import torch


def advantage_sign_metrics(advantage, active, entropy, weight, ratio, *, low, high, reduce):
    """All masses use the original token normalizer, including zero advantages.

    Conditional means/ESS/histograms must be formed AFTER summing native
    frames: divide a sign's mass by its token fraction, not by minibatch size.
    This representation composes across DP, CP and unequal microbatches.
    """
    if not torch.isfinite(advantage[active]).all() or not torch.isfinite(entropy[active]).all():
        raise ValueError("Nonfinite original advantage or training entropy")
    output = {"sign/original_valid_fraction": reduce(active.float()).detach()}
    for name, chosen in (
        ("positive", active & (advantage > 0)),
        ("negative", active & (advantage < 0)),
        ("zero", active & (advantage == 0)),
    ):
        prefix = f"sign/{name}"
        values = {
            "token_fraction": chosen.float(),
            "entropy_mass": torch.where(chosen, entropy.detach(), 0.0),
            "tis_weight_mass": torch.where(chosen, weight.detach(), 0.0),
            "tis_weight_squared_mass": torch.where(chosen, weight.detach().square(), 0.0),
            "tis_below_floor_fraction": (chosen & (ratio < low)).float(),
            "tis_floor_to_one_fraction": (chosen & (ratio >= low) & (ratio < 1)).float(),
            "tis_one_to_cap_fraction": (chosen & (ratio >= 1) & (ratio <= high)).float(),
            "tis_above_cap_fraction": (chosen & (ratio > high)).float(),
        }
        for key, value in values.items():
            output[f"{prefix}/{key}"] = reduce(value).detach()
    return output
