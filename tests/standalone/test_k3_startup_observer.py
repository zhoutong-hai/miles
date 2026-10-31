"""Frozen router and adapter-only startup acceptance rejects real mutations."""
import pytest
import torch
from miles.backends.training_utils.loss_hub.k3_startup import router_seals, validate_trainable


def model():
    root = torch.nn.Module()
    root.block = torch.nn.Module()
    root.block.router = torch.nn.Linear(3, 2)
    root.block.router.requires_grad_(False)
    root.block.router.register_buffer("local_tokens_per_expert", torch.zeros(2), persistent=False)
    root.block.lora_A = torch.nn.Linear(3, 2, bias=False)
    return root


def test_router_seal_covers_loaded_bias_and_detects_mutation():
    value = model()
    before = router_seals([value])
    assert len(before) == 2
    value.block.router.local_tokens_per_expert.add_(5)
    assert router_seals([value]) == before
    with torch.no_grad():
        value.block.router.bias.add_(.1)
    assert router_seals([value]) != before
    value.block.router.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        router_seals([value])


def test_trainable_policy_rejects_accidentally_unfrozen_base():
    value = model()
    assert validate_trainable([value])["trainable_tensors"] == 1
    value.block.base = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="only LoRA"):
        validate_trainable([value])
