"""Independent scalar/gradient checks for complete-batch prompt means across DP partitions."""

import math

import numpy as np
import pytest
import torch

from miles.utils.reinforce import fixed_mis_loss, prompt_factors

BOUNDS = (0.2, 5.0, 0.5, 2.0)


def test_dp2_microbatches_match_original_prompt_reference():
    groups = [7, 7, 8, 8, 9, 9]
    masks = [[1, 1], [1], [1, 0, 1], [1, 1, 1], [1], [1, 1]]
    advantages = [1.0, -1.0, 0.0, 0.75, -0.5, 0.0]
    ratios = [0.1, 1.0, 4.0, 5.2, 0.6, 1.0]
    scales, totals = prompt_factors(groups, masks, groups=3, siblings=2)
    n = sum(sum(m) for m in masks)
    theta = torch.tensor(0.37, requires_grad=True)
    feature = [torch.arange(1, len(m) + 1, dtype=torch.float32) for m in masks]
    # Siblings deliberately land on different DP replicas and microbatches.
    replica_losses = []
    for partition in [[0, 2, 5], [1, 3, 4]]:
        terms = []
        for i in partition:
            p = theta * feature[i]
            q = p.detach() - math.log(ratios[i])
            loss, _, _ = fixed_mis_loss(p, q, torch.full_like(p, advantages[i]), torch.tensor(masks[i]), torch.full_like(p, scales[i]), BOUNDS)
            terms.append(loss.sum())
        # Native DP loss scaling, followed by DP gradient averaging.
        replica_losses.append(sum(terms) * 2 / n)
    actual = sum(replica_losses) / 2
    actual.backward()
    expected, gradient = 0.0, 0.0
    group_tokens = {g: sum(sum(m) for gid, m in zip(groups, masks, strict=True) if gid == g) for g in set(groups)}
    for i, mask in enumerate(masks):
        low, high = BOUNDS[:2] if advantages[i] > 0 else BOUNDS[2:]
        w = ratios[i] if low <= ratios[i] <= high else 0.0
        for j, valid in enumerate(mask):
            if valid:
                coefficient = -advantages[i] * w * (j + 1) / (3 * group_tokens[groups[i]])
                expected += float(theta.detach()) * coefficient
                gradient += coefficient
    assert actual.item() == pytest.approx(expected, abs=2e-7)
    assert theta.grad.item() == pytest.approx(gradient, abs=2e-7)
    assert totals == [3, 3, 5, 5, 3, 3]


@pytest.mark.parametrize("advantage,bounds", [(1.0, (0.2, 5.0)), (-1.0, (0.5, 2.0)), (0.0, (0.5, 2.0))])
def test_inclusive_fp32_bounds_and_detached_coefficients(advantage, bounds):
    low, high = bounds
    q = -torch.tensor([math.log(x) for x in [low, 1.0, high]], dtype=torch.float32, requires_grad=True)
    q.retain_grad()
    p = torch.zeros(3, requires_grad=True)
    a = torch.full((3,), advantage, requires_grad=True)
    loss, weight, ratio = fixed_mis_loss(p, q, a, torch.ones(3), torch.ones(3), BOUNDS)
    expected_ratio = np.exp(-q.detach().numpy(), dtype=np.float32)
    # exp(log(FP32(.2))) can be one ULP below FP32(.2); compare the actual
    # returned ratio, without a tolerance that changes the scientific interval.
    expected_keep = (expected_ratio >= np.float32(low)) & (expected_ratio <= np.float32(high))
    np.testing.assert_allclose(weight.numpy(), np.where(expected_keep, expected_ratio, 0), rtol=1e-7)
    np.testing.assert_allclose(ratio.numpy(), expected_ratio, rtol=1e-7)
    # Exercise exact equality to the actual FP32 exp result as well.
    exact_bounds = (ratio[0].item(), ratio[-1].item())
    _, exact_weight, _ = fixed_mis_loss(p, q, a, torch.ones(3), torch.ones(3), (*exact_bounds, *exact_bounds))
    assert torch.equal(exact_weight, ratio)
    loss.sum().backward()
    torch.testing.assert_close(p.grad, -advantage * weight)
    assert q.grad is None and a.grad is None


@pytest.mark.parametrize("p,q,a", [(float("nan"), 0.0, 0.0), (0.0, float("inf"), -1.0), (1000.0, 0.0, -1.0), (-1000.0, 0.0, 0.0)])
def test_invalid_valid_ratios_rejected_before_mis_even_for_zero_advantage(p, q, a):
    with pytest.raises(ValueError):
        fixed_mis_loss(torch.tensor([p]), torch.tensor([q]), torch.tensor([a]), torch.ones(1), torch.ones(1), BOUNDS)


def test_padding_nan_is_not_a_valid_probability_or_denominator():
    p = torch.tensor([-0.5, float("nan")], requires_grad=True)
    loss, weight, ratio = fixed_mis_loss(p, torch.tensor([-0.5, float("nan")]), torch.tensor([1.0, float("nan")]), torch.tensor([1, 0]), torch.tensor([1.0, float("nan")]), BOUNDS)
    assert loss.tolist() == [0.5, 0.0] and weight.tolist() == [1.0, 0.0] and ratio.tolist() == [1.0, 0.0]
    loss.sum().backward()
    assert p.grad.tolist() == [-1.0, 0.0]


def test_incomplete_siblings_and_empty_action_masks_fail_closed():
    with pytest.raises(ValueError):
        prompt_factors([1, 2], [[1], [1]], groups=1, siblings=2)
    with pytest.raises(ValueError):
        prompt_factors([1, 1], [[1], [0]], groups=1, siblings=2)
