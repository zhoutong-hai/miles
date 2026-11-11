import torch
import torch.nn as nn


def kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cp_context=None,
) -> torch.Tensor:
    """fla delta-rule core; boundaries travel as ``cu_seqlens`` without CP or ``cp_context`` under CP, never both."""
    from fla.ops.kda import chunk_kda

    boundaries = {"cp_context": cp_context} if cp_context is not None else {"cu_seqlens": cu_seqlens}
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        **boundaries,
    )
    return output


def _situ_and_mul_reference(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)


_SITU_CHUNK_ELEMENTS = 16_777_216


class _ChunkedSiTU(torch.autograd.Function):
    """Bound FP32 pointwise temporaries while retaining native SiTU math."""

    @staticmethod
    def forward(ctx, x, beta, linear_beta):
        ctx.save_for_backward(x)
        ctx.beta, ctx.linear_beta = beta, linear_beta
        width = x.shape[-1]
        rows = max(1, _SITU_CHUNK_ELEMENTS // width)
        flat = x.view(-1, width)
        output = torch.empty((*x.shape[:-1], width // 2), dtype=x.dtype, device=x.device)
        target = output.view(-1, width // 2)
        for start in range(0, flat.shape[0], rows):
            target[start : start + rows].copy_(
                _situ_and_mul_reference(flat[start : start + rows], beta, linear_beta)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        width = x.shape[-1]
        rows = max(1, _SITU_CHUNK_ELEMENTS // width)
        flat = x.view(-1, width)
        incoming = grad_output.reshape(-1, width // 2)
        if torch.is_grad_enabled():
            # Higher derivatives are not the bounded training path; retain the native whole graph.
            with torch.enable_grad():
                output = _situ_and_mul_reference(x, ctx.beta, ctx.linear_beta)
                (gradient,) = torch.autograd.grad(output, x, grad_output, create_graph=True)
            return gradient, None, None
        gradient = torch.empty_like(x)
        target = gradient.view(-1, width)
        for start in range(0, flat.shape[0], rows):
            with torch.enable_grad():
                chunk = flat[start : start + rows].detach().requires_grad_(True)
                output = _situ_and_mul_reference(chunk, ctx.beta, ctx.linear_beta)
                (value,) = torch.autograd.grad(output, chunk, incoming[start : start + rows])
            target[start : start + rows].copy_(value)
        return gradient, None, None


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    # Keep all tokens and the original FP32 expression; only tile large contiguous inputs.
    if (
        x.numel() <= _SITU_CHUNK_ELEMENTS
        or not x.is_contiguous()
        or x.shape[-1] % 2
        or not isinstance(beta, (int, float))
        or not isinstance(linear_beta, (int, float))
    ):
        return _situ_and_mul_reference(x, beta, linear_beta)
    return _ChunkedSiTU.apply(x, beta, linear_beta)


class KimiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return output_norm(mixed)
