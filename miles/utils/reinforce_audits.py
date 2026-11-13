"""Bounded CUDA/Megatron diagnostics, independent of the applied REINFORCE graph."""

import json

import torch


def optimizer_model_parameters(optimizer):
    """Inspect the native optimizer's model groups, not its FP32 master shards."""
    if hasattr(optimizer, "chained_optimizers"):
        return [p for child in optimizer.chained_optimizers for p in optimizer_model_parameters(child)]
    if hasattr(optimizer, "float16_groups"):
        groups = [*optimizer.float16_groups, *optimizer.fp32_from_fp32_groups]
        return [p for group in groups for p in group]
    if hasattr(optimizer, "model_float16_groups"):
        groups = [*optimizer.model_float16_groups, *optimizer.model_fp32_groups]
        return [p for group in groups for p in group]
    return [p for group in optimizer.param_groups for p in group["params"]]


def attach_routers(model):
    """Counts describe executed routing, including padding and recomputation."""
    counts, handles = {}, []
    for chunk_id, chunk in enumerate(model):
        for name, module in chunk.named_modules():
            if not any(base.__name__ == "TopKRouter" for base in type(module).__mro__):
                continue
            key = f"chunk{chunk_id}/{name}"

            def hook(_module, _inputs, output, key=key):
                with torch.no_grad():
                    _, routing_map = output
                    value = routing_map.detach().sum(0, dtype=torch.int64)
                    if key not in counts:
                        counts[key] = value
                    else:
                        counts[key].add_(value)

            handles.append(module.register_forward_hook(hook))
    return counts, handles


def sparse_output_audit(context, args, batch, layout, logits, advantage, factors, weight, ratio):
    """Capture native numerator dL/dz on sparse coordinates; no parameter-gradient claim.

    Counterfactuals below are analytic raw/TIS output-logit VJPs at the SAME
    logits. They are not additional native backward passes or future updates.
    """
    if context.get("sparse_registered"):
        return
    from miles.backends.training_utils.loss.hub.logit_processors import _iter_response_chunks
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    chunks = _iter_response_chunks(logits, args=args, unconcat_tokens=batch["unconcat_tokens"], total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"], max_seq_lens=batch.get("max_seq_lens"), include_response_indices=False)
    references = []
    offset = 0
    with torch.no_grad():
        for i, (chunk, tokens, _) in enumerate(chunks):
            if len(references) < 4 and chunk.size(0):
                valid = torch.nonzero(batch["loss_masks"][i]).flatten()
                if valid.numel():
                    j = int(valid[0])
                    row = chunk[j].float()
                    maximum = row.max()
                    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=state.tp.group)
                    exponentials = (row - maximum).exp()
                    denominator = exponentials.sum()
                    torch.distributed.all_reduce(denominator, group=state.tp.group)
                    probabilities = exponentials / denominator
                    target = int(tokens[j])
                    local_target = target - state.tp.rank * row.numel()
                    columns = row.topk(min(8, row.numel())).indices.tolist()
                    if 0 <= local_target < row.numel() and local_target not in columns:
                        columns.append(local_target)
                    delta = torch.tensor([float(k == local_target) for k in columns], device=row.device)
                    vector = delta - probabilities[columns]
                    a, factor = advantage[offset + j], factors[offset + j]
                    w, r = weight[offset + j], ratio[offset + j]
                    base = -a * factor * vector
                    flat_row = (chunk.storage_offset() - logits.storage_offset()) // logits.size(-1) + j
                    references.append({"sample": int(batch["sample_indices"][i]), "response_position": j, "flat_row": flat_row, "columns": columns, "fixed_MIS_numerator_vjp": (base * w).tolist(), "raw_IS_analytic_numerator_vjp": (base * r).tolist(), "TIS_cap2_analytic_numerator_vjp": (base * r.clamp_max(2)).tolist()})
            offset += chunk.size(0)
    if not references:
        raise ValueError("Sparse audit has no original valid response token")
    context["sparse_registered"] = True
    path = context["root"] / f"rank-{context['rank']}-sparse-output-logit-vjp.json"

    def capture(gradient):
        flat = gradient.reshape(-1, gradient.size(-1))
        for row in references:
            actual = flat[row["flat_row"], row["columns"]].detach().float().cpu()
            expected = torch.tensor(row["fixed_MIS_numerator_vjp"])
            if not torch.isfinite(actual).all():
                raise ValueError("Nonfinite native sparse output gradient")
            row["actual_native_numerator_vjp"] = actual.tolist()
            row["max_abs_error"] = float((actual - expected).abs().max())
        value = {"rank": context["rank"], "completed_update": context["rollout"] + 1, "scope": "sparse pre-finalize numerator output-logit gradients; no parameter causality", "rows": references}
        with path.open("x") as stream:
            json.dump(value, stream, allow_nan=False)
        return gradient

    logits.register_hook(capture)
