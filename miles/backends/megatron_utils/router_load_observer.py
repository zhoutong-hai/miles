"""Opt-in, aggregate-only router observations for the held E010 learning screen.

Tap the existing stacked TP/CP/DP reduction, then assemble pipeline-stage
summaries once per accepted update. Never change routing, counts or bias values.
"""

import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


def summarize_counts(counts, rank_experts):
    """Summarize actual reduced dispatch counts using validated expert ownership."""
    counts = counts.to(dtype=torch.float64, device="cpu")
    flat = [expert for indices in rank_experts for expert in indices]
    if sorted(flat) != list(range(counts.numel())):
        raise ValueError("Expert ownership must partition the histogram exactly once")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid reduced router counts")
    mean = counts.mean().item()
    rank_load = torch.tensor([counts[indices].sum().item() for indices in rank_experts], dtype=torch.float64)
    return {
        "expert_max_mean": counts.max().item() / mean if mean else 0.0,
        "expert_cv": counts.std(correction=0).item() / mean if mean else 0.0,
        "unused_expert_fraction": (counts == 0).double().mean().item(),
        "ep_rank_max_mean": rank_load.max().item() / rank_load.mean().item() if mean else 0.0,
    }


def require_frozen_router(router, optimizer):
    params = [router.weight] + ([router.bias] if router.bias is not None else [])
    owned = {id(param) for group in optimizer.param_groups for param in group["params"]}
    for param in params:
        if param.requires_grad or id(param) in owned or id(getattr(param, "main_param", None)) in owned:
            raise ValueError("Constructed gate is trainable or belongs to the optimizer")
    return tuple((param, param.detach().cpu().clone()) for param in [*params, router.expert_bias])


def check_unchanged(snapshot):
    return all(torch.equal(param.detach().cpu(), before) for param, before in snapshot)


class RouterLoadObserver:
    """Own the temporary capture and the one-time frozen-state/count checks."""

    def __init__(self, args, config, model, optimizer):
        # Optional heavyweight imports only when the explicitly selected observer is enabled.
        from megatron.core import mpu
        from miles.backends.training_utils.parallel import get_parallel_state

        if not (config.moe_router_freeze_gate and config.moe_router_enable_expert_bias
                and config.moe_router_bias_update_rate == 0 and config.moe_aux_loss_coeff == 0
                and not config.freeze_e_score_correction_bias
                and args.recompute_method == "uniform" and args.recompute_num_layers == 1
                and args.qkv_format == "thd" and args.allgather_cp and args.sequence_parallel):
            raise ValueError("Router observation is qualified only for the selected frozen-gate packed baseline")
        self.args, self.config, self.mpu = args, config, mpu
        state = get_parallel_state()
        self.representative = state.tp.rank == 0 and state.effective_dp_cp.rank == 0
        self.cp_size = state.cp.size
        self.routers = []
        self.rank_experts = None
        for chunk in model:
            for module in chunk.modules():
                if hasattr(module, "local_expert_indices") and hasattr(module, "router"):
                    router = module.router
                    ep_size, ep_rank = dist.get_world_size(module.ep_group), dist.get_rank(module.ep_group)
                    width = router.num_experts // ep_size
                    mapping = [list(range(rank * width, (rank + 1) * width)) for rank in range(ep_size)]
                    if module.local_expert_indices != mapping[ep_rank]:
                        raise ValueError("Constructed expert-to-rank mapping differs")
                    if self.rank_experts is not None and self.rank_experts != mapping:
                        raise ValueError("Inconsistent expert ownership across layers")
                    self.rank_experts = mapping
                    self.routers.append(router)
        self.snapshot = tuple(item for router in self.routers for item in require_frozen_router(router, optimizer))
        self.finalize = importlib.import_module("megatron.core.distributed.finalize_model_grads")
        self.original = self.finalize.get_updated_expert_bias
        self.first_check = True
        self.accounting_valid = None
        self.pending = None
        self.real_positions = self.packed_positions = 0

    @contextmanager
    def step(self):
        self.pending = None
        self.real_positions = self.packed_positions = 0
        self.finalize.get_updated_expert_bias = self._capture
        try:
            yield
        finally:
            self.finalize.get_updated_expert_bias = self.original

    def record_batch(self, batch):
        # One outer forward_step per microbatch, not each recomputed transformer.
        self.real_positions += sum(batch["total_lengths"])
        self.packed_positions += batch["tokens"].numel() * self.cp_size

    def _capture(self, tokens_per_expert, expert_bias, expert_bias_update_rate):
        result = self.original(tokens_per_expert, expert_bias, expert_bias_update_rate)
        if self.representative:
            # Original all_reduce has already mutated the stacked tensor; copy before reset.
            self.pending = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64)
        return result

    def finish(self, rollout_id, step_id, num_steps):
        if self.first_check:
            if not check_unchanged(self.snapshot):
                raise RuntimeError("Frozen router gate or checkpoint-loaded correction bias changed")
            self.snapshot = ()
        if not self.representative:
            self.first_check = False
            return
        if self.pending is None or self.pending.shape[0] != len(self.routers):
            raise RuntimeError("Missing already-reduced router histogram")
        completed = rollout_id * num_steps + step_id + 1
        snapshot_due = step_id == num_steps - 1 and (rollout_id + 1) % self.args.router_load_snapshot_interval == 0
        layers = []
        for router, counts in zip(self.routers, self.pending, strict=True):
            item = {"layer": router.layer_number, "metrics": summarize_counts(counts, self.rank_experts)}
            if self.first_check:
                item["accounting"] = {"routed_assignments": counts.sum().item(),
                    "expected_assignments": self.packed_positions * self.config.moe_router_topk,
                    "real_positions": self.real_positions, "packed_positions": self.packed_positions,
                    "padding_positions": self.packed_positions - self.real_positions,
                    "counts_match_packed_work": counts.sum().item() == self.packed_positions * self.config.moe_router_topk}
            if snapshot_due:
                item["counts"] = counts.tolist()
            layers.append(item)
        group = self.mpu.get_pipeline_model_parallel_group()
        ranks = dist.get_process_group_ranks(group)
        writer = dist.get_rank() == ranks[-1]
        gathered = [None] * len(ranks) if writer else None
        # Only pipeline assembly: no second TP/CP/DP count reduction, no per-layer collective.
        dist.gather_object(layers, gathered, dst=ranks[-1], group=group)
        if writer:
            self._publish([item for stage in gathered for item in stage], rollout_id, step_id, completed, snapshot_due)
        self.first_check = False

    def _publish(self, layers, rollout_id, step_id, completed, snapshot_due):
        from miles.utils.tracking_utils import tracking

        layers.sort(key=lambda item: item["layer"])
        if len(layers) != 75 or len({item["layer"] for item in layers}) != 75:
            raise RuntimeError("Pipeline assembly did not cover the 75 GLM MoE layers exactly once")
        if self.first_check:
            self.accounting_valid = all(item["accounting"]["counts_match_packed_work"] for item in layers)
        metrics = {"router_load/completed_updates": completed, "router_load/rollout_id": rollout_id,
                   "router_load/accounting_passed": int(self.accounting_valid)}
        # Bad count accounting degrades observations, not a scientific balance threshold.
        # Do not publish misleading ratios or snapshots if the one-time check failed.
        if self.accounting_valid:
            for item in layers:
                for name, value in item["metrics"].items():
                    for axis in ("by_update", "by_rollout"):
                        metrics[f"router_load/{axis}/layer_{item['layer']}/{name}"] = value
        tracking.log(self.args, metrics, step_key="router_load/completed_updates")
        output = Path(self.args.router_load_output)
        output.mkdir(parents=True, exist_ok=True)
        if self.first_check:
            receipt = {"completed_updates": completed, "rollout_id": rollout_id,
                       "gates_frozen_optimizer_excluded_gate_and_bias_unchanged": True,
                       "counts_semantics": "actual packed dispatch work, INCLUDING explicitly counted alignment padding; not unique real-token counts",
                       "reduction": "one existing stacked TP/CP/DP reduction; one representative per PP stage",
                       "layers": layers, "expert_rank_mapping": self.rank_experts}
            (output / "first-update-accounting.json").write_text(json.dumps(receipt, allow_nan=False) + "\n")
        if snapshot_due and self.accounting_valid:
            payload = {"completed_updates": completed, "completed_rollouts": rollout_id + 1,
                       "counts": [item["counts"] for item in layers],
                       "layer_ids": [item["layer"] for item in layers], "expert_rank_mapping": self.rank_experts,
                       "counts_semantics": "actual packed dispatch including alignment padding"}
            (output / f"rollout_{rollout_id + 1:04d}.json").write_text(json.dumps(payload, allow_nan=False) + "\n")


def get_router_load_observer(args, config, model, optimizer):
    if not getattr(args, "observe_router_load", False):
        return None
    if not hasattr(config, "router_load_observer"):
        config.router_load_observer = RouterLoadObserver(args, config, model, optimizer)
    return config.router_load_observer
