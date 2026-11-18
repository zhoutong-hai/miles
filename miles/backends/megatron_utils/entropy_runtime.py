"""Megatron-only ownership and collectives for native entropy feedback."""

import json
import logging
from copy import deepcopy

import torch.distributed as dist

from miles.backends.training_utils.loss.hub.entropy_observation import capture_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.entropy_control import EntropyController, make_packet, merge_response_fragments, spec_from_args
from miles.utils.tracking_utils import tracking

logger = logging.getLogger(__name__)


class EntropyWorkerState:
    def __init__(self, args):
        self.args = args
        self.packet = None
        self.checkpoint_state = None

    def set_packet(self, packet):
        expected = make_packet(run=self.args.wandb_run_id, source=self.args.entropy_control_source_commit, spec=spec_from_args(self.args), for_update=packet["for_update"], scale=packet["scale"])
        if packet != expected:
            raise ValueError("Worker entropy packet source, constants or hash differs")
        if self.packet is not None and packet["for_update"] not in (self.packet["for_update"], self.packet["for_update"] + 1):
            raise ValueError("Worker entropy packet cursor is not consecutive")
        if self.packet is not None and packet["for_update"] == self.packet["for_update"] and packet != self.packet:
            raise ValueError("A retried entropy update cannot change its packet")
        self.packet = deepcopy(packet)
        return packet["sha256"]

    def set_checkpoint_state(self, state):
        controller = EntropyController.restore(state)
        if controller.run != self.args.wandb_run_id or controller.source != self.args.entropy_control_source_commit or controller.spec != spec_from_args(self.args) or controller.pending is None or controller.pending["used"] != self.packet:
            raise ValueError("Native checkpoint entropy state differs from the worker's used packet")
        self.checkpoint_state = deepcopy(state)
        return state["pending"]["used"]["sha256"]

    def capture(self, *, rollout_id):
        if self.packet is None or self.packet["for_update"] != rollout_id + 1:
            raise ValueError("Training update lacks its matching acknowledged entropy packet")
        parallel = get_parallel_state()
        return capture_entropy(self.packet, cp_rank=parallel.cp.rank, record_enabled=parallel.is_pp_last_stage and parallel.tp.rank == 0, tis_low=self.args.tis_clip_low, tis_high=self.args.tis_clip)

    def finish(self, capture):
        parallel = get_parallel_state()
        if not capture.record_enabled:
            return None
        group = parallel.intra_dp_cp
        if group.gloo_group is None:
            raise ValueError("Entropy response aggregation requires the native DP×CP Gloo group")
        gathered = [None] * group.size
        dist.all_gather_object(gathered, capture.fragments, group=group.gloo_group)
        if group.rank != 0:
            return None
        rows = merge_response_fragments([row for part in gathered for row in part], cp_size=parallel.cp.size)
        return dict(run=self.args.wandb_run_id, source=self.args.entropy_control_source_commit, completed_update=self.packet["for_update"], used_sha256=self.packet["sha256"], responses=rows)

    def log_publication(self, event, metrics):
        if get_parallel_state().is_metrics_rank:
            tracking.log(self.args, {"entropy/step": event["completed_update"], **metrics}, step_key="entropy/step")
            logger.info("ENTROPY_PUBLICATION %s", json.dumps(event, sort_keys=True, allow_nan=False))
        return event["completed_update"]
