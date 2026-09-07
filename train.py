import asyncio
import json
import logging
import os
from pathlib import Path

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.rollout.rollout_manager import get_rollout_offload_tags
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums
from miles.utils.audit_utils.experiment_receipt import resolve_experiment_run_id
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.lora import is_lora_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


async def train(args):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    if args.colocate_memory_peak_device == "gpu":
        assert args.offload_train and args.offload_rollout, "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"
        if is_lora_enabled(args):
            raise NotImplementedError("--colocate-memory-peak-device gpu only supports full-parameter training")

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    if args.debug_startup_finite_probe_only:
        await rollout_manager.onload_weights.remote()
        await rollout_manager.onload_kv.remote()
        probe = await rollout_manager.startup_finite_probe.remote()
        receipt_path = os.environ.get("MILES_STARTUP_FINITE_PROBE_RECEIPT")
        if not receipt_path:
            raise RuntimeError("MILES_STARTUP_FINITE_PROBE_RECEIPT is required")
        result = {
            "schema_version": 1,
            "status": "PASS",
            "run_id": resolve_experiment_run_id(args, receipt_path),
            "case": "greedy_batch8_base",
            "publication_events": 0,
            "rollouts": 0,
            "optimizer_updates": 0,
            "probe": probe,
        }
        target = Path(receipt_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(target)
        logger.info(
            "[STARTUP-FINITE-PROBE-TERMINAL] status=PASS engines=%d publication=0 rollouts=0 updates=0",
            len(probe["engines"]),
        )
        await rollout_manager.dispose.remote()
        return

    integrated_pre_trainer_probe = None
    integrated_pre_trainer_checksums = None
    if args.debug_integrated_base_probe_only:
        await rollout_manager.onload_weights.remote()
        await rollout_manager.onload_kv.remote()
        integrated_pre_trainer_probe = await rollout_manager.integrated_base_probe.remote("pre_trainer")
        integrated_pre_trainer_checksums = flatten_inference_engine_checksums(await rollout_manager.check_weights.remote("checksum", selector="target"))
        await rollout_manager.offload.remote(tags=get_rollout_offload_tags(args))

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.debug_integrated_base_probe_only:
        await rollout_manager.onload_weights.remote()
        await rollout_manager.onload_kv.remote()
        integrated_post_trainer_probes = []
        integrated_post_trainer_checksums = []
        for _ in range(3):
            integrated_post_trainer_probes.append(await rollout_manager.integrated_base_probe.remote("post_trainer", integrated_pre_trainer_probe["engines"]))
            integrated_post_trainer_checksums.append(flatten_inference_engine_checksums(await rollout_manager.check_weights.remote("checksum", selector="target")))
        if any(integrated_pre_trainer_checksums != post_checksums for post_checksums in integrated_post_trainer_checksums):
            raise RuntimeError("rollout weight checksums changed across trainer load/offload")
        receipt_path = os.environ.get("MILES_INTEGRATED_BASE_PROBE_RECEIPT")
        if not receipt_path:
            raise RuntimeError("MILES_INTEGRATED_BASE_PROBE_RECEIPT is required")
        result = {
            "schema_version": 1,
            "status": "PASS",
            "run_id": resolve_experiment_run_id(args, receipt_path),
            "publication_events": 0,
            "rollouts": 0,
            "optimizer_updates": 0,
            "pre_trainer_probe": integrated_pre_trainer_probe,
            "post_trainer_probe_repetitions": 3,
            "post_trainer_probes": integrated_post_trainer_probes,
            "pre_trainer_weight_checksums": integrated_pre_trainer_checksums,
            "post_trainer_weight_checksums": integrated_post_trainer_checksums,
            "weight_checksums_equal": True,
        }
        target = Path(receipt_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(target)
        logger.info(
            "[INTEGRATED-BASE-PROBE-TERMINAL] status=PASS engines=%d publication=0 rollouts=0 updates=0",
            integrated_post_trainer_probes[-1]["comparison"]["engine_count"],
        )
        await rollout_manager.dispose.remote()
        return

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    if args.offload_rollout and args.colocate_memory_peak_device != "gpu":
        await rollout_manager.onload_weights.remote()

    initial_lora_probe = None
    initial_lora_probe_kv_loaded = False
    if args.check_initial_lora_zero_effect:
        assert is_lora_enabled(args), "--check-initial-lora-zero-effect requires LoRA"
        assert args.check_lora_weight_equal, "--check-initial-lora-zero-effect requires --check-lora-weight-equal"
        await rollout_manager.onload_kv.remote()
        initial_lora_probe = await rollout_manager.initial_lora_zero_effect_probe.remote("before")
        if args.offload_rollout:
            await rollout_manager.offload_kv.remote()

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_initial_lora_zero_effect:
        await rollout_manager.onload_kv.remote()
        await rollout_manager.initial_lora_zero_effect_probe.remote("after", initial_lora_probe)
        await rollout_manager.initial_lora_slot_audit.remote()
        initial_lora_probe_kv_loaded = True

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout and not initial_lora_probe_kv_loaded:
        await rollout_manager.onload_kv.remote()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await rollout_manager.eval.remote(rollout_id=0)

    async def offload_train():
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1

        async def save_training_model(model):
            if args.use_critic and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic and args.offload_train:
                await model.offload()

        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await save_training_model(actor_model)
        if args.use_critic:
            await save_training_model(critic_model)
        await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_pack = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            if args.colocate_memory_peak_device == "gpu":
                await rollout_manager.offload_kv.remote()
                await actor_model.onload()
                await rollout_manager.offload_weights.remote()
            else:
                await rollout_manager.offload.remote(tags=get_rollout_offload_tags(args))

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_pack)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_pack, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
        else:
            await actor_model.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        # nothing generates after the last rollout, so its offload/reload/update only serves a final eval
        if rollout_id + 1 < args.num_rollout or should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            if args.colocate_memory_peak_device == "gpu":
                await actor_model.clear_memory()
                await rollout_manager.onload_weights.remote()
                await offload_train()
            else:
                await offload_train()
                if args.offload_rollout:
                    await rollout_manager.onload_weights.remote()
            await actor_model.update_weights(rollout_id=rollout_id)
            if args.offload_rollout:
                await rollout_manager.onload_kv.remote()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

        if args.debug_exit_after_rollout is not None and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout:
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
