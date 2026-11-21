"""Optional deployment checkpoint coordination outside trainer collectives."""

import asyncio
from functools import partial

from miles.utils.function_registry import load_function


class CheckpointLifecycle:
    def __init__(self, callback=None):
        self._callback = callback

    @classmethod
    def from_args(cls, args):
        selected = args.custom_checkpoint_lifecycle_hook_path
        if selected is None:
            return cls()
        # Optional native configuration dependencies are only needed by an enabled hook.
        from miles.utils.args.custom_view import compute_custom_function_config

        return cls(partial(load_function(selected.path, sync_required=True), compute_custom_function_config(args, selected)))

    async def event(self, phase, rollout_id=None):
        if self._callback is not None:
            # Admission can wait for storage. Keep the driver's event loop responsive.
            await asyncio.to_thread(self._callback, phase=phase, rollout_id=rollout_id)

    async def close(self):
        await self.event("close")
