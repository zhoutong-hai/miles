"""Run under torchrun --nproc-per-node=4 -m pytest for real CP4/CUDA coverage."""
import os

import pytest
import torch

from tests.standalone.test_glm_dsa_contract import callsite_case


@pytest.mark.parametrize("share", [False, True])
def test_glm_forward_with_native_cp_scatter_and_gpu_kernels(share):
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires an explicitly authorized four-GPU torchrun qualification")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region

    def scatter(value, **kwargs):
        return scatter_to_sequence_parallel_region(value, group=torch.distributed.group.WORLD)

    callsite_case(torch.distributed.get_rank(), share, device=torch.cuda.current_device(), gpu_kernels=True, scatter=scatter)
    torch.distributed.barrier()
