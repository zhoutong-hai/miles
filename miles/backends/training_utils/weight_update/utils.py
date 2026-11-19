from miles.backends.training_utils.weight_update.checksum_utils import hash_tensor_sha256


def record_lora_checksums(bucket, checksums) -> None:
    """Accumulate the sha256 checksums the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        checksums[lora_name][hf_key] = hash_tensor_sha256(tensor)


def record_lora_metadata(bucket, manifest) -> None:
    """Record completeness and layout without reading tensor contents."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        entries = manifest[lora_name]
        if hf_key in entries:
            raise ValueError(f"Duplicate streamed LoRA factor: {name}")
        if not tensor.shape or any(size <= 0 for size in tensor.shape) or not tensor.is_floating_point():
            raise ValueError(f"Invalid streamed LoRA factor: {name}")
        entries[hf_key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
