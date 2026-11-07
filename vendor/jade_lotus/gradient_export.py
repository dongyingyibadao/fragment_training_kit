"""Explicit host snapshot mode for independent Ascend gradient workers."""

FP32_SNAPSHOT = "fp32_snapshot_v1"
NATIVE_SNAPSHOT = "native_snapshot_v1"


def gradient_export_mode(config):
    """Reject unsupported combinations before leases or device work start."""
    mode = config.get("npu_gradient_export", FP32_SNAPSHOT)
    if mode not in (FP32_SNAPSHOT, NATIVE_SNAPSHOT):
        raise ValueError("npu_gradient_export must be fp32_snapshot_v1 or native_snapshot_v1")
    if mode == NATIVE_SNAPSHOT:
        if (config.get("backend") != "torch_npu"
                or config.get("training_mode") != "full"
                or config.get("atomic_batching") is not True
                or config.get("gradient_transport") != "sharded_blob_v1"
                or config.get("gradient_wire_dtype") != "native_mixed_exact"
                or config.get("npu_gradient_transfer") != "pinned_native_v1"):
            raise ValueError("native_snapshot_v1 requires torch_npu, full atomic sharded_blob_v1, "
                             "native_mixed_exact and pinned_native_v1")
        from jade_lotus.array_blob import native_wire_policy_sha256, validate_native_wire_policy

        policy = validate_native_wire_policy(config.get("gradient_wire_policy"))
        if native_wire_policy_sha256(policy) != config.get("gradient_wire_policy_sha256"):
            raise ValueError("Native snapshot wire policy checksum differs from configuration")
    return mode
