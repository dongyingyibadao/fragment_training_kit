"""Select a computation backend without importing another device runtime."""


def backend_name(config):
    name = config.get("backend", "jax")
    if name not in ("jax", "torch_npu"):
        raise ValueError(f"Unsupported gradient backend: {name!r}")
    return name


def create_gradient_engine(config, device_index=None):
    if backend_name(config) == "torch_npu":
        from jade_lotus.npu_gradient_model import TorchNpuGradientEngine

        return TorchNpuGradientEngine(config, device_index=device_index)
    from jade_lotus.gradient_model import GradientEngine

    return GradientEngine(config, device_index=device_index)
