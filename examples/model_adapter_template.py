"""Copy this file into your project, replace the three marked model/data hooks.

Set resident_adapter='your_package.adapter:create_engine' in training_config.
Freeze your package alongside the kit in runtime/ and ensure its dependencies
exist in the job image. First validate deterministic replay and full-batch
gradient equivalence on CPU, then a 2-NPU short run before scale-up.
"""
from fragment_training_kit.adapters import AtomicLossEngine


class YourEngine(AtomicLossEngine):
    def prepare_samples(self, positions, *, base_step):
        # HOOK 1: map canonical positions to dataset indices, epoch, augmentation
        # and explicit model-noise seeds. Load on CPU/prefetch there, then move
        # tensors to self.torch_device. The same (seed, position, base_step)
        # must reproduce the same example on another worker after a loss.
        raise NotImplementedError("Implement deterministic sample loading")

    def losses(self, batch):
        # HOOK 2: normal forward pass, yielding [N] losses. If your model emits
        # token losses [N,T], reduce over T here. Do not backward/update/reduce.
        raise NotImplementedError("Return one scalar loss per example")


def create_engine(config, *, device_index=0):
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(device_index)
    torch.manual_seed(int(config.get("seed", 7)))
    # HOOK 3: construct/load identical initial model. Disable gradient
    # checkpointing and module dropout; other random operators need an
    # explicit sample-scoped RNG path. Do not derive initialization from rank.
    # The current native-state contract supports eager torch.optim.AdamW,
    # contiguous named parameters/buffers, native moments, no FP32 master.
    raise NotImplementedError("Build identical initial model on every rank")
    # model = YourModel(...).to(dtype=...).train()
    # return YourEngine(config, model=model, device=f"npu:{device_index}")
