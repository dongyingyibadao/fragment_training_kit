"""Small deterministic regression adapter; no OpenPI/data/download dependency.

The 256 rows exhaust all +/-1 patterns in eight dimensions. Their order changes
with the committed step and seed but each global batch has the same objective.
The tiny FP32 model is for loss/gradient/recovery validation, not a transformer
throughput benchmark. Models and Adam state still execute on the allocated NPU.
"""
from __future__ import annotations

from fragment_training_kit.adapters import AtomicLossEngine


def build_model():
    import torch

    class Regression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, features):
            # Explicit elementwise math avoids device-specific matmul modes.
            return (features * self.weight).sum(dim=-1) + self.bias

    return Regression()


def batch_for_positions(positions, *, seed, device="cpu"):
    """Pure sample identity: independent of rank, worker count and process RNG."""
    import torch
    positions = list(positions)
    rows = [((position % 256) * 73 + seed + (position // 256) * 17) % 256 for position in positions]
    features = torch.tensor([[1. if row & (1 << bit) else -1. for bit in range(8)] for row in rows],
                            dtype=torch.float32, device=device)
    truth = torch.tensor([.25, -.5, .75, 1., -.75, .5, -.25, 1.25], dtype=torch.float32, device=device)
    targets = (features * truth).sum(dim=-1) + .375
    return features, targets


class ToyEngine(AtomicLossEngine):
    def __init__(self, config, *, device="cpu"):
        super().__init__(config, model=build_model(), device=device)

    def prepare_samples(self, positions, *, base_step):
        return batch_for_positions(positions, seed=int(self.config.get("seed", 7)), device=self.torch_device)

    def losses(self, batch):
        features, targets = batch
        return (self.model(features) - targets) ** 2


def create_engine(config, *, device_index=0):
    import torch
    import torch_npu  # noqa: F401 -- only the allocated child imports NPU backend
    torch.npu.set_device(device_index)
    return ToyEngine(config, device=f"npu:{device_index}")


def config(steps=20):
    return {"resident_adapter": "fragment_training_kit.examples.toy_regression:create_engine",
            "batch_size": 256, "worker_microbatch_size": 4,
            "resident_rng_scheme": "toy-orthogonal256-canonical-position-v1",
            "resident_precision_contract": "native FP32 parameters and FP32 AdamW moments; no FP32 master",
            "disable_activation_checkpointing": True, "resident_data_workers": 0,
            "seed": 7, "target_steps": steps, "peak_lr": .05, "end_lr": .05,
            "warmup_steps": 0, "decay_steps": steps, "clip_gradient_norm": 10.,
            "resident_final_state": "checkpoint", "resident_verify_every": 5}
