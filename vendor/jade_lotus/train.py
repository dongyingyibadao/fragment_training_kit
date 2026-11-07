"""Recoverable single-host OpenPI JAX runner; imports the user's existing checkout.

Checkpoints intentionally use a separate schema and directory from upstream.
The baseline saves the entire state. Adapter-only checkpoints are a later phase.
"""

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time
import urllib.parse

from jade_lotus.runtime import ResumableBatchSampler, atomic_json, emit, sample_seed, writer_lock


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--config", default="pi05_robodojo_abs_eef")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-seconds", type=float, default=300)
    parser.add_argument("--base-params", type=Path)
    parser.add_argument("--lora", action="store_true", help="Use OpenPI's existing LoRA filters (vision/projection weights remain trainable)")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--test-pause-after-step", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--test-pause-save-step", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.fsdp_devices < 1 or args.workers < 0:
        parser.error("steps, batch size, and fsdp devices must be positive; workers must be nonnegative")
    if args.save_steps < 0 or args.save_seconds <= 0:
        parser.error("save steps must be nonnegative and save seconds positive")
    if not args.cpu_smoke and (args.test_pause_after_step or args.test_pause_save_step):
        parser.error("fault injection is only allowed with --cpu-smoke")
    return args


class SeededDataset:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, cursor):
        import numpy as np
        import torch

        epoch, index = cursor
        seed = sample_seed(self.seed, epoch, index)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def stable(value):
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__qualname__, **{f.name: stable(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {str(k): stable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return getattr(value, "__module__", "") + "." + getattr(value, "__qualname__", type(value).__qualname__)
    if hasattr(value, "__dict__"):
        return {"type": type(value).__module__ + "." + type(value).__qualname__, "fields": stable(vars(value))}
    return type(value).__module__ + "." + type(value).__qualname__


def main(args):
    # Only handlers set flags. Saving in a signal handler is unsafe inside JAX/Orbax.
    requested_stop = {"signal": None}

    def request_stop(signum, _frame):
        requested_stop["signal"] = signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    if args.cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
    root, run_dir = args.openpi_root.resolve(), args.run_dir.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    import torch
    from openpi.models import model as model_module
    from openpi.shared import download
    from openpi.training import checkpoints, config as configs, data_loader, sharding, weight_loaders

    if jax.process_count() != 1:
        raise ValueError("This pilot supports one host/process; cross-node elasticity is not implemented")
    if not args.cpu_smoke and any(device.platform != "gpu" for device in jax.devices()):
        raise ValueError("GPU training requires visible GPUs; use --cpu-smoke for the reduced model test")
    if args.batch_size % jax.device_count():
        raise ValueError("Global batch size must be divisible by device count")
    if args.cpu_smoke and args.config != "debug":
        raise ValueError("--cpu-smoke requires --config debug; it does not load pretrained weights")
    if args.cpu_smoke and args.lora:
        raise ValueError("--lora selects real pretrained model variants and cannot be combined with --cpu-smoke")

    # All model and tokenizer inputs must already exist on shared storage.
    def local_only_download(url, **_kwargs):
        parsed = urllib.parse.urlparse(str(url))
        if not parsed.scheme:
            target = Path(url).expanduser()
        else:
            cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
            target = cache / parsed.netloc / parsed.path.lstrip("/")
        if not target.exists():
            raise FileNotFoundError(f"Offline mode: missing existing local asset {target}")
        return target.resolve()

    download.maybe_download = local_only_download
    config = configs.get_config(args.config)
    config = dataclasses.replace(
        config, batch_size=args.batch_size, fsdp_devices=args.fsdp_devices,
        num_workers=args.workers, overwrite=False, resume=True, wandb_enabled=False,
        assets_base_dir=str((root / config.assets_base_dir).resolve()),
    )
    if args.base_params:
        config = dataclasses.replace(config, weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base_params.resolve())))
    if args.lora:
        from openpi.models.pi0_config import Pi0Config

        if not isinstance(config.model, Pi0Config):
            raise ValueError("This pilot's LoRA option supports Pi0/Pi05 configs")
        lora_model = dataclasses.replace(config.model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
        config = dataclasses.replace(config, model=lora_model, freeze_filter=lora_model.get_freeze_filter(), ema_decay=None)
    if args.cpu_smoke:
        from openpi.models import siglip, gemma

        original_module = siglip.Module

        def small_vision(*positional, **kwargs):
            kwargs["variant"] = "mu/32"
            return original_module(*positional, **kwargs)

        siglip.Module = small_vision
        gemma.PALIGEMMA_VOCAB_SIZE = 2048
        # Force real parameter sharding in the reduced model; its matrices would
        # otherwise all fall below upstream's 4 MiB FSDP threshold.
        sharding.fsdp_sharding = functools.partial(sharding.fsdp_sharding, min_size_mbytes=0)
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, dtype="float32", action_dim=8, action_horizon=4, max_token_len=8),
        )

    spec = importlib.util.spec_from_file_location("jade_lotus_openpi_train", root / "scripts/train.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError("This pilot requires an indexed map dataset; RLDS cursor recovery is not implemented")
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(dataset, data_config)

    # Identity includes loader/model/optimizer configuration and lightweight dataset metadata.
    # Data payloads and base weights must remain immutable; their full contents are not hashed.
    identity = {
        "schema": 1, "sampler": "python-random-epoch-v1", "python": sys.version.split()[0],
        "model": stable(config.model), "data": stable(data_config), "dataset_length": len(dataset),
        "batch_size": config.batch_size, "seed": config.seed, "optimizer": stable(config.optimizer),
        "lr_schedule": stable(config.lr_schedule), "ema_decay": config.ema_decay,
        "freeze_filter": stable(config.freeze_filter), "weight_loader": stable(config.weight_loader),
        "cpu_smoke": args.cpu_smoke, "versions": {"jax": jax.__version__, "torch": torch.__version__},
        "source_hashes": {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                          for relative in ("scripts/train.py", "src/openpi/training/data_loader.py",
                                           "src/openpi/models/pi0.py", "src/openpi/models/pi0_config.py",
                                           "src/openpi/models/gemma.py", "src/openpi/models/siglip.py",
                                           "src/openpi/training/optimizer.py", "src/openpi/training/weight_loaders.py",
                                           "src/openpi/training/lerobot_dataset.py", "src/openpi/training/robodojo_dataset.py",
                                           "src/openpi/policies/robodojo_policy.py", "src/openpi/transforms.py")},
        "runner_source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), Path(__file__).with_name("runtime.py"))},
    }
    dataset_root = getattr(data_config, "dataset_root", None)
    if dataset_root:
        identity["dataset_metadata_sha256"] = {}
        for relative in ("meta/info.json", "meta/stats.json", "validation_report.json"):
            path = Path(dataset_root) / relative
            if path.is_file():
                identity["dataset_metadata_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(identity_text.encode()).hexdigest()

    with writer_lock(run_dir):
        emit(run_dir, "startup", devices=[str(d) for d in jax.devices()], fingerprint=fingerprint, cpu_smoke=args.cpu_smoke)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("Run identity changed; refusing to resume with a different model/data/batch/seed")
        else:
            atomic_json(manifest_path, {"fingerprint": fingerprint, "identity": identity})
        # Establish a correctness baseline before enabling a cross-process XLA cache.
        # The installed CPU stack showed native corruption with buffer donation/cache.
        jax.config.update("jax_enable_compilation_cache", False)
        mesh = sharding.make_mesh(args.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        manager = ocp.CheckpointManager(
            run_dir / "checkpoints",
            item_handlers={"state": ocp.PyTreeCheckpointHandler(), "metadata": ocp.JsonCheckpointHandler(), "assets": checkpoints.CallbackHandler()},
            options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=False),
        )
        try:
            latest = manager.latest_step()
            rng, init_rng = jax.random.split(jax.random.key(config.seed))
            state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=latest is not None)
            if latest is not None:
                metadata = manager.restore(latest, args=ocp.args.Composite(metadata=ocp.args.JsonRestore()))["metadata"]
                if metadata["fingerprint"] != fingerprint or metadata["consumed_batches"] != latest:
                    raise ValueError("Checkpoint metadata and run identity/cursor disagree")
                state = manager.restore(latest, args=ocp.args.Composite(state=ocp.args.PyTreeRestore(
                    state, restore_args=ocp.checkpoint_utils.construct_restore_args(state, state_sharding)
                )))["state"]
                if int(state.step) != latest:
                    raise ValueError("Checkpoint step and optimizer-update count disagree")
                emit(run_dir, "restored", step=latest, consumed_batches=metadata["consumed_batches"])
            jax.block_until_ready(state)
            sharded_leaf_count = sum(not leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(state)
                                     if hasattr(leaf, "sharding"))
            emit(run_dir, "initialized", step=int(state.step), sharded_leaf_count=sharded_leaf_count,
                 device_count=jax.device_count())
            start_step = int(state.step)
            if args.steps < start_step:
                raise ValueError("Requested target is behind the durable checkpoint")
            # Create/fetch the iterator only AFTER restoring completed optimizer updates.
            sampler = ResumableBatchSampler(len(dataset), config.batch_size, config.seed, start_step)
            loader = torch.utils.data.DataLoader(
                SeededDataset(dataset, config.seed), batch_sampler=sampler, num_workers=args.workers,
                collate_fn=data_loader._collate_fn,
                multiprocessing_context="spawn" if args.workers else None,
                generator=torch.Generator().manual_seed(config.seed),
            )
            iterator = iter(loader)
            train_step = jax.jit(
                functools.partial(upstream.train_step, config),
                in_shardings=(replicated, state_sharding, data_sharding),
                out_shardings=(state_sharding, replicated), donate_argnums=() if args.cpu_smoke else (1,),
            )
            last_saved = latest
            last_save_time = time.monotonic()
            losses = []

            def pause(event, step):
                emit(run_dir, event, step=step)
                while not requested_stop["signal"]:
                    time.sleep(0.1)

            def save():
                nonlocal last_saved, last_save_time
                completed = int(state.step)
                if completed == last_saved:
                    return

                def save_assets(directory):
                    if completed == args.test_pause_save_step:
                        pause("pause_during_save", completed)
                    if data_config.norm_stats is not None and data_config.asset_id is not None:
                        from openpi.shared import normalize

                        normalize.save(directory / data_config.asset_id, data_config.norm_stats)

                emit(run_dir, "save_started", step=completed)
                started = time.monotonic()
                manager.save(completed, args=ocp.args.Composite(
                    state=ocp.args.PyTreeSave(state),
                    metadata=ocp.args.JsonSave({"schema": 1, "fingerprint": fingerprint, "consumed_batches": completed,
                                               "consumed_samples": completed * config.batch_size}),
                    assets=checkpoints.CallbackSave(save_assets),
                ))
                manager.wait_until_finished()
                last_saved = completed
                last_save_time = time.monotonic()
                emit(run_dir, "checkpoint_committed", step=completed, seconds=last_save_time - started)

            while int(state.step) < args.steps and not requested_stop["signal"]:
                raw_batch = next(iterator)
                batch_digest = hashlib.sha256()
                for leaf in jax.tree.leaves(raw_batch):
                    batch_digest.update(np.ascontiguousarray(leaf).tobytes())
                batch = model_module.Observation.from_dict(raw_batch), raw_batch["actions"]
                batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, np.asarray(x)), batch)
                started = time.monotonic()
                with sharding.set_mesh(mesh):
                    state, info = train_step(rng, state, batch)
                jax.block_until_ready(state)
                completed = int(state.step)
                loss = float(info["loss"])
                if not np.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at update {completed}")
                record = {"step": completed, "loss": loss, "batch_digest": batch_digest.hexdigest()}
                losses.append(record)
                emit(run_dir, "update", **record, seconds=time.monotonic() - started)
                if completed == args.test_pause_after_step:
                    pause("pause_after_step", completed)
                periodic = args.save_steps > 0 and completed % args.save_steps == 0
                if periodic or time.monotonic() - last_save_time >= args.save_seconds or requested_stop["signal"]:
                    save()
            save()
            if requested_stop["signal"]:
                emit(run_dir, "stopped", step=int(state.step), signal=requested_stop["signal"], checkpoint_step=last_saved)
                return
            # Completion is written only after the final durable checkpoint has committed.
            summary = {"step": int(state.step), "checkpoint_step": last_saved, "fingerprint": fingerprint,
                       "losses": losses, "leaf_hashes": {}, "leaf_stats": {}, "cpu_smoke": args.cpu_smoke,
                       "device_count": jax.device_count(), "sharded_leaf_count": sharded_leaf_count}
            if args.cpu_smoke:
                arrays = {}
                for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
                    key = jax.tree_util.keystr(path)
                    array = np.array(leaf)
                    arrays[key] = array
                    summary["leaf_hashes"][key] = hashlib.sha256(array.tobytes()).hexdigest()
                    numeric = array.astype(np.float64)
                    summary["leaf_stats"][key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                                   "sum": float(numeric.sum()), "l2": float(np.linalg.norm(numeric)),
                                                   "max_abs": float(np.max(np.abs(numeric)))}
                np.savez(run_dir / "final_arrays.npz", **arrays)
            summary["state_digest"] = hashlib.sha256(json.dumps(summary["leaf_hashes"], sort_keys=True).encode()).hexdigest() if args.cpu_smoke else None
            atomic_json(run_dir / "final_summary.json", summary)
            emit(run_dir, "complete", step=int(state.step), checkpoint_step=last_saved)
        finally:
            manager.close()


if __name__ == "__main__":
    main(arguments())
