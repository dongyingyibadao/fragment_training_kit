"""Math and interface checks for plugging another model into resident training."""
import importlib.util
from pathlib import Path
import sys

# Works from the kit directory or its parent after copying the whole kit.
KIT = Path(__file__).resolve().parents[1]
if str(KIT.parent) not in sys.path:
    sys.path.insert(0, str(KIT.parent))

import pytest
import torch

from fragment_training_kit.adapters import accumulate_units, create_engine
from fragment_training_kit.examples.audit_toy import check_atomic_math, reference
from fragment_training_kit.examples.toy_regression import ToyEngine, batch_for_positions, config


def test_atomic_gradient_matches_full_batch_and_loss_repair():
    result = check_atomic_math(config())
    assert result["passed"]
    assert result["healthy_gradient_pointers_preserved"]
    assert all(row["max_gradient_abs_error"] == 0 for row in result["layouts"])


def test_cpu_adam_has_expected_nontrivial_descent():
    _model, _optimizer, history, loss = reference(config(), 20)
    assert history[0]["loss_before_update"] == 4.453125
    assert loss < .2
    assert all(a["loss_before_update"] > b["loss_before_update"] for a, b in zip(history, history[1:]))


def test_data_positions_replay_and_global_permutation():
    for step in (0, 3, 27):
        x, y = batch_for_positions(range(step * 256, (step + 1) * 256), seed=7)
        xx, yy = batch_for_positions(range(step * 256, (step + 1) * 256), seed=7)
        assert torch.equal(x, xx) and torch.equal(y, yy)
        assert torch.unique(x, dim=0).shape == (256, 8)
        torch.testing.assert_close(x.T @ x, torch.eye(8) * 256, rtol=0, atol=0)


def test_microbatch_does_not_change_math():
    reference_engine = ToyEngine(config())
    accumulate_units(reference_engine, base_step=4, units=list(range(64)))
    for micro in (1, 2):
        engine = ToyEngine(dict(config(), worker_microbatch_size=micro))
        accumulate_units(engine, base_step=4, units=list(range(64)))
        for name in engine.parameters:
            torch.testing.assert_close(engine.parameters[name].grad, reference_engine.parameters[name].grad, rtol=0, atol=0)


def test_bad_assignments_and_scalar_loss_are_rejected():
    engine = ToyEngine(config())
    for units in ([], [0, 0], [64], [-1]):
        with pytest.raises(ValueError):
            accumulate_units(engine, base_step=0, units=units)
    engine.losses = lambda batch: torch.zeros((), requires_grad=True)
    with pytest.raises(ValueError, match="one scalar loss per example"):
        accumulate_units(engine, base_step=0, units=[0])


def test_checkpointing_flag_is_rejected():
    engine = ToyEngine(config())
    engine.model.gradient_checkpointing = True
    with pytest.raises(ValueError, match="checkpointing"):
        accumulate_units(engine, base_step=0, units=[0])


def test_factory_cannot_execute_arbitrary_expression():
    with pytest.raises(ValueError, match="module:factory"):
        create_engine(dict(config(), resident_adapter="__import__('os').system('false')"))


def test_existing_training_loop_uses_explicit_callback():
    core = KIT / "vendor"
    path = core / "scripts/probe_resident_openpi_fault.py"
    spec = importlib.util.spec_from_file_location("portable_adapter_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = ToyEngine(config())
    row = module.backward_units(engine, base_step=0, units=[0, 2])
    assert row["samples"] == 8 and row["backward_calls"] == 2
