"""Fixed four-sample random streams, independent of worker task packing.

This module does not claim work or change the training protocol. Opt-in callers
must wrap ``mean_unit_loss`` in their existing single outer value_and_grad.
Mapping value_and_grad over units would instead return U full gradient trees.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import NamedTuple


ATOMIC_SIZE = 4
TASK_UNIT_COUNTS = (1, 2, 4)
RNG_SCHEME = "pi05-atomic4-blake2b64-fold32-v1"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def unit_seed(run_seed, version, start_position):
    """A stable 64-bit seed for one absolute aligned unit; retries do not enter it."""
    run_seed = _integer(run_seed, "run_seed")
    version = _integer(version, "version")
    start_position = _integer(start_position, "start_position")
    if start_position % ATOMIC_SIZE:
        raise ValueError("Atomic unit must start at an absolute multiple of four")
    encoded = f"{RNG_SCHEME}:{run_seed}:{version}:{start_position}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class AtomicUnit:
    version: int
    unit_id: int
    sample_positions: tuple[int, ...]
    seed: int


class AtomicPreparedTask(NamedTuple):
    """Resident arrays plus immutable sample identities, traversable as a PyTree."""

    observation: object
    actions: object
    rngs: object
    sample_positions: tuple[int, ...]
    version: int


def describe_units(sample_positions, *, run_seed, version, batch_size=256):
    """Validate a 4/8/16-sample assignment and describe its canonical units.

    Each four-position block must be contiguous and globally aligned. Blocks
    must be strictly ordered and disjoint but may be nonadjacent, so outstanding
    units can be repacked after a worker disappears. No unit may cross a global
    optimizer version's sample range.
    """
    run_seed = _integer(run_seed, "run_seed")
    version = _integer(version, "version")
    batch_size = _integer(batch_size, "batch_size", minimum=ATOMIC_SIZE)
    if batch_size % ATOMIC_SIZE:
        raise ValueError("Global batch must be divisible by four")
    if isinstance(sample_positions, (str, bytes)):
        raise ValueError("Sample positions must be an integer sequence")
    try:
        positions = tuple(_integer(value, "sample_position") for value in sample_positions)
    except TypeError:
        raise ValueError("Sample positions must be an integer sequence") from None
    if len(positions) not in tuple(ATOMIC_SIZE * count for count in TASK_UNIT_COUNTS):
        raise ValueError("An assignment must contain exactly 4, 8 or 16 sample positions")
    lower, upper = version * batch_size, (version + 1) * batch_size
    units = []
    for offset in range(0, len(positions), ATOMIC_SIZE):
        block = positions[offset:offset + ATOMIC_SIZE]
        start = block[0]
        if start % ATOMIC_SIZE or block != tuple(range(start, start + ATOMIC_SIZE)):
            raise ValueError("Each atomic unit must contain four contiguous aligned positions")
        if not lower <= start or block[-1] >= upper:
            raise ValueError("Atomic unit is outside the requested parameter version")
        if units and start <= units[-1].sample_positions[-1]:
            raise ValueError("Atomic units must be ordered and disjoint")
        units.append(AtomicUnit(version=version, unit_id=(start - lower) // ATOMIC_SIZE,
                                sample_positions=block, seed=unit_seed(run_seed, version, start)))
    return tuple(units)


def unit_rngs(units):
    """Return typed JAX keys, folding both halves of every 64-bit unit seed."""
    import jax
    import jax.numpy as jnp

    units = tuple(units)
    if len(units) not in TASK_UNIT_COUNTS or any(not isinstance(unit, AtomicUnit) for unit in units):
        raise ValueError("Expected one, two or four AtomicUnit descriptors")
    for unit in units:
        if type(unit.seed) is not int or not 0 <= unit.seed < 2 ** 64:
            raise ValueError("Atomic seed must be an unsigned 64-bit integer")
    low = jnp.asarray([unit.seed & 0xFFFFFFFF for unit in units], dtype=jnp.uint32)
    high = jnp.asarray([unit.seed >> 32 for unit in units], dtype=jnp.uint32)
    # RBG's vmap semantics can generate a mapped batch from its first key.
    # Fix Threefry explicitly so the unit contract does not depend on a global
    # JAX_DEFAULT_PRNG_IMPL setting in a newly admitted worker.
    return jax.vmap(lambda lo, hi: jax.random.fold_in(jax.random.key(lo, impl="threefry2x32"), hi))(low, high)


def _validate_batch(observation, actions, *, packed=False):
    import jax

    shape = getattr(actions, "shape", None)
    if shape is None or len(shape) < (4 if packed else 3):
        raise ValueError("Actions must have batch, action-horizon and action dimensions")
    prefix = tuple(shape[:2] if packed else shape[:1])
    if packed:
        if prefix[0] not in TASK_UNIT_COUNTS or prefix[1] != ATOMIC_SIZE:
            raise ValueError("Packed batch must have shape [1/2/4, 4, ...]")
    elif prefix[0] not in tuple(ATOMIC_SIZE * count for count in TASK_UNIT_COUNTS):
        raise ValueError("Prepared batch must contain exactly 4, 8 or 16 samples")
    leaves = jax.tree.leaves(observation)
    if not leaves:
        raise ValueError("Prepared observation must contain array leaves")
    for leaf in leaves:
        leaf_shape = getattr(leaf, "shape", None)
        if leaf_shape is None or tuple(leaf_shape[:len(prefix)]) != prefix:
            raise ValueError("Every observation array must share the action batch dimensions")
    return prefix


def pack_prepared_units(prepared_microbatches):
    """Split prepared 4/8/16 batches into four-sample units and stack [U,4,...].

    Data transfer and sample transformation stay with the caller. PyTree leaves
    (including an OpenPI Observation dataclass) retain their dtypes and values.
    This function does not receive positions; callers must separately validate
    positions with describe_units and match their descriptor count.
    """
    import jax
    import jax.numpy as jnp

    units = []
    for prepared in prepared_microbatches:
        if not isinstance(prepared, (tuple, list)) or len(prepared) != 2:
            raise ValueError("Each prepared microbatch must be (observation, actions)")
        observation, actions = prepared
        count = _validate_batch(observation, actions)[0]
        for start in range(0, count, ATOMIC_SIZE):
            units.append((jax.tree.map(lambda value: value[start:start + ATOMIC_SIZE], observation),
                          actions[start:start + ATOMIC_SIZE]))
    if len(units) not in TASK_UNIT_COUNTS:
        raise ValueError("Prepared assignment must contain one, two or four atomic units")
    schema = jax.tree.structure(units[0])
    reference = [(tuple(value.shape), str(value.dtype)) for value in jax.tree.leaves(units[0])]
    for unit in units[1:]:
        if jax.tree.structure(unit) != schema or [(tuple(value.shape), str(value.dtype)) for value in jax.tree.leaves(unit)] != reference:
            raise ValueError("Atomic observations/actions must have matching structures, shapes and dtypes")
    observation = jax.tree.map(lambda *values: jnp.stack(values), *(unit[0] for unit in units))
    actions = jnp.stack([unit[1] for unit in units])
    if [str(value.dtype) for value in jax.tree.leaves((observation, actions))] != [dtype for _, dtype in reference]:
        raise ValueError("Current JAX configuration cannot preserve the prepared input dtypes")
    return observation, actions


def unpack_prepared_units(packed_observation, packed_actions):
    """Return the original four-sample observation/action units without changes."""
    import jax

    count, _ = _validate_batch(packed_observation, packed_actions, packed=True)
    return tuple((jax.tree.map(lambda value: value[index], packed_observation), packed_actions[index])
                 for index in range(count))


def mean_unit_loss(unit_loss, rngs, packed_observation, packed_actions, *, return_unit_losses=False):
    """Vmap loss values first; the caller differentiates this scalar only once.

    Example inside an existing NNX loss_fn(model)::

        return mean_unit_loss(
            lambda rng, obs, act: model.compute_loss(rng, obs, act, train=True),
            rngs, packed_observation, packed_actions)

    The unit callable always sees batch four, preserving random draw shapes.
    This returns one loss scalar (or scalar plus tiny auxiliary unit-loss
    vector), not gradients. A single outer grad returns one gradient tree;
    compiler temporary memory and BF16 rounding still require GPU validation.
    """
    import jax
    import jax.numpy as jnp

    count, _ = _validate_batch(packed_observation, packed_actions, packed=True)
    if not hasattr(rngs, "shape") or len(rngs.shape) < 1 or rngs.shape[0] != count:
        raise ValueError("Exactly one RNG key is required per atomic unit")
    losses = jax.vmap(lambda rng, obs, act: jnp.mean(unit_loss(rng, obs, act)))(
        rngs, packed_observation, packed_actions,
    )
    mean = jnp.mean(losses)
    return (mean, losses) if return_unit_losses else mean
