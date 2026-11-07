"""Opt-in removal of explicitly declared, entirely masked pi05 image slots.

The host must validate every concrete batch before device transfer/JIT. The
compiled loss retains a dynamic all-false guard too, so bypassing that host
boundary cannot silently train on a batch with a newly valid skipped image.
No image or prompt is selected from its contents, and no global OpenPI symbols
are patched. An empty configuration delegates directly to the upstream loss.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class Pi05ImageCompaction:
    """Static image-slot selection; disabled unless keys are explicitly given.

    ``validate_observation`` belongs on the host, before ``device_put``. Call
    ``compute_loss`` instead of the model loss inside the normal differentiated
    callable. Parameters, trainable filters, task RNG folding and gradient
    accumulation are all the caller's existing responsibility.
    """

    skip_image_keys: tuple[str, ...] = ()

    def __post_init__(self):
        keys = self.skip_image_keys
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise TypeError("skip_image_keys must be a sequence of image-slot names")
        keys = tuple(keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Every skipped image-slot name must be a nonempty string")
        if len(set(keys)) != len(keys):
            raise ValueError("Skipped image-slot names must be unique")
        object.__setattr__(self, "skip_image_keys", keys)

    @property
    def enabled(self) -> bool:
        return bool(self.skip_image_keys)

    def _validate_structure(self, observation):
        """Check only static metadata; safe during JAX tracing."""
        images, masks = observation.images, observation.image_masks
        missing_images = set(self.skip_image_keys) - set(images)
        missing_masks = set(self.skip_image_keys) - set(masks)
        if missing_images or missing_masks:
            raise ValueError(
                f"Skipped image slots require explicit images and masks; "
                f"missing images={sorted(missing_images)}, masks={sorted(missing_masks)}"
            )
        if not set(images) - set(self.skip_image_keys):
            raise ValueError("Image compaction must retain at least one image slot")
        batch_shape = tuple(observation.state.shape[:-1])
        for key in self.skip_image_keys:
            mask = masks[key]
            shape = getattr(mask, "shape", None)
            if shape is None or tuple(shape) != batch_shape or math.prod(shape) == 0:
                raise ValueError(f"Skipped image mask {key!r} must be nonempty with batch shape {batch_shape}")
            if str(getattr(mask, "dtype", None)) != "bool":
                raise ValueError(f"Skipped image mask {key!r} must have boolean dtype")

    def validate_observation(self, observation) -> None:
        """Reject unsafe selection using concrete host masks for this batch.

        NumPy conversion is deliberately confined to this host-only method.
        Do not put it inside a JIT, gradient function, or a traced scan body.
        """
        if not self.enabled:
            return
        self._validate_structure(observation)
        import jax
        import numpy as np

        for key in self.skip_image_keys:
            mask = observation.image_masks[key]
            if isinstance(mask, jax.core.Tracer):
                raise ValueError("Image compaction validation must run on concrete host batches before JIT")
            if np.asarray(mask).any():
                raise ValueError(f"Skipped image mask {key!r} must be entirely false for every batch item")

    def _without_image_slots(self, observation):
        """Remove static keys after full upstream preprocessing, without reads."""
        return dataclasses.replace(
            observation,
            images={key: image for key, image in observation.images.items() if key not in self.skip_image_keys},
            image_masks={key: mask for key, mask in observation.image_masks.items() if key not in self.skip_image_keys},
        )

    def compute_loss(self, model, rng, observation, actions, *, train: bool = False):
        """The upstream pi05 training loss with masked image tokens omitted.

        Preprocess the complete observation before removing slots, preserving
        all augmentation calls and their original RNGs. The subsequent math
        follows OpenPI's ``Pi0.compute_loss``: the same three RNGs, flow target,
        suffix, attention mask, cumulative valid-token positions and MSE. Since
        omitted image tokens have false input masks and non-autoregressive
        prefix masks, retaining the other tokens preserves their positions and
        attention relations. A different matrix shape may still change floating
        point rounding; numerical equivalence needs empirical validation.
        """
        if not self.enabled:
            return model.compute_loss(rng, observation, actions, train=train)
        if not getattr(model, "pi05", False):
            raise ValueError("Image compaction is supported only for pi05 models")
        self._validate_structure(observation)

        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module
        from openpi.models import pi0

        valid_selection = jnp.asarray(True)
        for key in self.skip_image_keys:
            valid_selection = valid_selection & ~jnp.any(observation.image_masks[key])
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_module.preprocess_observation(preprocess_rng, observation, train=train)
        # Upstream may select its own supported image keys during preprocessing.
        # Reject an unsupported selection instead of silently dropping nothing
        # or removing every image that the model actually embeds.
        self._validate_structure(observation)
        observation = self._without_image_slots(observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        # This scalar mask check and select add no host synchronization. Normal
        # workers already reject nonfinite losses; direct benchmarks must too.
        return jnp.where(valid_selection, loss, jnp.full_like(loss, jnp.nan))
