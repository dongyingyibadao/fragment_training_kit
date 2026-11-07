"""Avoid decoding camera columns when LeRobot queries future action rows.

The original LeRobot ``__getitem__`` still chooses episode-safe indices,
padding, current observations and all transforms.  This opt-in patch changes
only numeric ``_query_hf_dataset`` lookups on one dataset instance; it never
changes the upstream class or files.  Call ``restore()`` to remove it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from types import MethodType
from typing import Any


def _find_dataset(dataset):
    """Find LeRobot underneath the OpenPI TransformedDataset wrappers."""
    seen = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        if callable(getattr(dataset, "_query_hf_dataset", None)) and hasattr(dataset, "hf_dataset"):
            return dataset
        dataset = getattr(dataset, "_dataset", getattr(dataset, "dataset", None))
    return None


@dataclass
class NumericActionQueries:
    """An installed instance patch and its optional immutable CPU tensor cache."""

    target: Any = None
    enabled: bool = False
    mode: str = "disabled"
    keys: tuple[str, ...] = ()
    initialization_seconds: float = 0.0
    cache_bytes: int = 0
    fallback_calls: int = 0
    reason: str = "disabled by configuration"
    _source: Any = field(default=None, repr=False)
    _original: Any = field(default=None, repr=False)
    _previous_instance_value: Any = field(default=None, repr=False)
    _had_instance_value: bool = field(default=False, repr=False)
    _views: dict = field(default_factory=dict, repr=False)
    _arrays: dict = field(default_factory=dict, repr=False)

    def summary(self):
        return {key: getattr(self, key) for key in (
            "enabled", "mode", "keys", "initialization_seconds", "cache_bytes", "fallback_calls", "reason"
        )}

    def query(self, query_indices):
        import torch

        # Recording/lazy reload can replace the HF dataset.  Never serve an old
        # cache for a new source, even when its shape or length is unchanged.
        if self.target.hf_dataset is not self._source:
            self.fallback_calls += 1
            return self._original(query_indices)
        others = {key: indices for key, indices in query_indices.items() if key not in self._views}
        result = self._original(others) if others else {}
        for key, indices in query_indices.items():
            if key not in self._views:
                continue
            if self.mode == "memory":
                # index_select creates a result owned by the caller, so a
                # downstream in-place transform cannot corrupt cached actions.
                index = torch.as_tensor(indices, dtype=torch.long)
                result[key] = self._arrays[key].index_select(0, index)
            else:
                result[key] = torch.stack(self._views[key][indices][key])
        return result

    def restore(self):
        if not self.enabled:
            return
        if getattr(self.target, "_jade_numeric_action_queries", None) is not self:
            raise RuntimeError("Another numeric-query patch replaced this handle")
        if self._had_instance_value:
            self.target._query_hf_dataset = self._previous_instance_value
        else:
            del self.target._query_hf_dataset
        del self.target._jade_numeric_action_queries
        self.enabled = False
        self.reason = "restored original query method"
        self._views.clear()
        self._arrays.clear()


def install_numeric_action_queries(dataset, *, enabled=True, mode="columns", chunk_rows=8192):
    """Optimize numeric delta queries, returning a removable patch handle.

    ``mode='columns'`` retains the original HF transform but reads only the
    requested column. ``mode='memory'`` materializes those same transformed
    values once in CPU tensors, normally a few MiB of LIBERO actions. Neither
    mode prefetches or changes image/RNG transforms. Unsupported datasets and
    explicit ``enabled=False`` leave the dataset untouched and return a
    disabled handle. Reinstalling on the same instance returns its handle.
    """
    began = time.monotonic()
    if not enabled:
        return NumericActionQueries()
    if mode not in ("columns", "memory"):
        raise ValueError("numeric action query mode must be 'columns' or 'memory'")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    target = _find_dataset(dataset)
    if target is None:
        return NumericActionQueries(reason="no compatible LeRobot dataset found")
    existing = getattr(target, "_jade_numeric_action_queries", None)
    if existing is not None:
        return existing
    source = target.hf_dataset
    if source is None or not callable(getattr(source, "select_columns", None)):
        return NumericActionQueries(reason="HF dataset is unavailable or has no column views")
    features = getattr(getattr(target, "meta", None), "features", {})
    query_keys = getattr(target, "delta_indices", None) or {}
    keys = tuple(key for key in query_keys
                 if str(features.get(key, {}).get("dtype", "")).startswith(("float", "int", "uint", "bool")))
    if not keys:
        return NumericActionQueries(reason="no numeric delta-query columns")
    handle = NumericActionQueries(
        target=target, mode=mode, keys=keys, _source=source,
        _original=target._query_hf_dataset,
        _had_instance_value="_query_hf_dataset" in vars(target),
        _previous_instance_value=vars(target).get("_query_hf_dataset"),
    )
    import torch

    for key in keys:
        view = source.select_columns([key])
        handle._views[key] = view
        if mode == "memory":
            chunks = []
            for start in range(0, len(view), chunk_rows):
                # Preserve precisely the original per-row HF torch transform,
                # including dtype; do not reconstruct from Python float lists.
                chunks.append(torch.stack(view[start:start + chunk_rows][key]))
            if not chunks:
                return NumericActionQueries(reason="numeric action dataset is empty")
            values = torch.cat(chunks, dim=0)
            handle._arrays[key] = values
            handle.cache_bytes += values.numel() * values.element_size()

    def query(_target, query_indices):
        return handle.query(query_indices)

    # Publish only after all views/caches were built successfully.  If their
    # construction raises, the original dataset instance is untouched.
    target._query_hf_dataset = MethodType(query, target)
    target._jade_numeric_action_queries = handle
    handle.enabled = True
    handle.reason = "numeric columns isolated from camera decoding"
    handle.initialization_seconds = time.monotonic() - began
    return handle
