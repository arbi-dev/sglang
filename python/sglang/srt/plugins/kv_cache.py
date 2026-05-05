"""Plugin registry for custom ``--kv-cache-dtype`` strings.

The built-in dtypes (``auto``, ``bfloat16``, ``fp8_e4m3``, ``fp8_e5m2``,
``fp4_e2m1``) are handled by hardcoded branches in
:meth:`ModelRunner.configure_kv_cache_dtype` and
:meth:`ModelRunnerKVCacheMixin._init_pools`. Out-of-tree backends that
introduce a custom KV-cache layout (e.g. quantized / bit-packed slabs)
need two seams:

  1. A torch storage dtype that the rest of the runner uses for
     bookkeeping (page sizes, allocators, etc.). Often ``torch.uint8``
     for raw byte-packed pools.
  2. A pool factory that constructs the actual ``KVCache`` subclass from
     the runner's already-resolved planning state (max tokens, page
     size, head count, layer count, device, ...).

This module exposes a small process-global registry keyed by the
``--kv-cache-dtype`` string. The configurator and pool selector consult
it before falling back to the built-in if/elif chain, so unrelated
dtype names behave exactly as before.

Typical use from a downstream plugin::

    import torch
    from sglang.srt.plugins.kv_cache import register

    def _build_my_pool(runner):
        from my_pkg.pool import MyPool
        return MyPool(
            size=runner.max_total_num_tokens,
            page_size=runner.page_size,
            ...,
        )

    register("mydtype", torch_dtype=torch.uint8, pool_factory=_build_my_pool)

After registration ``--kv-cache-dtype mydtype`` is accepted by argparse
(when the consumer also calls
:func:`sglang.srt.server_args.add_kv_cache_dtype_choices`) and routed
through the plugin's pool factory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)


class KVCacheDtypeSpec(NamedTuple):
    """Registered specification for a custom ``--kv-cache-dtype`` value."""

    torch_dtype: torch.dtype
    pool_factory: Callable[[Any], Any]


_REGISTRY: dict[str, KVCacheDtypeSpec] = {}


def register(
    name: str,
    *,
    torch_dtype: torch.dtype,
    pool_factory: Callable[[Any], Any],
) -> None:
    """Register a custom KV-cache dtype.

    Args:
        name: The string passed via ``--kv-cache-dtype``. Must not collide
            with a built-in name (``auto``, ``bfloat16``, ``bf16``,
            ``fp8_e4m3``, ``fp8_e5m2``, ``fp4_e2m1``).
        torch_dtype: The torch dtype the runner stores on
            ``self.kv_cache_dtype``. The dtype is passed to allocators
            and a few sizing helpers; it does not constrain the actual
            byte layout the pool uses.
        pool_factory: Callable ``(runner) -> KVCache`` that constructs
            and returns the pool. Invoked from ``_init_pools`` after the
            runner has resolved sizing / device / layer state.

    Idempotent: re-registering the same ``name`` with identical spec is
    a no-op; re-registering with a different spec overrides and emits a
    warning.
    """
    spec = KVCacheDtypeSpec(torch_dtype=torch_dtype, pool_factory=pool_factory)
    existing = _REGISTRY.get(name)
    if existing is not None and existing != spec:
        logger.warning(
            "Overriding existing kv_cache_dtype plugin %r (%s -> %s)",
            name,
            existing,
            spec,
        )
    _REGISTRY[name] = spec


def unregister(name: str) -> None:
    """Remove a previously-registered dtype. No-op if not registered.

    Provided for tests; production callers normally do not unregister.
    """
    _REGISTRY.pop(name, None)


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` was previously passed to :func:`register`."""
    return name in _REGISTRY


def get_spec(name: str) -> KVCacheDtypeSpec | None:
    """Return the full spec for ``name`` or ``None`` if not registered."""
    return _REGISTRY.get(name)


def get_torch_dtype(name: str) -> torch.dtype | None:
    """Return the torch dtype registered for ``name``, or ``None``."""
    spec = _REGISTRY.get(name)
    return spec.torch_dtype if spec is not None else None


def get_pool_factory(name: str) -> Callable[[Any], Any] | None:
    """Return the pool factory registered for ``name``, or ``None``."""
    spec = _REGISTRY.get(name)
    return spec.pool_factory if spec is not None else None


def registered_names() -> list[str]:
    """Return the list of currently-registered dtype names."""
    return list(_REGISTRY.keys())
