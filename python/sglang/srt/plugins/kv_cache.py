"""Plugin-facing API for registering custom KV-cache backends.

A plugin KV-cache binds three things together:
  - the string name accepted via ``--kv-cache-dtype``
  - the torch storage dtype for the underlying buffer
  - a factory ``(runner) -> token_to_kv_pool`` building the pool

Usage from a downstream plugin::

    import torch
    from sglang.srt.plugins.kv_cache import register
    from sglang.srt.server_args import add_kv_cache_dtype_choices

    def _build_my_pool(runner):
        from my_pkg.pool import MyPool
        return MyPool(runner)

    add_kv_cache_dtype_choices(["my_kv"])
    register("my_kv", torch_dtype=torch.uint8, pool_factory=_build_my_pool)

Then ``--kv-cache-dtype my_kv`` is accepted by argparse, the runner's
``self.kv_cache_dtype`` is set to ``torch.uint8`` (via
:meth:`ModelRunner.configure_kv_cache_dtype` consulting this registry),
and ``_init_pools`` constructs the pool by calling ``_build_my_pool(runner)``
instead of the built-in pool selection.

Symmetric with :mod:`sglang.srt.plugins.attention` — both registries
together let an out-of-tree compressed-KV backend (e.g. tqkv) plug in
without runtime monkey-patches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _Entry:
    torch_dtype: torch.dtype
    pool_factory: Callable[[Any], Any]


_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    *,
    torch_dtype: torch.dtype,
    pool_factory: Callable[[Any], Any],
) -> None:
    """Register a plugin KV-cache dtype.

    Args:
        name: The string accepted via ``--kv-cache-dtype``. Must also be
            added to argparse choices via
            :func:`sglang.srt.server_args.add_kv_cache_dtype_choices`.
        torch_dtype: The torch dtype the underlying pool buffer is
            allocated as. Compressed-KV plugins typically use
            ``torch.uint8``.
        pool_factory: ``(runner) -> token_to_kv_pool`` callable. Receives
            the partially-initialized :class:`ModelRunner` and returns
            the pool instance to assign to ``runner.token_to_kv_pool``.

    Idempotent — re-registering the same ``name`` overrides the
    previous entry.
    """
    _REGISTRY[name] = _Entry(torch_dtype=torch_dtype, pool_factory=pool_factory)


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is registered as a plugin KV-cache dtype."""
    return name in _REGISTRY


def registered_names() -> list[str]:
    """Return the currently-registered plugin KV-cache dtype names."""
    return list(_REGISTRY.keys())


def get_torch_dtype(name: str) -> torch.dtype:
    """Return the torch storage dtype registered for ``name``.

    Raises ``KeyError`` if not registered.
    """
    return _REGISTRY[name].torch_dtype


def build_pool(name: str, runner: Any) -> Any:
    """Invoke the registered ``pool_factory(runner)`` for ``name``.

    Raises ``KeyError`` if not registered.
    """
    return _REGISTRY[name].pool_factory(runner)


__all__ = [
    "register",
    "is_registered",
    "registered_names",
    "get_torch_dtype",
    "build_pool",
]
