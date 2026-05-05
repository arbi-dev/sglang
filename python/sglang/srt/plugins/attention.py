"""Plugin-facing API for registering custom attention backends.

Thin re-export of the existing
:mod:`sglang.srt.layers.attention.attention_registry` so out-of-tree
plugins have a single ``sglang.srt.plugins.*`` namespace to import from
(symmetric with :mod:`sglang.srt.plugins.kv_cache`).

Usage from a downstream plugin::

    from sglang.srt.plugins.attention import register

    def _build_my_backend(runner):
        from my_pkg.backend import MyAttnBackend
        return MyAttnBackend(runner)

    register("mybackend", _build_my_backend)

Then ``--attention-backend mybackend`` resolves to ``MyAttnBackend``.
The argparse ``choices=`` list for ``--attention-backend`` can be
extended via :func:`sglang.srt.server_args.add_attention_backend_choices`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    register_attention_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


def register(name: str, factory: Callable[[Any], "AttentionBackend"]) -> None:
    """Register an attention backend factory under ``name``.

    Equivalent to using :func:`register_attention_backend` as a
    decorator. Idempotent: re-registering the same ``name`` overrides
    the previous factory.

    Args:
        name: The string accepted via ``--attention-backend``.
        factory: ``(runner) -> AttentionBackend`` callable.
    """
    ATTENTION_BACKENDS[name] = factory


def is_registered(name: str) -> bool:
    """Return ``True`` if ``name`` is in the attention-backend registry."""
    return name in ATTENTION_BACKENDS


def registered_names() -> list[str]:
    """Return the currently-registered attention backend names."""
    return list(ATTENTION_BACKENDS.keys())


__all__ = [
    "register",
    "register_attention_backend",
    "is_registered",
    "registered_names",
    "ATTENTION_BACKENDS",
]
