"""Unit tests for the kv_cache and attention plugin registries.

These tests exercise the registry seams in isolation — they do not
spin up a real engine. They cover:

* :mod:`sglang.srt.plugins.kv_cache` register / lookup / unregister
* :mod:`sglang.srt.plugins.attention` register / lookup
* :func:`sglang.srt.server_args.add_kv_cache_dtype_choices` and
  :func:`add_attention_backend_choices` extending argparse choices
  so out-of-tree dtype / backend names parse successfully
* :meth:`ModelRunner.configure_kv_cache_dtype` consulting the registry
  for an unknown dtype string instead of raising
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# kv_cache plugin registry
# ---------------------------------------------------------------------------


def test_kv_cache_register_and_lookup():
    from sglang.srt.plugins import kv_cache as kv_plugin

    def _factory(_runner):
        return object()

    kv_plugin.register("__test_dtype__", torch_dtype=torch.uint8, pool_factory=_factory)
    try:
        assert kv_plugin.is_registered("__test_dtype__")
        assert kv_plugin.get_torch_dtype("__test_dtype__") is torch.uint8
        assert kv_plugin.get_pool_factory("__test_dtype__") is _factory
        assert "__test_dtype__" in kv_plugin.registered_names()
    finally:
        kv_plugin.unregister("__test_dtype__")
    assert not kv_plugin.is_registered("__test_dtype__")
    assert kv_plugin.get_torch_dtype("__test_dtype__") is None
    assert kv_plugin.get_pool_factory("__test_dtype__") is None


def test_kv_cache_register_idempotent_same_spec():
    from sglang.srt.plugins import kv_cache as kv_plugin

    def _factory(_runner):
        return object()

    kv_plugin.register("__test_dtype2__", torch_dtype=torch.uint8, pool_factory=_factory)
    # Re-registering identical spec should not raise.
    kv_plugin.register("__test_dtype2__", torch_dtype=torch.uint8, pool_factory=_factory)
    try:
        assert kv_plugin.is_registered("__test_dtype2__")
    finally:
        kv_plugin.unregister("__test_dtype2__")


# ---------------------------------------------------------------------------
# attention plugin registry
# ---------------------------------------------------------------------------


def test_attention_register_and_lookup():
    from sglang.srt.plugins import attention as attn_plugin

    def _factory(_runner):
        return object()

    name = "__test_attn_backend__"
    attn_plugin.register(name, _factory)
    try:
        assert attn_plugin.is_registered(name)
        assert name in attn_plugin.registered_names()
        assert attn_plugin.ATTENTION_BACKENDS[name] is _factory
    finally:
        attn_plugin.ATTENTION_BACKENDS.pop(name, None)
    assert not attn_plugin.is_registered(name)


# ---------------------------------------------------------------------------
# server_args argparse choice extension
# ---------------------------------------------------------------------------


def test_add_kv_cache_dtype_choices_accepts_extension():
    from sglang.srt.server_args import (
        KV_CACHE_DTYPE_CHOICES,
        add_kv_cache_dtype_choices,
    )

    name = "__test_argparse_kv__"
    assert name not in KV_CACHE_DTYPE_CHOICES
    add_kv_cache_dtype_choices([name])
    try:
        assert name in KV_CACHE_DTYPE_CHOICES
        # And argparse using the (mutated) module-level list accepts it.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--kv-cache-dtype", type=str, choices=KV_CACHE_DTYPE_CHOICES
        )
        ns = parser.parse_args(["--kv-cache-dtype", name])
        assert ns.kv_cache_dtype == name
    finally:
        KV_CACHE_DTYPE_CHOICES.remove(name)


def test_add_attention_backend_choices_accepts_extension():
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    name = "__test_argparse_attn__"
    assert name not in ATTENTION_BACKEND_CHOICES
    add_attention_backend_choices([name])
    try:
        assert name in ATTENTION_BACKEND_CHOICES
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--attention-backend", type=str, choices=ATTENTION_BACKEND_CHOICES
        )
        ns = parser.parse_args(["--attention-backend", name])
        assert ns.attention_backend == name
    finally:
        ATTENTION_BACKEND_CHOICES.remove(name)


# ---------------------------------------------------------------------------
# configure_kv_cache_dtype consults the registry
# ---------------------------------------------------------------------------


def test_configure_kv_cache_dtype_consults_plugin_registry():
    """``configure_kv_cache_dtype`` should resolve an unknown dtype via
    the plugin registry instead of raising ValueError."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.plugins import kv_cache as kv_plugin

    name = "__test_configure_dtype__"
    kv_plugin.register(
        name,
        torch_dtype=torch.uint8,
        pool_factory=lambda _runner: object(),
    )
    try:
        # Build a minimal stand-in for the runner — only attributes
        # touched by the plugin branch of configure_kv_cache_dtype are
        # populated.
        fake_runner = SimpleNamespace(
            server_args=SimpleNamespace(kv_cache_dtype=name),
            kv_cache_dtype=None,
            dtype=torch.bfloat16,
        )
        # Bind the unbound method and invoke it.
        ModelRunner.configure_kv_cache_dtype(fake_runner)
        assert fake_runner.kv_cache_dtype is torch.uint8
    finally:
        kv_plugin.unregister(name)


def test_configure_kv_cache_dtype_unknown_raises():
    """Truly unknown dtypes still raise (registry is the only escape hatch)."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    fake_runner = SimpleNamespace(
        server_args=SimpleNamespace(kv_cache_dtype="__definitely_not_registered__"),
        kv_cache_dtype=None,
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
        ModelRunner.configure_kv_cache_dtype(fake_runner)
