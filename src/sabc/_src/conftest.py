"""Shared pytest fixtures."""

import mlx.core as mx
import pytest


@pytest.fixture
def key() -> mx.array:
  """A deterministic MLX PRNG key."""
  return mx.random.key(0)
