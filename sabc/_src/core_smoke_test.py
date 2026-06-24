"""Smoke tests for the compiled _core extension."""


def test_core_imports_and_reports_version():
  from sabc import _core

  assert _core.ping() == 42


def test_core_doubles_an_mlx_array():
  import mlx.core as mx

  from sabc import _core

  x = mx.array([1.0, 2.0, 3.0])
  y = _core.double(x)
  assert mx.allclose(y, mx.array([2.0, 4.0, 6.0])).item()
