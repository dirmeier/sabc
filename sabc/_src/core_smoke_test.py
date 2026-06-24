"""Smoke tests for the compiled _core extension."""


def test_core_imports_and_reports_version():
  from sabc import _core

  assert _core.ping() == 42
