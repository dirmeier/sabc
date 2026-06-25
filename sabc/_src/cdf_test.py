import sys

import mlx.core as mx
import numpy as np
import pytest

REF = "/Users/simon/PROJECTS/sabc/sabc-bad/src"


@pytest.fixture
def build_cdf_ref():
  if REF not in sys.path:
    sys.path.insert(0, REF)
  from simulated_annealing_abc.cdf_estimators import build_cdf

  return build_cdf


def test_cdf_eval_matches_reference(build_cdf_ref):
  from sabc import _core

  rng = np.random.default_rng(0)
  rho = np.abs(rng.normal(size=(500, 2))).astype(np.float32)
  ref = build_cdf_ref(rho.astype(np.float64))

  query = np.abs(rng.normal(size=(50, 2))).astype(np.float32)
  ref_u = np.asarray(ref(query.astype(np.float64)))

  tables = _core.build_cdf(mx.array(rho))
  got_u = np.asarray(_core.cdf_eval(tables, mx.array(query)))

  assert np.allclose(got_u, ref_u, atol=1e-3)
