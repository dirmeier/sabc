import mlx.core as mx
import numpy as np

from sabc import _core


def test_de_proposal_shape_and_moves():
  rng = np.random.default_rng(0)
  theta = mx.array(rng.normal(size=(16, 2)).astype(np.float32))
  inactive = mx.array(rng.normal(size=(16, 2)).astype(np.float32))
  key = mx.random.key(0)
  prop = _core.de_propose(theta, inactive, 0.5, 1e-5, key)
  assert prop.shape == (16, 2)
  # With gamma0=0.5 and tiny jitter, proposals differ from theta.
  assert not mx.allclose(prop, theta).item()
