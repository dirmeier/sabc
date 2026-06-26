import numpy as np

import mlx.core as mx

from sabc import _core


def test_epsilon_single_satisfies_its_equation():
  for u_bar, v in [(0.05, 1.0), (0.2, 1.0), (0.49, 2.0)]:
    eps = _core.epsilon_single(u_bar, v)
    resid = eps ** 2 + v * eps ** 1.5 - u_bar ** 2
    assert abs(resid) < 1e-6
    assert 0.0 < eps <= u_bar + 1e-9


def test_epsilon_single_zero_for_tiny_ubar():
  assert _core.epsilon_single(0.0, 1.0) == 0.0


def test_epsilon_multi_shape_and_positive():
  rng = np.random.default_rng(1)
  u = mx.array((np.abs(rng.normal(size=(200, 3))) * 0.3).astype(np.float32))
  eps = np.asarray(_core.epsilon_multi(u, 1.0))
  assert eps.shape == (3,)
  assert np.all(eps > 0.0)
