import mlx.core as mx
import numpy as np

from sabc import _core


def test_resample_indices_favor_low_u():
  # Two particles; particle 0 has much lower u -> higher weight.
  # The weight formula w_i = exp(-sum_j u_ij * delta / mean(u_.j)) caps the
  # high-u particle's beta at 2*delta for two particles, so a clear majority
  # needs delta well above the 0.1 default; delta=2 gives p0 ~ 0.98.
  u = mx.array([[0.01], [5.0]])
  key = mx.random.key(0)
  idx = _core.resample_indices(u, 2.0, 10000, key)
  idx_np = np.asarray(idx)
  assert idx_np.shape == (10000,)
  # Particle 0 should dominate.
  assert (idx_np == 0).mean() > 0.9


def test_resample_ess_in_range():
  u = mx.array(np.abs(np.random.default_rng(0).normal(size=(100, 2))))
  ess = _core.resample_ess(mx.array(u), 0.1)
  assert 1.0 <= ess <= 100.0
