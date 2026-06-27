import mlx.core as mx
import numpy as np

from sabc import _core


def test_core_run_concentrates_near_observed():
  ss_obs = mx.array([0.0, 0.0])

  def simulator(theta):
    return theta  # identity model

  def stats(y):
    return y

  def rvs(key, size):
    return mx.random.normal((size, 2), key=key) * 3.0

  def logpdf(theta):
    return mx.sum(-0.5 * (theta / 3.0) ** 2, axis=-1)

  result = _core.run_str(
    simulator, stats, ss_obs, rvs, logpdf,
    1000, "single_eps", 1.0, 0.1, "abs", False, None, 1e-5, 50_000,
    mx.random.key(0),
  )
  pop = np.asarray(result.population)
  assert pop.shape == (1000, 2)
  assert np.abs(pop.mean(axis=0)).max() < 0.5
  assert pop.std(axis=0).max() < 1.5
