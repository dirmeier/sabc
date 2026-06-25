import mlx.core as mx

from sabc import _core


def test_fdist_abs_matches_manual():
  ss_obs = mx.array([1.0, 2.0])

  def simulator(theta):
    return theta  # identity model: y == theta

  def stats(y):
    return y  # stats == y

  theta = mx.array([[1.5, 2.0], [0.0, 0.0]])
  rho = _core.f_dist(simulator, stats, ss_obs, theta, "abs")
  assert mx.allclose(rho, mx.array([[0.5, 0.0], [1.0, 2.0]])).item()


def test_fdist_sq_matches_manual():
  ss_obs = mx.array([0.0])
  rho = _core.f_dist(
    lambda t: t, lambda y: y, ss_obs, mx.array([[3.0]]), "sq"
  )
  assert mx.allclose(rho, mx.array([[9.0]])).item()
