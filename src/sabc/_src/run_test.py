import mlx.core as mx
import pytest

import sabc
from sabc import distributions as dist
from sabc._src.run import _run


def _gaussian_prior(dim=2, scale=3.0):
  return dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(dim), mx.ones(dim) * scale))
  )


def test_public_run_functional_recovers_mean():
  observed = mx.array([1.0, -1.0])

  def f_dist(theta, obs):
    return mx.abs(theta - obs)  # identity model, per-dimension distance

  post = sabc.run(
    f_dist,
    prior=_gaussian_prior(),
    observed=observed,
    n_particles=1000,
    n_simulation=60_000,
    schedule=sabc.SingleEps(v=1.0),
    key=mx.random.key(0),
  )
  mean = post.samples.mean(axis=0)
  assert mx.all(mx.abs(mean - observed) < 0.4).item()
  assert post.samples.shape == (1000, 2)


def test_public_run_scalar_vs_vector_distance():
  observed = mx.array([1.0, -1.0])

  def vector(theta, obs):
    return mx.abs(theta - obs)

  def scalar(theta, obs):
    return mx.sum(mx.abs(theta - obs), axis=-1, keepdims=True)

  kw = dict(
    prior=_gaussian_prior(), observed=observed, n_particles=1000,
    n_simulation=60_000, key=mx.random.key(0),
  )
  pv = sabc.run(vector, schedule=sabc.MultiEps(v=1.0), **kw)
  ps = sabc.run(scalar, schedule=sabc.SingleEps(v=1.0), **kw)
  assert pv.rho.shape == (1000, 2)  # vector: one column per statistic
  assert ps.rho.shape == (1000, 1)  # scalar: a single aggregated column
  assert mx.all(mx.abs(pv.samples.mean(axis=0) - observed) < 0.4).item()
  assert mx.all(mx.abs(ps.samples.mean(axis=0) - observed) < 0.4).item()


def test_public_run_is_deterministic():
  observed = mx.array([0.0])

  def f_dist(theta, obs):
    return mx.abs(theta - obs)

  kw = dict(
    prior=_gaussian_prior(dim=1, scale=1.0), observed=observed,
    n_particles=200, n_simulation=4000, schedule=sabc.SingleEps(v=1.0),
  )
  a = sabc.run(f_dist, key=mx.random.key(7), **kw)
  b = sabc.run(f_dist, key=mx.random.key(7), **kw)
  assert mx.allclose(a.samples, b.samples).item()


def test_functional_and_string_paths_agree():
  # With an identity model and the same key, the Python `abs` distance (run)
  # and the in-C++ `abs` distance (_run) consume the same RNG and must produce
  # identical populations.
  observed = mx.array([1.0, -1.0])
  kw = dict(
    prior=_gaussian_prior(), observed=observed, n_particles=500,
    n_simulation=20_000, schedule=sabc.SingleEps(v=1.0), key=mx.random.key(3),
  )
  func = sabc.run(lambda t, o: mx.abs(t - o), **kw)
  strg = _run(lambda t: t, distance="abs", **kw)
  assert mx.allclose(func.samples, strg.samples).item()


@pytest.mark.parametrize("schedule", [sabc.SingleEps(v=1.0), sabc.MultiEps(v=1.0)])
@pytest.mark.parametrize("scalar", [False, True])
def test_internal_run_string_2x2(schedule, scalar):
  observed = mx.array([1.0, -1.0])
  post = _run(
    lambda t: t,
    prior=_gaussian_prior(),
    observed=observed,
    distance="abs",
    scalar=scalar,
    n_particles=1000,
    n_simulation=60_000,
    schedule=schedule,
    key=mx.random.key(0),
  )
  assert post.rho.shape == (1000, 1 if scalar else 2)
  assert mx.all(mx.abs(post.samples.mean(axis=0) - observed) < 0.4).item()
