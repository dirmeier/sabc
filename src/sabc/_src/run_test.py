import mlx.core as mx

import sabc
from sabc import distributions as dist


def test_public_run_recovers_gaussian_mean():
  observed = mx.array([1.0, -1.0])

  def simulator(theta):
    return theta  # identity model

  prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(2), mx.ones(2) * 3.0))
  )

  post = sabc.run(
    simulator,
    prior=prior,
    observed=observed,
    n_particles=1000,
    n_simulation=60_000,
    schedule=sabc.SingleEps(v=1.0),
    proposal=sabc.DiffEvolution(),
    distance="abs",
    key=mx.random.key(0),
  )
  mean = post.samples.mean(axis=0)
  assert mx.all(mx.abs(mean - observed) < 0.4).item()
  assert post.samples.shape == (1000, 2)


def test_public_run_is_deterministic():
  observed = mx.array([0.0])
  prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(1), mx.ones(1)))
  )
  kwargs = dict(
    prior=prior, observed=observed, n_particles=200, n_simulation=4000,
    schedule=sabc.SingleEps(v=1.0), proposal=sabc.DiffEvolution(),
    distance="abs",
  )
  a = sabc.run(lambda t: t, key=mx.random.key(7), **kwargs)
  b = sabc.run(lambda t: t, key=mx.random.key(7), **kwargs)
  assert mx.allclose(a.samples, b.samples).item()
