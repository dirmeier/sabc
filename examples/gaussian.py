"""Infer a 2D Gaussian location with sabc-mlx."""

import mlx.core as mx

import sabc
from sabc import distributions as dist


def main() -> None:
  """Run a small Gaussian-location inference and print the posterior mean."""
  observed = mx.array([1.0, -1.0])

  def simulator(theta: mx.array) -> mx.array:
    return theta + mx.random.normal(theta.shape) * 0.1

  prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(2), mx.ones(2) * 3.0))
  )

  post = sabc.run(
    simulator,
    prior=prior,
    observed=observed,
    n_particles=2000,
    n_simulation=200_000,
    schedule=sabc.SingleEps(v=1.0),
    proposal=sabc.DiffEvolution(),
    distance="abs",
    key=mx.random.key(0),
  )
  print("observed:      ", observed)
  print("posterior mean:", post.samples.mean(axis=0))
  print("posterior std: ", post.samples.std(axis=0))


if __name__ == "__main__":
  main()
