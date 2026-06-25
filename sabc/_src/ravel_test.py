import mlx.core as mx

from sabc._src.distributions import JointDistributionNamed, Normal
from sabc._src.ravel import FlatPrior


def test_flatprior_rvs_returns_flat_matrix(key):
  jd = JointDistributionNamed(dict(theta=Normal(mx.zeros(2), mx.ones(2))))
  prior = FlatPrior(jd, key)
  pop = prior.rvs(key, size=100)
  assert pop.shape == (100, 2)


def test_flatprior_logpdf_matches_joint(key):
  jd = JointDistributionNamed(
    dict(a=Normal(mx.zeros(1), mx.ones(1)),
         b=Normal(mx.zeros(2), mx.ones(2)))
  )
  prior = FlatPrior(jd, key)
  flat = mx.zeros((5, 3))  # 1 + 2 event dims
  direct = jd.log_prob({"a": mx.zeros((5, 1)), "b": mx.zeros((5, 2))})
  assert mx.allclose(prior.logpdf(flat), direct).item()
