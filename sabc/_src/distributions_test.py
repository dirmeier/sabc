import math

import mlx.core as mx

from sabc._src.distributions import (
  JointDistributionNamed,
  Normal,
  Uniform,
)


def test_normal_sample_shape_and_logprob(key):
  d = Normal(mx.zeros(2), mx.ones(2))
  x = d.sample(key, (1000,))
  assert x.shape == (1000, 2)
  lp = d.log_prob(mx.zeros((1, 2)))
  # standard normal log pdf at 0 is -0.5*log(2*pi) per dim, summed over event
  expected = 2 * (-0.5 * math.log(2 * math.pi))
  assert abs(lp.item() - expected) < 1e-4


def test_uniform_logprob_inside_and_outside(key):
  d = Uniform(mx.array([0.0]), mx.array([2.0]))
  inside = d.log_prob(mx.array([[1.0]]))
  outside = d.log_prob(mx.array([[3.0]]))
  assert abs(inside.item() - math.log(0.5)) < 1e-5
  assert outside.item() == float("-inf")


def test_joint_named_sample_and_logprob(key):
  jd = JointDistributionNamed(
    dict(theta=Normal(mx.zeros(2), mx.ones(2))), batch_ndims=0
  )
  s = jd.sample(key, (4,))
  assert set(s.keys()) == {"theta"}
  assert s["theta"].shape == (4, 2)
  lp = jd.log_prob({"theta": mx.zeros((4, 2))})
  assert lp.shape == (4,)


def test_joint_named_two_factors_logprob_is_sum(key):
  jd = JointDistributionNamed(
    dict(a=Normal(mx.zeros(1), mx.ones(1)),
         b=Normal(mx.zeros(1), mx.ones(1)))
  )
  x = {"a": mx.zeros((3, 1)), "b": mx.zeros((3, 1))}
  na = Normal(mx.zeros(1), mx.ones(1)).log_prob(x["a"])
  nb_ = Normal(mx.zeros(1), mx.ones(1)).log_prob(x["b"])
  assert mx.allclose(jd.log_prob(x), na + nb_).item()
