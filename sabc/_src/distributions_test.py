import math

import mlx.core as mx

from sabc._src.distributions import Normal, Uniform


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
