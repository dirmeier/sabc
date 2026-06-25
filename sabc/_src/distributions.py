"""MLX-native, TFP-style distributions (minimal subset for SABC priors)."""

import math

import mlx.core as mx

_LOG2PI = math.log(2.0 * math.pi)


class Normal:
  """Independent Normal over the last (event) axis.

  Args:
    loc: Mean, shape ``(event,)``.
    scale: Std-dev, scalar or shape ``(event,)``.
  """

  def __init__(self, loc: mx.array, scale: mx.array | float) -> None:
    self.loc = mx.array(loc)
    self.scale = mx.array(scale) * mx.ones_like(self.loc)
    self.event_size = int(self.loc.shape[-1])

  def sample(self, key: mx.array, sample_shape: tuple[int, ...]) -> mx.array:
    """Draw ``sample_shape + (event,)`` samples."""
    shape = (*sample_shape, self.event_size)
    return self.loc + self.scale * mx.random.normal(shape, key=key)

  def log_prob(self, x: mx.array) -> mx.array:
    """Log density summed over the event axis, shape ``x.shape[:-1]``."""
    z = (x - self.loc) / self.scale
    per_dim = -0.5 * (z * z) - mx.log(self.scale) - 0.5 * _LOG2PI
    return mx.sum(per_dim, axis=-1)


class Uniform:
  """Independent Uniform over the last (event) axis.

  Args:
    low: Lower bound, shape ``(event,)``.
    high: Upper bound, shape ``(event,)``.
  """

  def __init__(self, low: mx.array, high: mx.array) -> None:
    self.low = mx.array(low)
    self.high = mx.array(high)
    self.event_size = int(self.low.shape[-1])

  def sample(self, key: mx.array, sample_shape: tuple[int, ...]) -> mx.array:
    """Draw ``sample_shape + (event,)`` samples."""
    shape = (*sample_shape, self.event_size)
    return mx.random.uniform(
      low=self.low, high=self.high, shape=shape, key=key
    )

  def log_prob(self, x: mx.array) -> mx.array:
    """Log density summed over the event axis; ``-inf`` outside support."""
    inside = (x >= self.low) & (x <= self.high)
    density = -mx.log(self.high - self.low)
    per_dim = mx.where(inside, density, mx.array(float("-inf")))
    return mx.sum(per_dim, axis=-1)


class JointDistributionNamed:
  """Factorized joint distribution over a dict of named, independent parts.

  Mimics ``tfp.distributions.JointDistributionNamed`` for the factorized
  (non-conditional) case used by SABC priors.

  Args:
    distributions: Mapping name -> distribution (each with ``sample`` and
      ``log_prob``).
    batch_ndims: Accepted for TFP API parity; only ``0`` is supported.
  """

  def __init__(self, distributions: dict, batch_ndims: int = 0) -> None:
    if batch_ndims != 0:
      raise ValueError(f"Only batch_ndims=0 supported, got {batch_ndims}.")
    self.distributions = dict(distributions)

  def sample(self, key: mx.array, sample_shape: tuple[int, ...]) -> dict:
    """Sample each factor with an independent split key."""
    keys = mx.random.split(key, num=len(self.distributions))
    return {
      name: dist.sample(keys[i], sample_shape)
      for i, (name, dist) in enumerate(self.distributions.items())
    }

  def log_prob(self, value: dict) -> mx.array:
    """Sum the per-factor log densities."""
    total = None
    for name, dist in self.distributions.items():
      lp = dist.log_prob(value[name])
      total = lp if total is None else total + lp
    return total
