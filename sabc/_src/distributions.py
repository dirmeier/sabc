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
