"""MLX-native, TFP-style distributions (minimal subset for SABC priors)."""

import inspect
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
  """Joint distribution over a dict of named factors.

  Mimics ``tfp.distributions.JointDistributionNamed``. Each factor is
  either an independent marginal (a distribution instance) or a conditional
  factor (a callable returning a distribution, whose parameter names refer
  to preceding factor names). Factors are resolved in insertion order, so a
  callable may only depend on names defined earlier.

  Example::

    JointDistributionNamed(dict(
      a=Normal(mx.zeros(1), mx.ones(1)),
      b=lambda a: Normal(a, mx.ones(1)),  # p(b | a)
    ))

  Args:
    distributions: Mapping name -> distribution instance or callable
      returning a distribution.
    batch_ndims: Accepted for TFP API parity; only ``0`` is supported.
  """

  def __init__(self, distributions: dict, batch_ndims: int = 0) -> None:
    if batch_ndims != 0:
      raise ValueError(f"Only batch_ndims=0 supported, got {batch_ndims}.")
    self.distributions = dict(distributions)

  @staticmethod
  def _resolve(factor: object, available: dict) -> object:
    """Resolve a factor to a distribution against available values.

    If ``factor`` is callable, its signature parameter names are looked up
    in ``available`` and passed as keyword arguments; otherwise the factor
    is already a distribution instance and is returned unchanged.

    Args:
      factor: A distribution instance or a callable returning one.
      available: Mapping of preceding factor names to values.

    Returns:
      A distribution instance with ``sample`` and ``log_prob``.
    """
    if callable(factor):
      names = inspect.signature(factor).parameters
      return factor(**{name: available[name] for name in names})
    return factor

  def sample(self, key: mx.array, sample_shape: tuple[int, ...]) -> dict:
    """Sample each factor in order, conditioning on preceding draws."""
    keys = mx.random.split(key, num=len(self.distributions))
    sampled: dict = {}
    for i, (name, factor) in enumerate(self.distributions.items()):
      dist = self._resolve(factor, sampled)
      sampled[name] = dist.sample(keys[i], sample_shape)
    return sampled

  def log_prob(self, value: dict) -> mx.array:
    """Sum the per-factor log densities, conditioning on parent values."""
    total = None
    for name, factor in self.distributions.items():
      dist = self._resolve(factor, value)
      lp = dist.log_prob(value[name])
      total = lp if total is None else total + lp
    return total
