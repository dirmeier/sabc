"""Adapter presenting a named joint prior as a flat (N, n_para) prior."""

import mlx.core as mx

from sabc._src.distributions import JointDistributionNamed


class FlatPrior:
  """Wrap a JointDistributionNamed as a flat ``rvs``/``logpdf`` prior.

  Concatenates the named event leaves along the last axis (sorted by the
  distribution's insertion order) into one ``(N, n_para)`` matrix, and splits
  back when computing ``logpdf``.

  Args:
    joint: A ``JointDistributionNamed`` (or any object with matching
      ``sample``/``log_prob`` and a ``distributions`` mapping).
    probe_key: A key used once to probe per-leaf event sizes.
  """

  def __init__(
    self, joint: JointDistributionNamed, probe_key: mx.array
  ) -> None:
    self.joint = joint
    sample = joint.sample(probe_key, (1,))
    self.names = list(joint.distributions.keys())
    self.sizes = [int(sample[n].shape[-1]) for n in self.names]
    self.n_para = sum(self.sizes)

  def rvs(self, key: mx.array, size: int) -> mx.array:
    """Draw ``size`` flattened samples, shape ``(size, n_para)``."""
    sample = self.joint.sample(key, (size,))
    return mx.concatenate([sample[n] for n in self.names], axis=-1)

  def logpdf(self, theta: mx.array) -> mx.array:
    """Log prior of a flat batch, shape ``(theta.shape[0],)``."""
    named, start = {}, 0
    for name, size in zip(self.names, self.sizes, strict=True):
      named[name] = theta[:, start : start + size]
      start += size
    return self.joint.log_prob(named)
