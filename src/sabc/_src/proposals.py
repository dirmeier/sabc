"""Proposal configuration wrappers (config only; logic lives in C++)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffEvolution:
  """Differential Evolution proposal config.

  Args:
    gamma0: DE scale. If ``None``, the core uses ``2.38 / sqrt(2 * n_para)``.
    sigma_gamma: Multiplicative jitter on gamma.
  """

  gamma0: float | None = None
  sigma_gamma: float = 1e-5
