"""Annealing (epsilon) schedule selectors."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleEps:
  """Single shared epsilon. ``v`` is the annealing speed (> 0)."""

  v: float = 1.0
  algorithm: str = "single_eps"

  def __post_init__(self) -> None:
    """Validate v."""
    if self.v <= 0:
      raise ValueError(f"v must be positive, got {self.v}.")


@dataclass(frozen=True)
class MultiEps:
  """One epsilon per summary statistic. ``v`` is the annealing speed (> 0)."""

  v: float = 1.0
  algorithm: str = "multi_eps"

  def __post_init__(self) -> None:
    """Validate v."""
    if self.v <= 0:
      raise ValueError(f"v must be positive, got {self.v}.")
