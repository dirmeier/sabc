"""Public functional entry point for SABC."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from sabc import _core
from sabc._src.proposals import DiffEvolution
from sabc._src.ravel import FlatPrior
from sabc._src.schedule import MultiEps, SingleEps


@dataclass
class Posterior:
  """Result of an SABC run.

  Attributes:
    samples: Posterior population, shape ``(n_particles, n_para)``.
    u: Transformed distances, shape ``(n_particles, n_stats)``.
    rho: Raw distances, shape ``(n_particles, n_stats)``.
    epsilon_history: Per-update epsilon snapshots.
    u_history: Per-update mean transformed distances.
  """

  samples: mx.array
  u: mx.array
  rho: mx.array
  epsilon_history: list
  u_history: list


def run(  # noqa: PLR0913
  simulator: Callable,
  *,
  prior: object,
  observed: mx.array,
  n_particles: int = 1000,
  n_simulation: int = 100_000,
  schedule: SingleEps | MultiEps | None = None,
  proposal: DiffEvolution | None = None,
  distance: str = "abs",
  delta: float = 0.1,
  stats: Callable | None = None,
  key: mx.array | None = None,
) -> Posterior:
  """Run Simulated Annealing ABC.

  Args:
    simulator: ``theta:(B, n_para) -> y`` returning an ``mx.array``.
    prior: A ``JointDistributionNamed`` (raveled internally) or any object with
      ``rvs(key, size)`` and ``logpdf(theta)``.
    observed: Observed summary statistics, ``mx.array`` shape ``(n_stats,)``.
    n_particles: Population size.
    n_simulation: Total simulation budget.
    schedule: ``SingleEps`` (default) or ``MultiEps``.
    proposal: ``DiffEvolution`` (default).
    distance: ``"abs"``, ``"sq"``, or ``"weighted_sq"``.
    delta: Resampling parameter (positive).
    stats: Optional ``y -> ss`` summary function; identity if ``None``.
    key: MLX PRNG key; defaults to ``mx.random.key(0)``.

  Returns:
    A ``Posterior``.
  """
  schedule = schedule or SingleEps()
  proposal = proposal or DiffEvolution()
  key = key if key is not None else mx.random.key(0)
  stats = stats or (lambda y: y)

  if hasattr(prior, "rvs") and hasattr(prior, "logpdf"):
    flat = prior
  else:
    flat = FlatPrior(prior, key)

  result = _core.run(
    simulator,
    stats,
    mx.array(observed),
    flat.rvs,
    flat.logpdf,
    n_particles,
    schedule.algorithm,
    float(schedule.v),
    float(delta),
    distance,
    proposal.gamma0,
    float(proposal.sigma_gamma),
    int(n_simulation),
    key,
  )
  return Posterior(
    samples=result.population,
    u=result.u,
    rho=result.rho,
    epsilon_history=list(result.epsilon_history),
    u_history=list(result.u_history),
  )
