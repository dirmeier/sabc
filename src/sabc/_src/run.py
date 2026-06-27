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


def _flat_prior(prior: object, key: mx.array) -> object:
  """Return ``prior`` if it already exposes ``rvs``/``logpdf``, else ravel it."""
  if hasattr(prior, "rvs") and hasattr(prior, "logpdf"):
    return prior
  return FlatPrior(prior, key)


def _posterior(result: object) -> Posterior:
  """Wrap a ``_core`` result in a :class:`Posterior`."""
  return Posterior(
    samples=result.population,
    u=result.u,
    rho=result.rho,
    epsilon_history=list(result.epsilon_history),
    u_history=list(result.u_history),
  )


def run(  # noqa: PLR0913
  f_dist: Callable,
  *,
  prior: object,
  observed: object,
  n_particles: int = 1000,
  n_simulation: int = 100_000,
  schedule: SingleEps | MultiEps | None = None,
  proposal: DiffEvolution | None = None,
  delta: float = 0.1,
  key: mx.array | None = None,
) -> Posterior:
  """Run Simulated Annealing ABC with a user-defined distance function.

  The user supplies a single ``f_dist`` that folds simulation, summary
  statistics, and the distance metric into one function. Whether the inference
  is scalar or per-statistic (vector) is decided entirely by ``f_dist``: return
  one distance per particle for a scalar run, or several for a vector run. This
  is orthogonal to the epsilon ``schedule`` (``SingleEps`` vs ``MultiEps``).

  Args:
    f_dist: ``f_dist(theta, observed) -> rho`` where ``theta`` has shape
      ``(B, n_para)`` and ``rho`` is an ``mx.array`` of shape ``(B,)`` / ``(B,
      1)`` (scalar) or ``(B, n_stats)`` (vector), all non-negative.
    prior: A ``JointDistributionNamed`` (raveled internally) or any object with
      ``rvs(key, size)`` and ``logpdf(theta)``.
    observed: Observed data, passed through unchanged to ``f_dist``.
    n_particles: Population size.
    n_simulation: Total simulation budget.
    schedule: ``SingleEps`` (default) or ``MultiEps``.
    proposal: ``DiffEvolution`` (default).
    delta: Resampling parameter (positive).
    key: MLX PRNG key; defaults to ``mx.random.key(0)``.

  Returns:
    A ``Posterior``.
  """
  schedule = schedule or SingleEps()
  proposal = proposal or DiffEvolution()
  key = key if key is not None else mx.random.key(0)
  flat = _flat_prior(prior, key)

  def rho_fn(theta: mx.array) -> mx.array:
    rho = f_dist(theta, observed)
    return mx.reshape(rho, (theta.shape[0], -1))

  result = _core.run_fdist(
    rho_fn,
    flat.rvs,
    flat.logpdf,
    n_particles,
    schedule.algorithm,
    float(schedule.v),
    float(delta),
    proposal.gamma0,
    float(proposal.sigma_gamma),
    int(n_simulation),
    key,
  )
  return _posterior(result)


def _run(  # noqa: PLR0913
  simulator: Callable,
  *,
  prior: object,
  observed: mx.array,
  summary_fn: Callable | None = None,
  distance: str = "abs",
  scalar: bool = False,
  n_particles: int = 1000,
  n_simulation: int = 100_000,
  schedule: SingleEps | MultiEps | None = None,
  proposal: DiffEvolution | None = None,
  delta: float = 0.1,
  key: mx.array | None = None,
) -> Posterior:
  """Run SABC with the in-C++ string distance (internal; not exported).

  Equivalent to :func:`run` but computes the distance inside the C++ core from a
  ``simulator`` + ``summary_fn`` pipeline and a named ``distance`` metric. Kept
  for benchmarking the in-C++ distance against the functional callback path.

  Args:
    simulator: ``theta:(B, n_para) -> y`` returning an ``mx.array``.
    prior: As in :func:`run`.
    observed: Observed summary statistics, ``mx.array`` shape ``(n_stats,)``.
    summary_fn: Optional ``y -> ss`` summary function; identity if ``None``.
    distance: ``"abs"``, ``"sq"``, or ``"weighted_sq"`` (per-statistic).
    scalar: If ``True``, sum the per-statistic distances into a single value
      (``n_stats == 1``); otherwise keep them per-statistic.
    n_particles: Population size.
    n_simulation: Total simulation budget.
    schedule: ``SingleEps`` (default) or ``MultiEps``.
    proposal: ``DiffEvolution`` (default).
    delta: Resampling parameter (positive).
    key: MLX PRNG key; defaults to ``mx.random.key(0)``.

  Returns:
    A ``Posterior``.
  """
  schedule = schedule or SingleEps()
  proposal = proposal or DiffEvolution()
  key = key if key is not None else mx.random.key(0)
  summary_fn = summary_fn or (lambda y: y)
  flat = _flat_prior(prior, key)

  result = _core.run_str(
    simulator,
    summary_fn,
    mx.array(observed),
    flat.rvs,
    flat.logpdf,
    n_particles,
    schedule.algorithm,
    float(schedule.v),
    float(delta),
    distance,
    bool(scalar),
    proposal.gamma0,
    float(proposal.sigma_gamma),
    int(n_simulation),
    key,
  )
  return _posterior(result)
