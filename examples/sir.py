"""Stochastic SIR inference with SABC."""

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import sabc
from sabc import distributions as dist

S0, I0, R0 = 99, 1, 0
N = S0 + I0 + R0
T_MAX = 160.0
MAX_EVENTS = 250
THETA_TRUE = mx.array([0.3, 0.1])  # (beta, gamma)
N_PARTICLES = 1000
N_SIMULATION = 100_000


def simulate(theta: mx.array) -> mx.array:
  """Batched Gillespie SIR.

  Args:
    theta: shape ``(B, 2)`` with columns ``(beta, gamma)``.

  Returns:
    Summaries ``(B, 3)`` = ``(total_infected, peak_infected, t_peak)``.
  """
  b = theta.shape[0]
  beta, gamma = theta[:, 0], theta[:, 1]
  s = mx.full((b,), float(S0))
  i = mx.full((b,), float(I0))
  r = mx.zeros((b,))
  t = mx.zeros((b,))
  peak_i = i
  t_peak = mx.zeros((b,))
  for _ in range(MAX_EVENTS):
    alive = mx.logical_and(i > 0, t < T_MAX)
    inf_rate = beta * s * i / N
    rec_rate = gamma * i
    total = inf_rate + rec_rate
    total_safe = mx.where(total > 0, total, 1.0)
    dt = -mx.log(mx.random.uniform(shape=(b,))) / total_safe
    is_inf = mx.random.uniform(shape=(b,)) < inf_rate / total_safe
    inf_ev = mx.logical_and(alive, is_inf)
    rec_ev = mx.logical_and(alive, mx.logical_not(is_inf))
    s = s + mx.where(inf_ev, -1.0, 0.0)
    i = i + mx.where(inf_ev, 1.0, 0.0) + mx.where(rec_ev, -1.0, 0.0)
    r = r + mx.where(rec_ev, 1.0, 0.0)
    t = t + mx.where(alive, dt, 0.0)
    newpeak = mx.logical_and(alive, i > peak_i)
    peak_i = mx.where(newpeak, i, peak_i)
    t_peak = mx.where(newpeak, t, t_peak)
  return mx.stack([r, peak_i, t_peak], axis=-1)


def f_dist(theta: mx.array, observed: mx.array) -> mx.array:
  """Per-statistic absolute distance to the observed summaries, ``(B, 3)``."""
  return mx.abs(simulate(theta) - observed)


def _plot(samples: np.ndarray, truth: np.ndarray) -> None:
  """Corner plot of the 2-D ``(beta, gamma)`` posterior (red = truth)."""
  labels = [r"$\beta$", r"$\gamma$"]
  fig, axes = plt.subplots(2, 2, figsize=(7, 7))
  for ii in range(2):
    for jj in range(2):
      ax = axes[ii, jj]
      if jj > ii:
        ax.axis("off")
        continue
      if ii == jj:
        ax.hist(samples[:, ii], bins=40, color="C0", density=True)
        ax.axvline(truth[ii], color="C3", lw=1.5)
      else:
        ax.scatter(samples[:, jj], samples[:, ii], s=5, alpha=0.15, color="C0")
        ax.plot(truth[jj], truth[ii], "*", color="C3", ms=12)
      if jj == 0:
        ax.set_ylabel(labels[ii])
      if ii == 1:
        ax.set_xlabel(labels[jj])
  fig.suptitle("SIR posterior (red = truth)")
  fig.tight_layout()
  plt.show()


def main() -> None:
  """Run SABC on the SIR model and show the posterior plot."""
  mx.random.seed(0)
  prior = dist.JointDistributionNamed(
    {"theta": dist.Uniform(mx.array([0.1, 0.05]), mx.array([1.0, 0.5]))}
  )
  observed = simulate(THETA_TRUE.reshape(1, 2)).reshape(-1)
  mx.eval(observed)

  post = sabc.run(
    f_dist,
    prior=prior,
    observed=observed,
    n_particles=N_PARTICLES,
    n_simulation=N_SIMULATION,
    schedule=sabc.MultiEps(v=1.0),
    key=mx.random.key(0),
  )

  samples = np.asarray(post.samples)
  truth = np.asarray(THETA_TRUE)
  _plot(samples, truth)


if __name__ == "__main__":
  main()
