# sabc

[![ci](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sabc.svg?colorB=black&style=flat)](https://pypi.org/project/sabc/)

>  Simulated Annealing ABC in MLX

`sabc` performs likelihood-free Bayesian inference (Approximate Bayesian
Computation) via Simulated Annealing. The annealing loop — empirical-CDF
distance transform, importance resampling, epsilon schedule, and a
Differential-Evolution MCMC kernel — runs in a compiled `nanobind` extension
(`sabc._core`), while models, priors, and the entry point live in Python and
exchange `mlx.core.array`s with the C++ core zero-copy via DLPack.

## Quickstart

All you provide is a **prior**, the **observed data**, and a **distance
function** `f_dist(theta, observed)`. The distance function is yours to write —
it folds simulation, summary statistics, and the metric into one place — and it
returns either one distance per particle (*scalar*) or one per summary statistic
(*vector*). Here we infer the location of a 2-D model observed at `[1, -1]`:

```python
import mlx.core as mx

import sabc
from sabc import distributions as dist

observed = mx.array([1.0, -1.0])

prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(2), mx.ones(2) * 3.0))
)


def f_dist(theta, observed):
    # theta: (n_particles, n_para). Simulate, then return the distance(s).
    y = theta + mx.random.normal(theta.shape) * 0.1
    return mx.abs(y - observed)              # per-dimension (vector) distance


posterior = sabc.run(
    f_dist,
    prior=prior,
    observed=observed,
    n_particles=2000,
    n_simulation=200_000,
    schedule=sabc.SingleEps(v=1.0),
    key=mx.random.key(0),
)

print(posterior.samples.mean(axis=0))      # ~ [1, -1]
```

`sabc.run` returns a `Posterior` with `samples`, `u`, `rho`, and the
`epsilon_history` / `u_history` traces.

### Scalar vs vector distances, single vs multiple epsilon

The shape of what `f_dist` returns chooses the inference flavour, independently
of the epsilon schedule:

- return `(n_particles,)` / `(n_particles, 1)` for a single **scalar** distance,
- return `(n_particles, n_stats)` to keep one distance **per summary statistic**.

A per-statistic distance pairs naturally with `MultiEps`, which anneals one
temperature per statistic (`SingleEps` uses one shared temperature):

```python
def f_dist(theta, observed):
    y = simulate(theta)
    return mx.abs(summary(y) - observed)    # (n_particles, n_stats)

posterior = sabc.run(
    f_dist, prior=prior, observed=observed,
    schedule=sabc.MultiEps(v=1.0),          # one epsilon per statistic
)
```

### Priors

Priors are built from MLX-native, TFP-style distributions:

```python
prior = dist.JointDistributionNamed(dict(
    a=dist.Normal(mx.zeros(1), mx.ones(1)),
    b=lambda a: dist.Normal(a, mx.ones(1)),   # p(b | a)
))
```

## Examples

Self-contained examples are in [examples](./examples).

## Installation

`sabc` targets Apple Silicon (Metal); Linux support is experimental. It requires
Python ≥ 3.11. Install the released package from PyPI:

```bash
pip install sabc
```

For a development checkout, use [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/dirmeier/sabc && cd sabc
uv sync --all-extras
```

## Development

The project uses `uv`, `scikit-build-core` (CMake/C++23), `ruff` for Python, and
`clang-format` + `cpplint` for C++. The `Makefile` wraps the common tasks:

```bash
make check    # cpp checks
make tests    # run the tests
make format   # autoformat
make lints    # lint
```

Install the git hooks once (the `commit-msg` type is needed for `gitlint`):

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The hooks then run automatically on changed files at commit time. To run every
hook against the whole repository (useful after adding a hook or before a PR):

```bash
uv run pre-commit run --all-files
```

### Linux build (Docker)

The CI builds the C++/MLX extension under `gcc` on Linux. Apple `clang` accepts
some MLX overloads that `gcc` rejects, so a dev container is provided to
reproduce and verify the Linux build locally without GitHub Actions:

```bash
docker build -t sabc-linux -f .devcontainer/Dockerfile .
docker run --rm -v "$PWD":/work -w /work sabc-linux \
  bash -lc "uv sync --all-extras && make tests"
```

## Citing SABC

If you find this package relevant to your research, please consider citing:

```
@article{albert2025simulated,
  title={Simulated Annealing ABC with multiple summary statistics},
  author={Albert, Carlo and Ulzega, Simone and Dirmeier, Simon and Scheidegger, Andreas and Bassi, Alberto and Mira, Antonietta},
  journal={arXiv preprint arXiv:2505.23261},
  year={2025}
}
```
