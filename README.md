# sabc

[![ci](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sabc.svg?colorB=black&style=flat)](https://pypi.org/project/sabc/)


`sabc` performs likelihood-free Bayesian inference (Approximate Bayesian
Computation) via Simulated Annealing. The annealing loop — empirical-CDF
distance transform, importance resampling, epsilon schedule, and a
Differential-Evolution MCMC kernel — runs in a compiled `nanobind` extension
(`sabc._core`), while models, priors, and the entry point live in Python and
exchange `mlx.core.array`s with the C++ core zero-copy via DLPack.

## Quickstart

Infer the location of a 2-D model whose summary statistics are observed at
`[1, -1]`:

```python
import mlx.core as mx

import sabc
from sabc import distributions as dist


def simulator(theta):
    return theta + mx.random.normal(theta.shape) * 0.1

observed = mx.array([1.0, -1.0])

prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(2), mx.ones(2) * 3.0))
)

posterior = sabc.run(
    simulator,
    prior=prior,
    observed=observed,
    n_particles=2000,
    n_simulation=200_000,
    schedule=sabc.SingleEps(v=1.0),
    proposal=sabc.DiffEvolution(),
    distance="abs",
    key=mx.random.key(0),
)

print(posterior.samples.mean(axis=0))      # ~ [1, -1]
```

`sabc.run` returns a `Posterior` with `samples`, `u`, `rho`, and the
`epsilon_history` / `u_history` traces.

### Priors

Priors are built from MLX-native, TFP-style distributions:

```python
prior = dist.JointDistributionNamed(dict(
    a=dist.Normal(mx.zeros(1), mx.ones(1)),
    b=lambda a: dist.Normal(a, mx.ones(1)),   # p(b | a)
))
```

## Examples

Self-contained examples can be found in [examples](./examples).

## Installation

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). MLX targets Apple
Silicon (Metal); Linux support is experimental.

```bash
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

## Author

Simon Dirmeier <a href="mailto:simd23 @ pm me">simd23 @ pm me</a>
