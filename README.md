# sabc

[![ci](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/sabc/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/sabc.svg?colorB=black&style=flat)](https://pypi.org/project/sabc/)

> Simulated Annealing ABC with multiple summary statistics, with the core
> algorithm implemented in MLX C++ and a thin, MLX-native Python API.

`sabc` performs likelihood-free Bayesian inference (Approximate Bayesian
Computation) via simulated annealing. The annealing loop — empirical-CDF
distance transform, importance resampling, epsilon schedule, and a
Differential-Evolution MCMC kernel — runs in a compiled `nanobind` extension
(`sabc._core`), while models, priors, and the entry point live in Python and
exchange `mlx.core.array`s with the C++ core zero-copy via DLPack.

It implements the thermodynamic, multiple-summary-statistics SABC of

> C. Albert, S. Ulzega, S. Dirmeier, A. Scheidegger, A. Bassi, A. Mira.
> *A thermodynamic approach to Approximate Bayesian Computation with multiple
> summary statistics.* arXiv:2505.23261 (2025).

building on the original SABC of Albert, Künsch & Scheidegger,
*Statistics and Computing* 25 (2015).

## Installation

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). MLX targets Apple
Silicon (Metal); Linux support is experimental.

```bash
uv sync --all-extras   # creates the venv and compiles sabc._core
```

Or, into an existing environment:

```bash
pip install .
```

## Quickstart

Infer the location of a 2-D model whose summary statistics are observed at
`[1, -1]`:

```python
import mlx.core as mx

import sabc
from sabc import distributions as dist

observed = mx.array([1.0, -1.0])

def simulator(theta):                 # theta: (B, n_para) -> data (B, ...)
    return theta + mx.random.normal(theta.shape) * 0.1

prior = dist.JointDistributionNamed(
    dict(theta=dist.Normal(mx.zeros(2), mx.ones(2) * 3.0))
)

post = sabc.run(
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

print(post.samples.mean(axis=0))      # ~ [1, -1]
```

`sabc.run` returns a `Posterior` with `samples`, `u`, `rho`, and the
`epsilon_history` / `u_history` traces.

### Priors

Priors are built from MLX-native, TFP-style distributions. `JointDistributionNamed`
supports both independent factors and **conditional** factors `p(b | a)` (a factor
may be a callable whose argument names refer to preceding factors):

```python
prior = dist.JointDistributionNamed(dict(
    a=dist.Normal(mx.zeros(1), mx.ones(1)),
    b=lambda a: dist.Normal(a, mx.ones(1)),   # p(b | a)
))
```

## Project layout

```
src/
  sabc/        Python package (public API + co-located *_test.py)
  csrc/        C++ sources for the sabc._core nanobind extension
examples/      runnable examples (gaussian.py)
CMakeLists.txt scikit-build-core + nanobind build of sabc._core
```

The split follows nanobind's recommended `src/` layout: the importable package
and the C++ translation units live under `src/`; only the package (with the
compiled `_core` installed into it) ships in the wheel.

## Development

The project uses `uv`, `scikit-build-core` (CMake/C++23), `ruff` for Python, and
`clang-format` + `cpplint` for C++. Common tasks are in the `Makefile`:

```bash
make build      # editable rebuild of the extension
make tests      # uv run pytest
make lints      # ruff (Python) + cpplint (C++)
make format     # ruff format/import-sort + clang-format
make check      # cppcheck static analysis (requires cppcheck)
make docs       # Doxygen API docs -> build/doxygen
make example    # run examples/gaussian.py
```

Install the git hooks once:

```bash
uv run pre-commit install
```

`pre-commit` runs `ruff`, `clang-format`, `cpplint`, and assorted hygiene checks,
and blocks direct commits to `main`.

## Contributing

1. Branch off `main` (direct commits to `main` are blocked by a pre-commit hook).
2. Make your change. Python lives in `src/sabc`, C++ in `src/csrc`.
   - Python: 2-space indent, 80 columns, Google-style docstrings (enforced by
     `ruff`).
   - C++: 2-space indent, 80 columns, Google `clang-format`; document public
     declarations with Doxygen `@brief`/`@param`/`@return`. The algorithm code in
     `src/csrc/*.{hpp,cpp}` stays free of `nanobind`; all Python interop is
     isolated in `bindings.cpp` and the `conv.hpp` DLPack helpers.
3. Validate locally: `make format && make lints && make tests`.
4. Commit with [Conventional Commits](https://www.conventionalcommits.org)
   (`feat:`, `fix:`, `refactor:`, `build:`, `docs:`, `chore:`); `gitlint` enforces
   the subject. Keep subjects ≤ 72 characters.
5. Open a PR. CI runs the pre-commit hooks and the build/test suite on macOS
   (Apple Silicon) and, best-effort, on Linux x64/arm64.

Correctness is validated through self-contained generative examples (running the
full inference on a model with a known posterior), not an external reference.

## License

Apache-2.0. See [LICENSE](LICENSE).
