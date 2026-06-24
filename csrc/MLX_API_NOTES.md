# MLX C++ API Notes

Environment verification for Task 1 (feat/mlx-core branch).
Date: 2026-06-24.

---

## 1. Python environment

| Item | Value |
|---|---|
| Python | 3.13.13 (CPython) |
| uv | 0.11.14 |
| mlx | 0.31.2 |
| mlx-metal | 0.31.2 |
| nanobind | 2.13.0 |
| Default device | `Device(gpu, 0)` |

Install command used:
```
uv venv --python 3.13 && source .venv/bin/activate
uv pip install mlx nanobind
```

---

## 2. MLX package layout

`mlx` is a namespace package (`mlx.__file__` is `None`).

Base path:
```
.venv/lib/python3.13/site-packages/mlx/
```

Key subdirectories:
- `include/mlx/`        — C++ headers (full set, see §3)
- `include/metal_cpp/`  — Metal C++ headers
- `lib/`                — `libmlx.dylib`, `mlx.metallib`, `libjaccl.dylib`
- `share/cmake/MLX/`    — CMake config files (see §4)

---

## 3. MLX C++ include directory

```
.venv/lib/python3.13/site-packages/mlx/include
```

Absolute path on this machine:
```
/Users/simon/PROJECTS/sabc/sabc-mlx/.worktrees/feat-mlx-core/.venv/lib/python3.13/site-packages/mlx/include
```

Key headers present under `mlx/`:
- `mlx.h`, `array.h`, `ops.h`, `random.h`, `transforms.h`
- `linalg.h`, `fft.h`, `einsum.h`, `fast.h`, `io.h`
- `dtype.h`, `device.h`, `stream.h`
- `compile.h`, `compile_impl.h`
- `graph_utils.h`, `primitives.h`
- `backend/` subdirectory (Metal, no-metal, common)

---

## 4. CMake configuration

### MLXConfig.cmake path

```
/Users/simon/PROJECTS/sabc/sabc-mlx/.worktrees/feat-mlx-core/.venv/lib/python3.13/site-packages/mlx/share/cmake/MLX/MLXConfig.cmake
```

All cmake files present:
- `MLXConfig.cmake`
- `MLXConfigVersion.cmake`
- `MLXTargets.cmake`
- `MLXTargets-release.cmake`
- `extension.cmake`   — provides `mlx_build_metallib()` macro

CMake config sets:
- `MLX_INCLUDE_DIRS` — includes both `mlx/include` and `mlx/include/metal_cpp`
- `MLX_LIBRARIES` = `mlx`
- `MLX_BUILD_ACCELERATE` = ON
- `MLX_BUILD_METAL` = ON
- `MLX_CXX_FLAGS` = `-DACCELERATE_NEW_LAPACK -D_METAL_`
- Target CXX standard: 17 (project can raise to 23)

How to consume in `find_package`:
```cmake
set(MLX_DIR "<venv>/lib/python3.13/site-packages/mlx/share/cmake/MLX")
find_package(MLX REQUIRED)
```

Note: `mlx.extension.py` also sets `MLX_DIR` env var for CMake builds via
`os.environ["MLX_DIR"] = str(mlx.__path__[0])` — the CMake search path must
point into `share/cmake/MLX/`, not the package root.

### nanobind cmake_dir

```
/Users/simon/PROJECTS/sabc/sabc-mlx/.worktrees/feat-mlx-core/.venv/lib/python3.13/site-packages/nanobind/cmake
```

---

## 5. C++ compiler

```
Apple clang version 21.0.0 (clang-2100.0.123.102)
Target: arm64-apple-darwin25.2.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

C++23 support: **CONFIRMED** (`-std=c++23` accepted, trivial program compiles and links).

---

## 6. Op availability

### 6.1 `mlx::core::searchsorted`

**NOT FOUND** in `mlx/ops.h` or any other header under `mlx/include/mlx/`.

Implication: the CDF resampling step must use a broadcast fallback
(e.g., `mx.cumsum` + threshold comparison + `mx.argmax`) instead of a native
`searchsorted`.

### 6.2 `mlx::core::random::categorical`

**FOUND** — three overloads in `mlx/random.h`:

```cpp
// Overload 1: explicit shape
MLX_API array categorical(
    const array& logits,
    int axis,
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

// Overload 2: num_samples
MLX_API array categorical(
    const array& logits_,
    int axis,
    int num_samples,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

// Overload 3: single sample (axis defaults to -1)
MLX_API array categorical(
    const array& logits,
    int axis = -1,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
```

Implication: direct categorical resampling is available — no inverse-CDF fallback needed.

---

## 7. Exact declarations for all target ops

### `mlx/ops.h`

```cpp
// linspace
MLX_API array linspace(
    double start,
    double stop,
    int num = 50,
    Dtype dtype = float32,
    StreamOrDevice s = {});

// slice (with strides)
MLX_API array slice(
    const array& a,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s = {});

// slice (stride-1)
MLX_API array slice(
    const array& a,
    Shape start,
    Shape stop,
    StreamOrDevice s = {});

// slice (dynamic start indices)
MLX_API array slice(
    const array& a,
    const array& start,
    std::vector<int> axes,
    Shape slice_size,
    StreamOrDevice s = {});

// slice_update (with strides)
MLX_API array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s = {});

// slice_update (stride-1)
MLX_API array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    StreamOrDevice s = {});

// sort
MLX_API array sort(const array& a, StreamOrDevice s = {});
MLX_API array sort(const array& a, int axis, StreamOrDevice s = {});

// argsort
MLX_API array argsort(const array& a, StreamOrDevice s = {});
MLX_API array argsort(const array& a, int axis, StreamOrDevice s = {});

// take
MLX_API array take(const array& a, int index, int axis, StreamOrDevice s = {});
MLX_API array take(const array& a, const array& indices, StreamOrDevice s = {});
MLX_API array take(const array& a, int index, StreamOrDevice s = {});

// take_along_axis
MLX_API array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});

// where
MLX_API array where(
    const array& condition,
    const array& x,
    const array& y,
    StreamOrDevice s = {});

// clip
MLX_API array clip(
    const array& a,
    const std::optional<array>& a_min = std::nullopt,
    const std::optional<array>& a_max = std::nullopt,
    StreamOrDevice s = {});

// isfinite
MLX_API array isfinite(const array& a, StreamOrDevice s = {});

// logical_and
MLX_API array logical_and(const array& a, const array& b, StreamOrDevice s = {});

// concatenate
MLX_API array concatenate(std::vector<array> arrays, StreamOrDevice s = {});

// reshape
MLX_API array reshape(const array& a, Shape shape, StreamOrDevice s = {});

// sum (axes variants also exist)
MLX_API array sum(const array& a, bool keepdims, StreamOrDevice s = {});

// mean (axes variants also exist)
MLX_API array mean(const array& a, bool keepdims, StreamOrDevice s = {});

// abs
MLX_API array abs(const array& a, StreamOrDevice s = {});

// square
MLX_API array square(const array& a, StreamOrDevice s = {});

// exp
MLX_API array exp(const array& a, StreamOrDevice s = {});

// log
MLX_API array log(const array& a, StreamOrDevice s = {});

// maximum
MLX_API array maximum(const array& a, const array& b, StreamOrDevice s = {});

// searchsorted — NOT PRESENT
```

### `mlx/random.h`

```cpp
// key
MLX_API array key(uint64_t seed);

// split (pair)
MLX_API std::pair<array, array> split(const array& key, StreamOrDevice s = {});

// split (num keys)
MLX_API array split(const array& key, int num, StreamOrDevice s = {});

// uniform (low/high arrays)
MLX_API array uniform(
    const array& low,
    const array& high,
    const Shape& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

// uniform (unit [0,1))
MLX_API array uniform(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

// normal
MLX_API array normal(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& loc,
    const std::optional<array>& scale,
    const std::optional<array>& key,
    StreamOrDevice s = {});

// randint
MLX_API array randint(
    const array& low,
    const array& high,
    const Shape& shape,
    Dtype dtype = int32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

// categorical (3 overloads — see §6.2 above)
```

---

## 8. Nanobind MLX array type_caster

Searched all `.h`/`.hpp` files under `mlx/include/` for strings `type_caster`
or `nanobind`.

**Result: NOT FOUND.**

The MLX pip wheel does NOT ship a nanobind type-caster for `mlx::core::array`.

Consequence: zero-copy Python↔C++ `mlx::core::array` interop cannot be achieved
via a pre-built caster. Options:

1. **Use the Python C API directly** — call into `mlx.core` via `PyObject*` and
   extract the buffer/DLPack pointer manually.
2. **Copy via DLPack** — MLX arrays export DLPack; nanobind has DLPack support
   (`nb::dlpack`). This avoids a write-back copy for read-only inputs.
3. **Write a thin type_caster** — round-trip through `mlx.core.array.__dlpack__`
   on the Python side and reconstruct in C++. Adds one Python call per
   boundary crossing.
4. **Call MLX entirely from Python** — keep the C++ extension for pure
   computation that does not pass `mlx::core::array` across the boundary,
   and marshal results as numpy/raw buffers.

This is a **critical finding** — investigate before proceeding to the build
step.

---

## 9. Summary table

| Check | Result |
|---|---|
| MLX version | 0.31.2 |
| Default device | `Device(gpu, 0)` (Metal GPU) |
| MLX C++ headers | PRESENT at `.venv/.../mlx/include/mlx/` |
| MLXConfig.cmake | PRESENT at `.venv/.../mlx/share/cmake/MLX/MLXConfig.cmake` |
| nanobind cmake_dir | `.venv/.../nanobind/cmake` |
| `searchsorted` | NOT PRESENT — use broadcast fallback |
| `random::categorical` | PRESENT (3 overloads) |
| `random::normal` | PRESENT |
| `random::uniform` | PRESENT |
| `random::randint` | PRESENT |
| `random::split` | PRESENT |
| `random::key` | PRESENT |
| `slice` / `slice_update` | PRESENT (multiple overloads) |
| `sort` / `argsort` | PRESENT |
| `take` / `take_along_axis` | PRESENT |
| `where` | PRESENT |
| `linspace` | PRESENT |
| `clip` | PRESENT |
| `isfinite` | PRESENT |
| `logical_and` | PRESENT |
| C++23 toolchain | CONFIRMED (Apple clang 21.0.0, `-std=c++23` OK) |
| nanobind array type_caster | NOT FOUND — interop mechanism needs investigation |
