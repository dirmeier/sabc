#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "common.hpp"

namespace nb = nanobind;

// Convenience alias for a 1-D float32 MLX ndarray (DLPack interop).
using MlxArray1D = nb::ndarray<nb::mlx, float, nb::ndim<1>>;

NB_MODULE(_core, m) {
  m.doc() = "sabc MLX C++ core";
  m.def("ping", []() { return 42; });

  // Accepts a Python mlx.core.array via DLPack, doubles every element, and
  // returns a new mlx.core.array via DLPack.  Both directions go through
  // nanobind's built-in MLX / DLPack bridge (framework id 8).
  m.def("double", [](MlxArray1D arr) -> MlxArray1D {
    // Build mx::array from the DLPack-imported data pointer.  Pass a no-op
    // deleter because nanobind owns the backing storage for the lifetime of
    // this call.
    const std::size_t n = arr.shape(0);
    mx::array x(
        arr.data(),
        {static_cast<int>(n)},
        mx::float32,
        [](void*) {});

    // Compute and materialise the result before the input ndarray goes out of
    // scope.
    auto result = std::make_shared<mx::array>(mx::multiply(x, mx::array(2.0f)));
    mx::eval(*result);

    // Store the shared_ptr on the heap so the capsule deleter can reach it.
    // The capsule keeps the mx::array — and its buffer — alive for as long as
    // the returned ndarray is alive.  nanobind will call mlx.core.array(o) on
    // export (framework 8), which copies the data into a proper Python array.
    auto* stored = new std::shared_ptr<mx::array>(result);
    nb::capsule keep_alive(
        stored,
        [](void* ptr) noexcept {
          delete static_cast<std::shared_ptr<mx::array>*>(ptr);
        });

    const std::size_t shape[1] = {n};
    return MlxArray1D(
        result->data<float>(),
        1,
        shape,
        keep_alive);
  });

  // Invokes a Python callable (a simulator) with the supplied mlx.core.array
  // and returns the array it produced.  The input is forwarded to Python as-is;
  // the callable's result is imported back into C++ as an mx::array via DLPack,
  // evaluated, and re-exported — proving the full C++ -> Python -> C++ round
  // trip across the DLPack boundary.
  m.def("apply_callback", [](nb::callable fn, nb::object theta) -> MlxArray1D {
    nb::object produced = fn(theta);

    // Import the Python mlx.core.array result into C++ via DLPack, exactly as
    // the "double" binding builds an mx::array from its ndarray argument.
    MlxArray1D out = nb::cast<MlxArray1D>(produced);
    const std::size_t n = out.shape(0);
    mx::array y(
        out.data(),
        {static_cast<int>(n)},
        mx::float32,
        [](void*) {});

    auto result = std::make_shared<mx::array>(y);
    mx::eval(*result);

    auto* stored = new std::shared_ptr<mx::array>(result);
    nb::capsule keep_alive(
        stored,
        [](void* ptr) noexcept {
          delete static_cast<std::shared_ptr<mx::array>*>(ptr);
        });

    const std::size_t shape[1] = {n};
    return MlxArray1D(
        result->data<float>(),
        1,
        shape,
        keep_alive);
  });
}
