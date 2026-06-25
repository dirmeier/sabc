#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "common.hpp"
#include "conv.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "sabc MLX C++ core";
  m.def("ping", []() { return 42; });

  // Accepts a Python mlx.core.array via DLPack, doubles every element, and
  // returns a new mlx.core.array via DLPack.  Both directions go through the
  // shared conv.hpp helpers (no-op-deleter import + owning copy on the way in,
  // shared_ptr + capsule keep-alive on the way out).
  m.def("double", [](nb::object arr) -> sabc::MlxArray {
    mx::array x = sabc::to_mx(arr);
    return sabc::to_py(mx::multiply(x, mx::array(2.0f)));
  });

  // Invokes a Python callable (a simulator) with the supplied mlx.core.array
  // and returns the array it produced, proving the full C++ -> Python -> C++
  // round trip across the DLPack boundary.
  m.def("apply_callback",
        [](nb::callable fn, nb::object theta) -> sabc::MlxArray {
          auto cb = sabc::make_callback(fn);
          return sabc::to_py(cb(sabc::to_mx(theta)));
        });
}
