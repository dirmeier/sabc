#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "common.hpp"
#include "conv.hpp"
#include "distance.hpp"

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

  // Runs the simulator/stats pipeline for `theta` and returns per-statistic
  // distances to `ss_obs` under the named metric.  Callables and arrays are
  // converted at this edge; sabc::f_dist itself stays nanobind-free.
  m.def("f_dist",
        [](nb::callable sim, nb::callable stats, nb::object ss_obs,
           nb::object theta, const std::string& dist) {
          return sabc::to_py(sabc::f_dist(sabc::make_callback(sim),
                                          sabc::make_callback(stats),
                                          sabc::to_mx(ss_obs),
                                          sabc::to_mx(theta), dist));
        });
}
