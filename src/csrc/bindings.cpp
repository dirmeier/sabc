#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "cdf.hpp"
#include "common.hpp"
#include "conv.hpp"
#include "distance.hpp"
#include "epsilon.hpp"
#include "proposals.hpp"
#include "resample.hpp"
#include "sabc.hpp"

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
  m.def("f_dist", [](nb::callable sim, nb::callable stats, nb::object ss_obs,
                     nb::object theta, const std::string& dist) {
    return sabc::to_py(
        sabc::f_dist(sabc::make_callback(sim), sabc::make_callback(stats),
                     sabc::to_mx(ss_obs), sabc::to_mx(theta), dist));
  });

  // Empirical CDF tables (one interpolation table per statistic) exposed as an
  // opaque handle; arrays cross the boundary via conv.hpp, cdf.cpp stays
  // nanobind-free.
  nb::class_<sabc::CdfTables>(m, "CdfTables");
  m.def("build_cdf",
        [](nb::object rho) { return sabc::build_cdf(sabc::to_mx(rho)); });
  m.def("cdf_eval", [](const sabc::CdfTables& t, nb::object rho) {
    return sabc::to_py(sabc::cdf_eval(t, sabc::to_mx(rho)));
  });

  // Importance resampling: draw indices ~ Categorical(weights) derived from
  // transformed distances.  The PRNG key is uint32, imported via to_mx_u32;
  // integer indices are cast to float32 for the float-only to_py exporter.
  m.def("resample_indices",
        [](nb::object u, float delta, int size, nb::object key) {
          mx::array idx = sabc::resample_indices(sabc::to_mx(u), delta, size,
                                                 sabc::to_mx_u32(key));
          return sabc::to_py(mx::astype(idx, mx::float32));
        });
  m.def("resample_ess", [](nb::object u, float delta) {
    return sabc::resample_ess(sabc::to_mx(u), delta);
  });

  // Annealing-threshold (epsilon) solvers.  epsilon_single works in doubles
  // (no array crossing); epsilon_multi converts via conv.hpp and stays
  // nanobind-free.
  m.def("epsilon_single", &sabc::epsilon_single);
  m.def("epsilon_multi", [](nb::object u, double v) {
    return sabc::to_py(sabc::epsilon_multi(sabc::to_mx(u), v));
  });

  // Differential Evolution proposal.  theta/inactive cross via to_mx; the PRNG
  // key is uint32 via to_mx_u32; de_propose itself stays nanobind-free.
  m.def("de_propose", [](nb::object theta, nb::object inactive, double g0,
                         double sg, nb::object key) {
    return sabc::to_py(sabc::de_propose(sabc::to_mx(theta),
                                        sabc::to_mx(inactive), g0, sg,
                                        sabc::to_mx_u32(key)));
  });

  // SABC core loop.  Result members are mx::array, exposed via converting
  // getters (def_ro would return a raw mx::array and fail to cross the
  // boundary).  Callables/arrays are converted at this edge; sabc::run stays
  // nanobind-free.  The prior rvs callback receives a uint32 PRNG key, so it
  // uses the uint32 export path (to_py_u32), never the float32 to_py.
  nb::class_<sabc::Result>(m, "Result")
      .def_prop_ro(
          "population",
          [](const sabc::Result& r) { return sabc::to_py(r.population); },
          nb::rv_policy::move)
      .def_prop_ro(
          "u", [](const sabc::Result& r) { return sabc::to_py(r.u); },
          nb::rv_policy::move)
      .def_prop_ro(
          "rho", [](const sabc::Result& r) { return sabc::to_py(r.rho); },
          nb::rv_policy::move)
      .def_prop_ro("epsilon_history",
                   [](const sabc::Result& r) {
                     nb::list out;
                     for (const auto& e : r.epsilon_history) {
                       out.append(sabc::to_py(e));
                     }
                     return out;
                   })
      .def_prop_ro("u_history", [](const sabc::Result& r) {
        nb::list out;
        for (const auto& e : r.u_history) out.append(sabc::to_py(e));
        return out;
      });

  m.def("run", [](nb::callable sim, nb::callable stats, nb::object ss_obs,
                  nb::callable rvs, nb::callable logpdf, int n_particles,
                  const std::string& algorithm, double v, double delta,
                  const std::string& distance, std::optional<double> gamma0,
                  double sigma_gamma, long n_simulation, nb::object key) {
    sabc::RunArgs a{sabc::make_callback(sim),
                    sabc::make_callback(stats),
                    sabc::to_mx(ss_obs),
                    {},
                    sabc::make_callback(logpdf),
                    n_particles,
                    algorithm,
                    v,
                    delta,
                    distance,
                    gamma0.value_or(-1.0),
                    sigma_gamma,
                    n_simulation,
                    sabc::to_mx_u32(key)};
    a.rvs = [rvs](const mx::array& k, int size) {
      return sabc::to_mx(rvs(sabc::to_py_u32(k), size));
    };
    return sabc::run(a);
  });
}
