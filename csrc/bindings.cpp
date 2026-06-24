#include <nanobind/nanobind.h>

#include "common.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "sabc MLX C++ core";
  m.def("ping", []() { return 42; });
}
