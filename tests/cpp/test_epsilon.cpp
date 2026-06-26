#include <cmath>

#include <doctest/doctest.h>
#include "epsilon.hpp"

// Read element i of a 1D float32 array after forcing evaluation.
static float at1(const mx::array& a, int i) {
  mx::array x = a;
  mx::eval(x);
  return x.data<float>()[i];
}

TEST_CASE("epsilon_single solves eps^2 + v*eps^1.5 - u_bar^2 = 0") {
  double u_bar = 0.8;
  double v = 0.5;
  double eps = sabc::epsilon_single(u_bar, v);
  CHECK(eps > 0.0);
  CHECK(eps <= u_bar);
  double residual =
      eps * eps + v * std::pow(eps, 1.5) - u_bar * u_bar;
  CHECK(residual == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("epsilon_single returns 0 for non-positive u_bar") {
  CHECK(sabc::epsilon_single(0.0, 1.0) == doctest::Approx(0.0));
  CHECK(sabc::epsilon_single(-1.0, 1.0) == doctest::Approx(0.0));
}

TEST_CASE("epsilon_single with v=0 gives eps = u_bar") {
  double u_bar = 0.6;
  double eps = sabc::epsilon_single(u_bar, 0.0);
  CHECK(eps == doctest::Approx(u_bar));
}

TEST_CASE("epsilon_multi returns one positive value per statistic") {
  mx::array u({0.1f, 0.2f, 0.3f, 0.4f, 0.2f, 0.1f}, {3, 2});
  mx::array eps = sabc::epsilon_multi(u, 1.0);
  CHECK(eps.ndim() == 1);
  CHECK(eps.shape(0) == 2);
  for (int j = 0; j < 2; ++j) {
    CHECK(at1(eps, j) > 0.0f);
  }
}
