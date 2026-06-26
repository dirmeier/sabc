#include "cdf.hpp"
#include <doctest/doctest.h>

// Read element (i,j) of a 2D float32 array after forcing evaluation.
static float at2(const mx::array& a, int i, int j) {
  mx::array x = a;
  mx::eval(x);
  int cols = x.shape(1);
  return x.data<float>()[i * cols + j];
}

TEST_CASE("cdf_eval stays in [0,1] and is monotone in the input") {
  // A single statistic column with strictly increasing positive distances.
  mx::array rho({0.5f, 1.0f, 2.0f, 4.0f}, {4, 1});
  sabc::CdfTables tables = sabc::build_cdf(rho);
  CHECK(tables.n_stats == 1);

  mx::array u = sabc::cdf_eval(tables, rho);
  CHECK(u.shape(0) == 4);
  CHECK(u.shape(1) == 1);

  float prev = -1.0f;
  for (int i = 0; i < 4; ++i) {
    float v = at2(u, i, 0);
    CHECK(v >= 0.0f);
    CHECK(v <= 1.0f);
    CHECK(v >= prev);  // monotone non-decreasing
    prev = v;
  }
}

TEST_CASE("cdf_eval clamps below the min to 0 and above the max to 1") {
  mx::array rho({1.0f, 2.0f, 3.0f, 4.0f}, {4, 1});
  sabc::CdfTables tables = sabc::build_cdf(rho);

  // Below the smallest knot (0.0) clamps to 0; far above the inflated upper
  // knot (1.5 * max) clamps to 1.
  mx::array probe({-5.0f, 100.0f}, {2, 1});
  mx::array u = sabc::cdf_eval(tables, probe);
  CHECK(at2(u, 0, 0) == doctest::Approx(0.0f));
  CHECK(at2(u, 1, 0) == doctest::Approx(1.0f));
}

TEST_CASE("cdf_eval handles multiple statistics independently") {
  mx::array rho({0.5f, 4.0f, 1.0f, 8.0f, 2.0f, 16.0f}, {3, 2});
  sabc::CdfTables tables = sabc::build_cdf(rho);
  CHECK(tables.n_stats == 2);

  mx::array u = sabc::cdf_eval(tables, rho);
  CHECK(u.shape(0) == 3);
  CHECK(u.shape(1) == 2);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      float v = at2(u, i, j);
      CHECK(v >= 0.0f);
      CHECK(v <= 1.0f);
    }
  }
}
