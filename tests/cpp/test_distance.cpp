#include "distance.hpp"
#include <doctest/doctest.h>

using sabc::Callback;

// Read element (i,j) of a 2D float32 array after forcing evaluation.
static float at2(const mx::array& a, int i, int j) {
  mx::array x = a;
  mx::eval(x);
  int cols = x.shape(1);
  return x.data<float>()[i * cols + j];
}

TEST_CASE("f_dist abs metric on identity sim/stats") {
  Callback id = [](const mx::array& x) { return x; };
  mx::array ss_obs({1.0f, 2.0f}, {2});
  mx::array theta({1.5f, 2.0f, 0.0f, 0.0f}, {2, 2});

  mx::array d = sabc::f_dist(id, id, ss_obs, theta, "abs");
  CHECK(d.shape(0) == 2);
  CHECK(d.shape(1) == 2);
  CHECK(at2(d, 0, 0) == doctest::Approx(0.5f));
  CHECK(at2(d, 0, 1) == doctest::Approx(0.0f));
  CHECK(at2(d, 1, 0) == doctest::Approx(1.0f));
  CHECK(at2(d, 1, 1) == doctest::Approx(2.0f));
}

TEST_CASE("f_dist sq metric") {
  Callback id = [](const mx::array& x) { return x; };
  mx::array ss_obs({0.0f}, {1});
  mx::array theta({3.0f}, {1, 1});

  mx::array d = sabc::f_dist(id, id, ss_obs, theta, "sq");
  CHECK(d.shape(0) == 1);
  CHECK(d.shape(1) == 1);
  CHECK(at2(d, 0, 0) == doctest::Approx(9.0f));
}

TEST_CASE("f_dist weighted_sq applies per-stat weights") {
  Callback id = [](const mx::array& x) { return x; };
  mx::array ss_obs({0.0f, 0.0f}, {2});
  mx::array theta({2.0f, 3.0f}, {1, 2});
  mx::array weights({0.5f, 2.0f}, {2});

  mx::array d = sabc::f_dist(id, id, ss_obs, theta, "weighted_sq", &weights);
  CHECK(at2(d, 0, 0) == doctest::Approx(2.0f));   // 0.5 * 4
  CHECK(at2(d, 0, 1) == doctest::Approx(18.0f));  // 2.0 * 9
}

TEST_CASE("f_dist rejects unknown metric and missing weights") {
  Callback id = [](const mx::array& x) { return x; };
  mx::array ss_obs({0.0f}, {1});
  mx::array theta({1.0f}, {1, 1});
  CHECK_THROWS(sabc::f_dist(id, id, ss_obs, theta, "nope"));
  CHECK_THROWS(sabc::f_dist(id, id, ss_obs, theta, "weighted_sq"));
}
