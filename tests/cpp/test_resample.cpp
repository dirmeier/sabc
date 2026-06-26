#include <doctest/doctest.h>
#include "resample.hpp"

// Read element i of a 1D float32 array after forcing evaluation.
static float at1(const mx::array& a, int i) {
  mx::array x = a;
  mx::eval(x);
  return x.data<float>()[i];
}

TEST_CASE("resample_weights favors lower-distance particles") {
  // Two particles, one statistic. Smaller u -> larger weight.
  mx::array u({0.1f, 1.0f}, {2, 1});
  mx::array w = sabc::resample_weights(u, 1.0f);
  CHECK(w.shape(0) == 2);
  CHECK(at1(w, 0) > at1(w, 1));
}

TEST_CASE("resample_ess lies in [1, B]") {
  int B = 5;
  mx::array u({0.1f, 0.5f, 1.0f, 1.5f, 2.0f}, {5, 1});
  double ess = sabc::resample_ess(u, 1.0f);
  CHECK(ess >= 1.0);
  CHECK(ess <= static_cast<double>(B));
}

TEST_CASE("resample_ess is maximal for uniform weights") {
  // Identical distances -> equal weights -> ESS == B.
  int B = 4;
  mx::array u({0.7f, 0.7f, 0.7f, 0.7f}, {4, 1});
  double ess = sabc::resample_ess(u, 1.0f);
  CHECK(ess == doctest::Approx(static_cast<double>(B)));
}

TEST_CASE("resample_indices draws in-range indices of requested size") {
  mx::array u({0.1f, 0.5f, 1.0f}, {3, 1});
  mx::array key = mx::random::key(0);
  int size = 8;
  mx::array idx = sabc::resample_indices(u, 1.0f, size, key);
  CHECK(idx.shape(0) == size);
  // Cast to int32 so the read is dtype-agnostic across MLX versions.
  mx::array idx_i = mx::astype(idx, mx::int32);
  mx::eval(idx_i);
  const int32_t* p = idx_i.data<int32_t>();
  for (int i = 0; i < size; ++i) {
    CHECK(p[i] >= 0);
    CHECK(p[i] <= 2);
  }
}
