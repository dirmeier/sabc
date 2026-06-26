#include <cmath>

#include <doctest/doctest.h>
#include "proposals.hpp"

TEST_CASE("de_propose preserves theta shape") {
  mx::array theta({0.0f, 0.0f, 1.0f, 1.0f}, {2, 2});
  mx::array inactive({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  mx::array key = mx::random::key(0);

  mx::array out = sabc::de_propose(theta, inactive, 0.5, 0.1, key);
  CHECK(out.shape(0) == theta.shape(0));
  CHECK(out.shape(1) == theta.shape(1));
}

TEST_CASE("de_propose with gamma0>0 moves theta") {
  mx::array theta({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});
  mx::array inactive({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  mx::array key = mx::random::key(1);

  mx::array out = sabc::de_propose(theta, inactive, 0.8, 0.1, key);
  mx::array diff = mx::abs(mx::subtract(out, theta));
  mx::array total = mx::sum(diff, false);
  mx::eval(total);
  CHECK(total.item<float>() > 0.0f);
}

TEST_CASE("de_propose with gamma0=0 leaves theta unchanged") {
  mx::array theta({0.3f, 0.7f, 1.1f, 2.2f}, {2, 2});
  mx::array inactive({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  mx::array key = mx::random::key(2);

  mx::array out = sabc::de_propose(theta, inactive, 0.0, 0.1, key);
  mx::array diff = mx::abs(mx::subtract(out, theta));
  mx::array total = mx::sum(diff, false);
  mx::eval(total);
  CHECK(total.item<float>() == doctest::Approx(0.0f));
}
