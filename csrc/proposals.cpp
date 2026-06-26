#include "proposals.hpp"

namespace sabc {

mx::array de_propose(const mx::array& theta, const mx::array& inactive,
                     double gamma0, double sigma_gamma, const mx::array& key) {
  int B = theta.shape(0);
  int M = inactive.shape(0);

  mx::array keys = mx::random::split(key, 3);  // (3, 2) uint32
  mx::array k1 = mx::take(keys, 0, 0);         // (2,) key row
  mx::array k2 = mx::take(keys, 1, 0);
  mx::array k3 = mx::take(keys, 2, 0);

  // Partner indices: i1 in [0,M); i2 in [0,M-1) shifted to skip i1 (distinct).
  mx::array i1 = mx::random::randint(0, M, mx::Shape{B}, mx::int32, k1);
  mx::array i2 = mx::random::randint(0, M - 1, mx::Shape{B}, mx::int32, k2);
  mx::array bump = mx::astype(mx::greater_equal(i2, i1), mx::int32);
  i2 = mx::add(i2, bump);

  mx::array p1 = mx::take(inactive, i1, 0);  // (B, n_para)
  mx::array p2 = mx::take(inactive, i2, 0);

  mx::array noise = mx::random::normal(mx::Shape{B, 1}, k3);
  mx::array gamma = mx::multiply(
      mx::array(static_cast<float>(gamma0)),
      mx::add(mx::array(1.0f),
              mx::multiply(mx::array(static_cast<float>(sigma_gamma)), noise)));
  return mx::add(theta, mx::multiply(gamma, mx::subtract(p1, p2)));
}

}  // namespace sabc
