#include "resample.hpp"

namespace sabc {

mx::array resample_weights(const mx::array& u, float delta) {
  mx::array u_bar = mx::mean(u, 0, true);  // (1, n_stats)
  u_bar = mx::maximum(u_bar, mx::array(1e-12f));
  mx::array scaled = mx::multiply(u, mx::divide(mx::array(delta), u_bar));
  return mx::exp(mx::negative(mx::sum(scaled, 1)));  // (B,)
}

mx::array resample_indices(const mx::array& u, float delta, int size,
                           const mx::array& key) {
  mx::array w = resample_weights(u, delta);
  mx::array logits = mx::log(w);  // (B,)
  mx::array idx = mx::random::categorical(logits, -1, mx::Shape{size}, key);
  mx::eval(idx);
  return idx;
}

double resample_ess(const mx::array& u, float delta) {
  mx::array w = resample_weights(u, delta);
  mx::array num = mx::square(mx::sum(w));
  mx::array den = mx::sum(mx::square(w));
  mx::array ess = mx::divide(num, den);
  mx::eval(ess);
  return ess.item<float>();
}

}  // namespace sabc
