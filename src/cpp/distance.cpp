#include "distance.hpp"

#include <stdexcept>
#include <string>

namespace sabc {

mx::array f_dist(const Callback& simulator, const Callback& stats,
                 const mx::array& ss_obs, const mx::array& theta,
                 const std::string& distance, bool scalar,
                 const mx::array* weights) {
  mx::array y = simulator(theta);
  mx::array ss = stats(y);
  mx::array d = mx::subtract(ss, ss_obs);  // broadcasts (B,n_stats)-(n_stats,)
  mx::array rho = d;
  if (distance == "abs") {
    rho = mx::abs(d);
  } else if (distance == "sq") {
    rho = mx::square(d);
  } else if (distance == "weighted_sq") {
    if (weights == nullptr) {
      throw std::invalid_argument("weighted_sq requires weights.");
    }
    rho = mx::multiply(mx::square(d), *weights);
  } else {
    throw std::invalid_argument("Unknown distance: " + distance);
  }
  // scalar: aggregate the per-statistic distances into one (Julia SABC's
  // single = sum(multi)); keepdims so the result stays (B, 1).
  if (scalar) {
    rho = mx::sum(rho, 1, true);
  }
  return rho;
}

}  // namespace sabc
