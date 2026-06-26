#include "distance.hpp"

#include <stdexcept>
#include <string>

namespace sabc {

mx::array f_dist(const Callback& simulator, const Callback& stats,
                 const mx::array& ss_obs, const mx::array& theta,
                 const std::string& distance, const mx::array* weights) {
  mx::array y = simulator(theta);
  mx::array ss = stats(y);
  mx::array d = mx::subtract(ss, ss_obs);  // broadcasts (B,n_stats)-(n_stats,)
  if (distance == "abs") {
    return mx::abs(d);
  }
  if (distance == "sq") {
    return mx::square(d);
  }
  if (distance == "weighted_sq") {
    if (weights == nullptr) {
      throw std::invalid_argument("weighted_sq requires weights.");
    }
    return mx::multiply(mx::square(d), *weights);
  }
  throw std::invalid_argument("Unknown distance: " + distance);
}

}  // namespace sabc
