#pragma once
#include <functional>
#include <string>

#include "common.hpp"

namespace sabc {

using Callback = std::function<mx::array(const mx::array&)>;

// Returns per-statistic distances, shape (B, n_stats). `distance` is one of
// "abs", "sq", "weighted_sq" (weighted_sq requires `weights`).
mx::array f_dist(const Callback& simulator, const Callback& stats,
                 const mx::array& ss_obs, const mx::array& theta,
                 const std::string& distance,
                 const mx::array* weights = nullptr);

}  // namespace sabc
