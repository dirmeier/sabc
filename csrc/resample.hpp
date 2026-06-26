#pragma once
#include "common.hpp"

namespace sabc {

// Importance weights from transformed distances u (B, n_stats):
//   w_i = exp(-sum_j u_ij * delta / mean_over_particles(u_.j)).
mx::array resample_weights(const mx::array& u, float delta);

// Draw `size` resampling indices (size,) ~ Categorical(weights).
mx::array resample_indices(const mx::array& u, float delta, int size,
                           const mx::array& key);

// Effective sample size = (sum w)^2 / sum(w^2).
double resample_ess(const mx::array& u, float delta);

}  // namespace sabc
