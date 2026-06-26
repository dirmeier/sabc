#pragma once
#include "common.hpp"

namespace sabc {

// Differential Evolution batch proposal. For each active particle, picks two
// distinct partners from `inactive` and steps:
//   theta' = theta + gamma * (partner1 - partner2),
// with gamma = gamma0 * (1 + sigma_gamma * N(0,1)). Returns (B, n_para).
mx::array de_propose(const mx::array& theta, const mx::array& inactive,
                     double gamma0, double sigma_gamma, const mx::array& key);

}  // namespace sabc
