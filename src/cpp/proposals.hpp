#pragma once
#include "common.hpp"

namespace sabc {

/**
 * @brief Differential Evolution batch proposal.
 *
 * For each active particle, picks two distinct partners from @p inactive and
 * steps theta' = theta + gamma * (partner1 - partner2), with
 * gamma = gamma0 * (1 + sigma_gamma * N(0,1)).
 *
 * @param theta Active parameter batch, shape (B, n_para).
 * @param inactive Frozen parameter pool to draw partners from, (M, n_para).
 * @param gamma0 Base step size.
 * @param sigma_gamma Relative Gaussian jitter on the step size.
 * @param key MLX PRNG key.
 * @return Proposed parameter batch, shape (B, n_para).
 */
mx::array de_propose(const mx::array& theta, const mx::array& inactive,
                     double gamma0, double sigma_gamma, const mx::array& key);

}  // namespace sabc
