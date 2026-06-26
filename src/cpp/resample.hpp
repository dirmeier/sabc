#pragma once
#include "common.hpp"

namespace sabc {

/**
 * @brief Importance weights from transformed distances.
 *
 * Computes w_i = exp(-sum_j u_ij * delta / mean_over_particles(u_.j)).
 *
 * @param u Transformed distances, shape (B, n_stats).
 * @param delta Annealing temperature scaling the exponent.
 * @return Per-particle importance weights, shape (B,).
 */
mx::array resample_weights(const mx::array& u, float delta);

/**
 * @brief Draw resampling indices from a categorical over the weights.
 *
 * @param u Transformed distances, shape (B, n_stats).
 * @param delta Annealing temperature scaling the exponent.
 * @param size Number of indices to draw.
 * @param key MLX PRNG key.
 * @return Resampling indices, shape (size,) ~ Categorical(weights).
 */
mx::array resample_indices(const mx::array& u, float delta, int size,
                           const mx::array& key);

/**
 * @brief Effective sample size of the importance weights.
 *
 * Computes (sum w)^2 / sum(w^2).
 *
 * @param u Transformed distances, shape (B, n_stats).
 * @param delta Annealing temperature scaling the exponent.
 * @return Effective sample size as a scalar.
 */
double resample_ess(const mx::array& u, float delta);

}  // namespace sabc
