#pragma once
#include "common.hpp"

namespace sabc {

/**
 * @brief Solve eps^2 + v*eps^1.5 - u_bar^2 = 0 on (0, u_bar].
 *
 * @param u_bar Mean transformed distance; returns 0 if u_bar <= 0.
 * @param v Annealing speed coefficient.
 * @return The annealing threshold epsilon.
 */
double epsilon_single(double u_bar, double v);

/**
 * @brief Per-statistic epsilon vector from transformed distances.
 *
 * @param u Transformed distances, shape (B, n_stats).
 * @param v Annealing speed coefficient.
 * @return Per-statistic epsilon, shape (n_stats,) float32.
 */
mx::array epsilon_multi(const mx::array& u, double v);

}  // namespace sabc
