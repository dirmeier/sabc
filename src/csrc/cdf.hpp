#pragma once
#include <vector>

#include "common.hpp"

namespace sabc {

/// @brief Per-statistic empirical-CDF interpolation tables.
struct CdfTables {
  std::vector<mx::array> values;  ///< Per-stat sorted distance knots, (K_j,).
  std::vector<mx::array> probs;   ///< Per-stat probabilities in [0,1], (K_j,).
  int n_stats;                    ///< Number of summary statistics.
};

/**
 * @brief Build per-statistic empirical-CDF interpolation tables.
 *
 * For each statistic column the distances are sorted, prepended with 0.0 and
 * appended with @p a times the column max, paired with a uniform probability
 * grid linspace(0, 1, K).
 *
 * @param rho Per-statistic distances, shape (B, n_stats).
 * @param a Inflation factor for the appended upper knot.
 * @return Interpolation tables, one (values, probs) pair per statistic.
 */
CdfTables build_cdf(const mx::array& rho, float a = 1.5f);

/**
 * @brief Map distances to CDF probabilities via per-statistic interpolation.
 *
 * @param tables Interpolation tables produced by build_cdf.
 * @param rho Per-statistic distances, shape (B, n_stats).
 * @return Transformed distances in [0, 1], shape (B, n_stats).
 */
mx::array cdf_eval(const CdfTables& tables, const mx::array& rho);

}  // namespace sabc
