#pragma once
#include <functional>
#include <string>

#include "common.hpp"

namespace sabc {

/// @brief Maps an MLX array to an MLX array (simulator / stats / prior).
using Callback = std::function<mx::array(const mx::array&)>;

/**
 * @brief Per-statistic distances between simulated and observed summaries.
 *
 * Runs @p simulator on @p theta, reduces with @p stats, and compares to
 * @p ss_obs under the chosen metric.
 *
 * @param simulator Maps a parameter batch to simulated data.
 * @param stats Maps simulated data to summary statistics.
 * @param ss_obs Observed summary statistics, shape (n_stats,).
 * @param theta Parameter batch, shape (B, n_para).
 * @param distance One of "abs", "sq", "weighted_sq".
 * @param weights Per-statistic weights for "weighted_sq" (else nullptr).
 * @return Per-statistic distances, shape (B, n_stats).
 */
mx::array f_dist(const Callback& simulator, const Callback& stats,
                 const mx::array& ss_obs, const mx::array& theta,
                 const std::string& distance,
                 const mx::array* weights = nullptr);

}  // namespace sabc
