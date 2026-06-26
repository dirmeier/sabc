#pragma once
#include <functional>
#include <string>
#include <vector>

#include "cdf.hpp"
#include "common.hpp"
#include "distance.hpp"

namespace sabc {

/// @brief Samples a prior population from a key and size: (key, size)->(N, P).
using PriorRvs = std::function<mx::array(const mx::array&, int)>;
/// @brief Evaluates the prior log-density: (N, P)->(N,).
using PriorLogpdf = std::function<mx::array(const mx::array&)>;

/// @brief Final state and per-iteration history of an SABC run.
struct Result {
  mx::array population;  ///< Final parameter population, shape (N, P).
  mx::array u;           ///< Final transformed distances, shape (N, n_stats).
  mx::array rho;         ///< Final per-statistic distances, (N, n_stats).
  std::vector<mx::array> epsilon_history;  ///< Epsilon per iteration.
  std::vector<mx::array> u_history;  ///< Mean transformed distance per it.
};

/// @brief Configuration for a single SABC run.
struct RunArgs {
  Callback simulator;     ///< Maps a parameter batch to simulated data.
  Callback stats;         ///< Maps simulated data to summary statistics.
  mx::array ss_obs;       ///< Observed summary statistics, shape (n_stats,).
  PriorRvs rvs;           ///< Prior sampler, (key, size)->(N, P).
  PriorLogpdf logpdf;     ///< Prior log-density, (N, P)->(N,).
  int n_particles;        ///< Population size N.
  std::string algorithm;  ///< "single_eps" | "multi_eps".
  double v;               ///< Annealing speed coefficient.
  double delta;           ///< Resampling annealing temperature.
  std::string distance;   ///< "abs" | "sq" | "weighted_sq".
  double gamma0;          ///< Base step; <=0 means default 2.38/sqrt(2P).
  double sigma_gamma;     ///< Relative Gaussian jitter on the step size.
  long n_simulation;      ///< Total simulation budget.
  mx::array key;          ///< MLX PRNG key.
};

/**
 * @brief Run the Simulated Annealing ABC sampler.
 *
 * @param args Run configuration and callbacks.
 * @return Final population and per-iteration history.
 */
Result run(const RunArgs& args);

}  // namespace sabc
