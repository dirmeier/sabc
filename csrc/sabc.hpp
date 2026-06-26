#pragma once
#include <functional>
#include <string>
#include <vector>

#include "cdf.hpp"
#include "common.hpp"
#include "distance.hpp"

namespace sabc {

using PriorRvs = std::function<mx::array(const mx::array&, int)>;  // (key,size)
using PriorLogpdf = std::function<mx::array(const mx::array&)>;   // (N,P)->(N,)

struct Result {
  mx::array population;
  mx::array u;
  mx::array rho;
  std::vector<mx::array> epsilon_history;
  std::vector<mx::array> u_history;
};

struct RunArgs {
  Callback simulator;
  Callback stats;
  mx::array ss_obs;
  PriorRvs rvs;
  PriorLogpdf logpdf;
  int n_particles;
  std::string algorithm;  // "single_eps" | "multi_eps"
  double v;
  double delta;
  std::string distance;   // "abs" | "sq" | "weighted_sq"
  double gamma0;          // <=0 means "use default 2.38/sqrt(2P)"
  double sigma_gamma;
  long n_simulation;
  mx::array key;
};

Result run(const RunArgs& args);

}  // namespace sabc
