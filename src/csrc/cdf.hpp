#pragma once
#include <vector>

#include "common.hpp"

namespace sabc {

struct CdfTables {
  std::vector<mx::array> values;  // per-stat sorted knots, shape (K_j,)
  std::vector<mx::array> probs;   // per-stat probs in [0,1], shape (K_j,)
  int n_stats;
};

CdfTables build_cdf(const mx::array& rho, float a = 1.5f);
mx::array cdf_eval(const CdfTables& tables, const mx::array& rho);

}  // namespace sabc
