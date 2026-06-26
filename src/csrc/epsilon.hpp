#pragma once
#include "common.hpp"

namespace sabc {

// Solve eps^2 + v*eps^1.5 - u_bar^2 = 0 on (0, u_bar]. Returns 0 if u_bar<=0.
double epsilon_single(double u_bar, double v);

// Per-statistic epsilon vector from transformed distances u (B, n_stats).
// Returns a (n_stats,) float32 array.
mx::array epsilon_multi(const mx::array& u, double v);

}  // namespace sabc
