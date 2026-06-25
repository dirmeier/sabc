#include "cdf.hpp"

#include <stdexcept>

namespace sabc {

// xs,ys: (K,), q: (B,) -> (B,). Linear interp, clamped outside the knot range.
static mx::array interp_1d(const mx::array& xs, const mx::array& ys,
                           const mx::array& q) {
  int K = xs.shape(0);
  int B = q.shape(0);
  // idx = count of knots <= q, clamped to [1, K-1]; interval [idx-1, idx].
  mx::array le = mx::less_equal(mx::reshape(xs, {1, K}),
                               mx::reshape(q, {B, 1}));   // (B, K) bool
  mx::array idx = mx::astype(mx::sum(le, 1, false), mx::int32);  // (B,) int
  idx = mx::clip(idx, mx::array(1), mx::array(K - 1));
  mx::array x1 = mx::take(xs, idx, 0);
  mx::array x0 = mx::take(xs, mx::subtract(idx, mx::array(1)), 0);
  mx::array y1 = mx::take(ys, idx, 0);
  mx::array y0 = mx::take(ys, mx::subtract(idx, mx::array(1)), 0);
  mx::array t = mx::divide(mx::subtract(q, x0), mx::subtract(x1, x0));
  t = mx::clip(t, mx::array(0.0f), mx::array(1.0f));
  return mx::add(y0, mx::multiply(t, mx::subtract(y1, y0)));
}

// Build the (values, probs) knot table for one statistic column `col` (B,).
// Sort, prepend a single 0.0, append a*max, uniform prob grid linspace(0,1,K).
//
// LIMITATION: the NumPy reference also drops exact zeros and de-duplicates
// knots (np.unique) before building the grid. We do not -- MLX lacks a `unique`
// op and needs fixed-shape tables. For continuous distance data (no exact zeros
// or ties) this matches the reference exactly; with zero or tied distances the
// knot count and probabilities diverge (errors ~0.1). SABC distances are
// continuous, so this holds in practice.
static void build_table_1d(const mx::array& col, float a, mx::array& values,
                           mx::array& probs) {
  int B = col.shape(0);
  mx::array sorted = mx::sort(col, 0);
  mx::array zero = mx::reshape(mx::array(0.0f), {1});
  mx::array maxv = mx::max(sorted, false);
  mx::array inflated = mx::reshape(mx::multiply(maxv, mx::array(a)), {1});
  values = mx::concatenate({zero, sorted, inflated}, 0);
  int K = B + 2;
  probs = mx::linspace(0.0, 1.0, K);
  mx::eval(values);
  mx::eval(probs);
}

CdfTables build_cdf(const mx::array& rho, float a) {
  if (rho.ndim() != 2) {
    throw std::invalid_argument("build_cdf expects a 2D (B, n_stats) array.");
  }
  int n_stats = rho.shape(1);
  CdfTables tables;
  tables.n_stats = n_stats;
  tables.values.reserve(n_stats);
  tables.probs.reserve(n_stats);
  for (int j = 0; j < n_stats; ++j) {
    mx::array col = mx::reshape(
        mx::slice(rho, {0, j}, {rho.shape(0), j + 1}), {rho.shape(0)});
    mx::array values = mx::array(0.0f);
    mx::array probs = mx::array(0.0f);
    build_table_1d(col, a, values, probs);
    tables.values.push_back(values);
    tables.probs.push_back(probs);
  }
  return tables;
}

mx::array cdf_eval(const CdfTables& tables, const mx::array& rho) {
  if (rho.ndim() != 2) {
    throw std::invalid_argument("cdf_eval expects a 2D (B, n_stats) array.");
  }
  if (rho.shape(1) != tables.n_stats) {
    throw std::invalid_argument("cdf_eval: stat-count mismatch.");
  }
  int B = rho.shape(0);
  std::vector<mx::array> cols;
  cols.reserve(tables.n_stats);
  for (int j = 0; j < tables.n_stats; ++j) {
    mx::array q =
        mx::reshape(mx::slice(rho, {0, j}, {B, j + 1}), {B});
    mx::array u = interp_1d(tables.values[j], tables.probs[j], q);
    cols.push_back(mx::reshape(u, {B, 1}));
  }
  mx::array out = mx::concatenate(cols, 1);
  mx::eval(out);
  return out;
}

}  // namespace sabc
