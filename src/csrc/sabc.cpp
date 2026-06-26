#include "sabc.hpp"

#include <cmath>
#include <string>

#include "epsilon.hpp"
#include "proposals.hpp"
#include "resample.hpp"

namespace sabc {

// Gather rows of `x` (N,*) at integer indices `idx` (M,).
static mx::array gather(const mx::array& x, const mx::array& idx) {
  return mx::take(x, idx, 0);
}

// Build a single-element (1,) float32 array from a scalar value.  The array
// init-list ctor is float-typed but does not take a Shape from a vector, so we
// reshape a scalar instead.
static mx::array scalar_1d(double value) {
  return mx::reshape(mx::array(static_cast<float>(value)), {1});
}

// Recompute the annealing threshold from the current transformed distances.
static mx::array compute_epsilon(const mx::array& u, const std::string& algo,
                                 double v) {
  if (algo == "multi_eps") {
    return epsilon_multi(u, v);
  }
  mx::array ubar = mx::mean(u);
  mx::eval(ubar);
  return scalar_1d(epsilon_single(ubar.item<float>(), v));
}

// Update one half-batch in place (functional): rewrites the active slice
// [lo, hi) of population/u/rho/logprior.  `inactive` is the frozen other half.
static void update_half(mx::array& population, mx::array& u, mx::array& rho,
                        mx::array& logprior, int lo, int hi,
                        const mx::array& inactive, const RunArgs& a,
                        const CdfTables& cdf, const mx::array& inv_eps,
                        double gamma0, mx::array key) {
  int B = hi - lo;
  mx::array theta_cur =
      mx::slice(population, {lo, 0}, {hi, population.shape(1)});
  mx::array u_cur = mx::slice(u, {lo, 0}, {hi, u.shape(1)});
  mx::array rho_cur = mx::slice(rho, {lo, 0}, {hi, rho.shape(1)});
  mx::array lp_cur = mx::slice(logprior, {lo}, {hi});

  mx::array keys = mx::random::split(key, 2);
  mx::array k_prop = mx::take(keys, mx::array(0), 0);
  mx::array k_acc = mx::take(keys, mx::array(1), 0);

  mx::array theta_prop =
      de_propose(theta_cur, inactive, gamma0, a.sigma_gamma, k_prop);
  mx::array lp_prop = a.logpdf(theta_prop);  // (B,)
  mx::array rho_prop =
      f_dist(a.simulator, a.stats, a.ss_obs, theta_prop, a.distance);
  mx::array u_prop = cdf_eval(cdf, rho_prop);  // (B, n_stats)

  // log acceptance = lp_prop - lp_cur + sum((u_cur - u_prop) * inv_eps).
  mx::array dterm =
      mx::sum(mx::multiply(mx::subtract(u_cur, u_prop), inv_eps), 1);
  mx::array log_acc = mx::add(mx::subtract(lp_prop, lp_cur), dterm);
  mx::array finite = mx::isfinite(lp_prop);
  mx::array uacc = mx::random::uniform(mx::Shape{B}, k_acc);
  mx::array accept =
      mx::logical_and(finite, mx::less(mx::log(uacc), log_acc));  // (B,)
  mx::array accept_col = mx::reshape(accept, {B, 1});

  mx::array new_theta = mx::where(accept_col, theta_prop, theta_cur);
  mx::array new_u = mx::where(accept_col, u_prop, u_cur);
  mx::array new_rho = mx::where(accept_col, rho_prop, rho_cur);
  mx::array new_lp = mx::where(accept, lp_prop, lp_cur);

  population = mx::slice_update(population, new_theta, {lo, 0},
                                {hi, population.shape(1)});
  u = mx::slice_update(u, new_u, {lo, 0}, {hi, u.shape(1)});
  rho = mx::slice_update(rho, new_rho, {lo, 0}, {hi, rho.shape(1)});
  logprior = mx::slice_update(logprior, new_lp, {lo}, {hi});
}

Result run(const RunArgs& a) {
  mx::array key = a.key;
  int N = a.n_particles;

  // --- init: draw prior, distances, CDF, resample, initial epsilon.
  mx::array k_init = mx::random::split(key, 2);
  mx::array k0 = mx::take(k_init, mx::array(0), 0);
  key = mx::take(k_init, mx::array(1), 0);

  mx::array population = a.rvs(k0, N);  // (N, P)
  int P = population.shape(1);
  mx::array rho =
      f_dist(a.simulator, a.stats, a.ss_obs, population, a.distance);
  mx::array logprior = a.logpdf(population);

  CdfTables cdf = build_cdf(rho);
  mx::array u = cdf_eval(cdf, rho);
  mx::eval(population);
  mx::eval(u);
  mx::eval(rho);
  mx::eval(logprior);

  // initial resample
  mx::array k_rs = mx::random::split(key, 2);
  mx::array k_rs0 = mx::take(k_rs, mx::array(0), 0);
  key = mx::take(k_rs, mx::array(1), 0);
  mx::array idx = resample_indices(u, a.delta, N, k_rs0);
  population = gather(population, idx);
  u = gather(u, idx);
  rho = gather(rho, idx);
  logprior = gather(logprior, idx);
  mx::eval(population);
  mx::eval(u);
  mx::eval(rho);
  mx::eval(logprior);

  double gamma0 = a.gamma0 > 0.0 ? a.gamma0 : 2.38 / std::sqrt(2.0 * P);

  mx::array epsilon = compute_epsilon(u, a.algorithm, a.v);

  Result res{population, u, rho, {}, {}};
  res.epsilon_history.push_back(epsilon);
  res.u_history.push_back(mx::mean(u, 0));

  long n_updates = a.n_simulation / N;
  int mid = N / 2;
  long n_accept = 0;
  int n_resampling = 1;
  long resample_every = 2L * N;

  for (long it = 0; it < n_updates; ++it) {
    mx::array inv_eps = mx::divide(mx::array(1.0f), epsilon);

    // half 1 uses half 2 as inactive, then half 2 sees the updated half 1.
    mx::array k_step = mx::random::split(key, 3);
    key = mx::take(k_step, mx::array(2), 0);
    mx::array inactive2 = mx::slice(population, {mid, 0}, {N, P});
    update_half(population, u, rho, logprior, 0, mid, inactive2, a, cdf,
                inv_eps, gamma0, mx::take(k_step, mx::array(0), 0));
    mx::array inactive1 = mx::slice(population, {0, 0}, {mid, P});
    update_half(population, u, rho, logprior, mid, N, inactive1, a, cdf,
                inv_eps, gamma0, mx::take(k_step, mx::array(1), 0));

    // bound the graph each iteration.
    mx::eval(population);
    mx::eval(u);
    mx::eval(rho);
    mx::eval(logprior);

    // resample on a fixed cadence (MVP proxy counter).
    n_accept += N;
    if (n_accept >= static_cast<long>(n_resampling + 1) * resample_every) {
      mx::array k_r = mx::random::split(key, 2);
      key = mx::take(k_r, mx::array(1), 0);
      mx::array ridx =
          resample_indices(u, a.delta, N, mx::take(k_r, mx::array(0), 0));
      population = gather(population, ridx);
      u = gather(u, ridx);
      rho = gather(rho, ridx);
      logprior = gather(logprior, ridx);
      mx::eval(population);
      mx::eval(u);
      mx::eval(rho);
      mx::eval(logprior);
      ++n_resampling;
    }

    epsilon = compute_epsilon(u, a.algorithm, a.v);
    res.epsilon_history.push_back(epsilon);
    res.u_history.push_back(mx::mean(u, 0));
  }

  res.population = population;
  res.u = u;
  res.rho = rho;
  return res;
}

}  // namespace sabc
