#include "epsilon.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace sabc {

// Brent's method root finder on [a, b]; requires f(a)*f(b) <= 0.
static double brentq(const std::function<double(double)>& f, double a, double b,
                     double tol = 1e-12, int max_iter = 200) {
  double fa = f(a);
  double fb = f(b);
  if (fa * fb > 0.0) {
    throw std::runtime_error("brentq: root not bracketed");
  }
  if (std::abs(fa) < std::abs(fb)) {
    std::swap(a, b);
    std::swap(fa, fb);
  }
  double c = a;
  double fc = fa;
  bool mflag = true;
  double d = 0.0;
  for (int iter = 0; iter < max_iter; ++iter) {
    if (fb == 0.0 || std::abs(b - a) < tol) {
      return b;
    }
    double s;
    if (fa != fc && fb != fc) {
      s = a * fb * fc / ((fa - fb) * (fa - fc)) +
          b * fa * fc / ((fb - fa) * (fb - fc)) +
          c * fa * fb / ((fc - fa) * (fc - fb));
    } else {
      s = b - fb * (b - a) / (fb - fa);
    }
    double lo = (3.0 * a + b) / 4.0;
    bool cond1 = !((s > std::min(lo, b)) && (s < std::max(lo, b)));
    bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
    bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
    bool cond4 = mflag && std::abs(b - c) < tol;
    bool cond5 = !mflag && std::abs(c - d) < tol;
    if (cond1 || cond2 || cond3 || cond4 || cond5) {
      s = (a + b) / 2.0;
      mflag = true;
    } else {
      mflag = false;
    }
    double fs = f(s);
    d = c;
    c = b;
    fc = fb;
    if (fa * fs < 0.0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }
    if (std::abs(fa) < std::abs(fb)) {
      std::swap(a, b);
      std::swap(fa, fb);
    }
  }
  return b;
}

double epsilon_single(double u_bar, double v) {
  if (u_bar <= 1e-12) return 0.0;
  auto f = [u_bar, v](double eps) {
    return eps * eps + v * std::pow(eps, 1.5) - u_bar * u_bar;
  };
  return brentq(f, 0.0, u_bar);
}

mx::array epsilon_multi(const mx::array& u, double v) {
  int n = static_cast<int>(u.shape(1));
  mx::array u_bar_arr = mx::mean(u, 0, false);  // (n_stats,)
  u_bar_arr = mx::maximum(u_bar_arr, mx::array(1e-12f));
  mx::eval(u_bar_arr);
  const float* ub_ptr = u_bar_arr.data<float>();
  std::vector<double> u_bar(n);
  for (int j = 0; j < n; ++j) {
    u_bar[j] = static_cast<double>(ub_ptr[j]);
  }

  // cn = (2n+2)! / ((n+1)! (n+2)!).  Computed as C(2n+2, n+1) / (n+2) via a
  // ratio product to avoid factorial overflow.
  double cn = 1.0;
  for (int k = 1; k <= n + 1; ++k) {
    cn *= static_cast<double>(n + 1 + k) / static_cast<double>(k);
  }
  cn /= static_cast<double>(n + 2);

  std::vector<float> out(n);
  for (int i = 0; i < n; ++i) {
    double ub = u_bar[i];
    if (ub >= 0.5) ub = 0.5 - 1e-9;
    auto g = [ub, n](double beta) {
      double val;
      if (beta < 1e-6) {
        val = 0.5 - beta / 12.0;
      } else {
        double e = std::exp(-beta);
        val = (1.0 - e * (1.0 + beta)) / (beta * (1.0 - e));
      }
      return val - ub;
    };
    double bhi = std::max(1.0, 10.0 / ub);
    double ga = g(1e-6);
    int doublings = 0;
    while (ga * g(bhi) > 0.0 && doublings < 60) {
      bhi *= 2.0;
      ++doublings;
    }
    double beta = brentq(g, 1e-6, bhi);

    double num = 1.0;
    double prod = 1.0;
    for (int j = 0; j < n; ++j) {
      num += std::pow(u_bar[j] / ub, n / 2.0);
      prod *= (u_bar[j] / ub);
    }
    double den = cn * (n + 1) * std::pow(ub, 1.0 + n / 2.0) * prod;
    double eps = 1.0 / (beta + v * num / den);
    out[i] = static_cast<float>(eps);
  }

  mx::array view(out.data(), mx::Shape{n}, mx::float32, [](void*) {});
  mx::array owned = mx::add(view, mx::array(0.0f));
  mx::eval(owned);
  return owned;
}

}  // namespace sabc
