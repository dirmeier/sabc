#pragma once
#include <cstddef>
#include <functional>
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "common.hpp"

namespace nb = nanobind;

namespace sabc {

// float32 MLX ndarray of arbitrary rank, for DLPack interop with mlx.core.
using MlxArray = nb::ndarray<nb::mlx, float>;

// Import a Python mlx.core.array (any rank, float32) into an OWNING mx::array.
//
// The borrowed DLPack buffer is wrapped with a no-op deleter, then copied into
// MLX-owned storage via mx::add before mx::eval so the result is independent of
// the Python array's lifetime.
inline mx::array to_mx(const nb::object& obj) {
  auto a = nb::cast<MlxArray>(obj);
  mx::Shape shape;
  shape.reserve(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    shape.push_back(static_cast<mx::ShapeElem>(a.shape(i)));
  }
  mx::array view(a.data(), shape, mx::float32, [](void*) {});
  mx::array owned = mx::add(view, mx::array(0.0f));
  mx::eval(owned);
  return owned;
}

// Import a Python mlx.core.array of uint32 (e.g. a PRNG key) into an OWNING
// mx::array.  Mirrors to_mx but for the key dtype: the float caster would throw
// on a uint32 buffer, so the cast and the owning-copy zero are uint32-typed.
inline mx::array to_mx_u32(const nb::object& obj) {
  auto a = nb::cast<nb::ndarray<nb::mlx, uint32_t>>(obj);
  mx::Shape shape;
  shape.reserve(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    shape.push_back(static_cast<mx::ShapeElem>(a.shape(i)));
  }
  mx::array view(a.data(), shape, mx::uint32, [](void*) {});
  mx::array owned = mx::add(view, mx::array(0u, mx::uint32));
  mx::eval(owned);
  return owned;
}

// Export an mx::array (float32) to a float32 MLX ndarray via DLPack, with a
// capsule keep-alive holding a shared_ptr to the evaluated array so its buffer
// stays valid for the lifetime of the returned ndarray.
inline MlxArray to_py(const mx::array& a) {
  auto result = std::make_shared<mx::array>(a);
  mx::eval(*result);
  auto* stored = new std::shared_ptr<mx::array>(result);
  nb::capsule keep(stored, [](void* p) noexcept {
    delete static_cast<std::shared_ptr<mx::array>*>(p);
  });
  std::vector<std::size_t> shape;
  shape.reserve(result->ndim());
  for (auto d : result->shape()) {
    shape.push_back(static_cast<std::size_t>(d));
  }
  return MlxArray(result->data<float>(), shape.size(), shape.data(), keep);
}

// Wrap a Python callable (simulator / stats / prior) as a C++ callback.  The
// std::function lives entirely inside C++; nanobind only sees the nb::callable.
inline std::function<mx::array(const mx::array&)> make_callback(
    nb::callable fn) {
  return [fn](const mx::array& x) -> mx::array { return to_mx(fn(to_py(x))); };
}

}  // namespace sabc
