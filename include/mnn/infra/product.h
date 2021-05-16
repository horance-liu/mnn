/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <numeric>

#include "mnn/infra/macro.h"

namespace vectorize {
namespace detail {

// traits
template <typename T>
struct scalar_generic {
  typedef T register_type;
  typedef T value_type;
  enum { unroll_size = 1 };
  static MNN_MUST_INLINE register_type set1(const value_type &x) { return x; }
  static MNN_MUST_INLINE register_type zero() { return register_type(0); }
  static MNN_MUST_INLINE register_type mul(const register_type &v1,
                                           const register_type &v2) {
    return v1 * v2;
  }
  static MNN_MUST_INLINE register_type add(const register_type &v1,
                                           const register_type &v2) {
    return v1 + v2;
  }
  static MNN_MUST_INLINE register_type madd(const register_type &v1,
                                            const register_type &v2,
                                            const register_type &v3) {
    return v1 * v2 + v3;
  }

  template <typename aligned>
  static MNN_MUST_INLINE register_type load(const value_type *px) {
    return *px;
  }
  template <typename aligned>
  static MNN_MUST_INLINE void store(value_type *px, const register_type &v) {
    *px = v;
  }

  static MNN_MUST_INLINE value_type resemble(const register_type &x) {
    return x;
  }

  static MNN_MUST_INLINE bool is_aligned(value_type *p) { return true; }
};

// generic dot-product
template <typename T, typename f1_aligned, typename f2_aligned>
MNN_MUST_INLINE typename T::value_type dot_product(
  const typename T::value_type *f1,
  const typename T::value_type *f2,
  std::size_t size) {
  typename T::register_type r0 = T::zero();
  typename T::register_type r1 = T::zero();
  typename T::register_type r2 = T::zero();
  typename T::register_type r3 = T::zero();
  auto sz                      = T::unroll_size;
  auto sz4                     = T::unroll_size * 4;
  auto n4                      = size / sz4;
  auto n1                      = (size % sz4) / sz;
  auto remain                  = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto s10 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 0]);
    auto s11 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 1]);
    auto s12 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 2]);
    auto s13 = T::template load<f1_aligned>(&f1[i * sz4 + sz * 3]);
    auto s20 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 0]);
    auto s21 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 1]);
    auto s22 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 2]);
    auto s23 = T::template load<f2_aligned>(&f2[i * sz4 + sz * 3]);
    r0       = T::madd(s10, s20, r0);
    r1       = T::madd(s11, s21, r1);
    r2       = T::madd(s12, s22, r2);
    r3       = T::madd(s13, s23, r3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto s1 = T::template load<f1_aligned>(&f1[idx + i * sz]);
    auto s2 = T::template load<f2_aligned>(&f2[idx + i * sz]);
    r0      = T::madd(s1, s2, r0);
  }
  r0                         = T::add(r0, r1);
  r2                         = T::add(r2, r3);
  r0                         = T::add(r0, r2);
  typename T::value_type sum = T::resemble(r0);
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    sum += f1[idx + i] * f2[idx + i];
  }
  return sum;
}

template <typename T, typename dst_aligned>
MNN_MUST_INLINE void add(typename T::value_type c,
                         std::size_t size,
                         typename T::value_type *dst) {
  typename T::register_type c2 = T::set1(c);
  auto sz                      = T::unroll_size;
  auto sz4                     = T::unroll_size * 4;
  auto n4                      = size / sz4;
  auto n1                      = (size % sz4) / sz;
  auto remain                  = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    d0      = T::add(c2, d0);
    d1      = T::add(c2, d1);
    d2      = T::add(c2, d2);
    d3      = T::add(c2, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    d      = T::add(c2, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += c;
  }
}

template <typename T, typename src_aligned, typename dst_aligned>
MNN_MUST_INLINE void add(const typename T::value_type *src,
                         std::size_t size,
                         typename T::value_type *dst) {
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::add(s0, d0);
    d1      = T::add(s1, d1);
    d2      = T::add(s2, d2);
    d3      = T::add(s3, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::add(s, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i];
  }
}

// TODO(beru): documentation
/**
 *
 * @tparam T
 * @tparam src_aligned
 * @tparam dst_aligned
 * @param src
 * @param c
 * @param size
 * @param dst
 */
template <typename T, typename src_aligned, typename dst_aligned>
MNN_MUST_INLINE void muladd(const typename T::value_type *src,
                            typename T::value_type c,
                            std::size_t size,
                            typename T::value_type *dst) {
  auto factor = T::set1(c);
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::madd(s0, factor, d0);
    d1      = T::madd(s1, factor, d1);
    d2      = T::madd(s2, factor, d2);
    d3      = T::madd(s3, factor, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::madd(s, factor, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i] * c;
  }
}

template <typename T, typename src_aligned, typename dst_aligned>
MNN_MUST_INLINE void reduce(const typename T::value_type *src,
                            std::size_t size,
                            typename T::value_type *dst) {
  auto sz     = T::unroll_size;
  auto sz4    = T::unroll_size * 4;
  auto n4     = size / sz4;
  auto n1     = (size % sz4) / sz;
  auto remain = size % sz;
  for (size_t i = 0; i < n4; ++i) {
    auto d0 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 0]);
    auto d1 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 1]);
    auto d2 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 2]);
    auto d3 = T::template load<dst_aligned>(&dst[i * sz4 + sz * 3]);
    auto s0 = T::template load<src_aligned>(&src[i * sz4 + sz * 0]);
    auto s1 = T::template load<src_aligned>(&src[i * sz4 + sz * 1]);
    auto s2 = T::template load<src_aligned>(&src[i * sz4 + sz * 2]);
    auto s3 = T::template load<src_aligned>(&src[i * sz4 + sz * 3]);
    d0      = T::add(s0, d0);
    d1      = T::add(s1, d1);
    d2      = T::add(s2, d2);
    d3      = T::add(s3, d3);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 0], d0);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 1], d1);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 2], d2);
    T::template store<dst_aligned>(&dst[i * sz4 + sz * 3], d3);
  }
  size_t idx = n4 * sz4;
  for (size_t i = 0; i < n1; ++i) {
    auto d = T::template load<dst_aligned>(&dst[idx + i * sz]);
    auto s = T::template load<src_aligned>(&src[idx + i * sz]);
    d      = T::add(s, d);
    T::template store<dst_aligned>(&dst[idx + i * sz], d);
  }
  idx += n1 * sz;
  for (size_t i = 0; i < remain; ++i) {
    dst[idx + i] += src[idx + i];
  }
}

template <typename T>
void fill(T *dst, size_t size, T value) {
  std::fill(dst, dst + size, value);
}


#ifdef MNN_USE_DOUBLE
#define MNN_VECTORIZE_TYPE detail::scalar_generic<double>
#else
#define MNN_VECTORIZE_TYPE detail::scalar_generic<float>
#endif

}  // namespace detail

// dst[i] += c
template <typename T>
void add(T c, std::size_t size, T *dst) {
  bool is_dst_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)dst);
  if (is_dst_aligned) {
    detail::add<MNN_VECTORIZE_TYPE, std::true_type>(c, size, dst);
  } else {
    detail::add<MNN_VECTORIZE_TYPE, std::false_type>(c, size, dst);
  }
}

// dst[i] += src[i]
template <typename T>
void add(const T *src, std::size_t size, T *dst) {
  bool src_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::add<MNN_VECTORIZE_TYPE, std::true_type, std::true_type>(src, size,
                                                                      dst);
    } else {
      detail::add<MNN_VECTORIZE_TYPE, std::true_type, std::false_type>(
        src, size, dst);
    }
  } else {
    if (dst_aligned) {
      detail::add<MNN_VECTORIZE_TYPE, std::false_type, std::true_type>(
        src, size, dst);
    } else {
      detail::add<MNN_VECTORIZE_TYPE, std::false_type, std::false_type>(
        src, size, dst);
    }
  }
}

// dst[i] += c * src[i]
template <typename T>
void muladd(const T *src, T c, std::size_t size, T *dst) {
  bool src_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::muladd<MNN_VECTORIZE_TYPE, std::true_type, std::true_type>(
        src, c, size, dst);
    } else {
      detail::muladd<MNN_VECTORIZE_TYPE, std::true_type, std::false_type>(
        src, c, size, dst);
    }
  } else {
    if (dst_aligned) {
      detail::muladd<MNN_VECTORIZE_TYPE, std::false_type, std::true_type>(
        src, c, size, dst);
    } else {
      detail::muladd<MNN_VECTORIZE_TYPE, std::false_type, std::false_type>(
        src, c, size, dst);
    }
  }
}

// sum(s1[i] * s2[i])
template <typename T>
T dot(const T *s1, const T *s2, std::size_t size) {
  bool s1_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)s1);
  bool s2_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)s2);
  if (s1_aligned) {
    if (s2_aligned) {
      return detail::dot_product<MNN_VECTORIZE_TYPE, std::true_type,
                                 std::true_type>(s1, s2, size);
    } else {
      return detail::dot_product<MNN_VECTORIZE_TYPE, std::true_type,
                                 std::false_type>(s1, s2, size);
    }
  } else {
    if (s2_aligned) {
      return detail::dot_product<MNN_VECTORIZE_TYPE, std::false_type,
                                 std::true_type>(s1, s2, size);
    } else {
      return detail::dot_product<MNN_VECTORIZE_TYPE, std::false_type,
                                 std::false_type>(s1, s2, size);
    }
  }
}

/// dst[i] += src[i]
template <typename T>
void reduce(const T *src, std::size_t size, T *dst) {
  bool src_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)src);
  bool dst_aligned =
    MNN_VECTORIZE_TYPE::is_aligned((MNN_VECTORIZE_TYPE::value_type *)dst);
  if (src_aligned) {
    if (dst_aligned) {
      detail::reduce<MNN_VECTORIZE_TYPE, std::true_type, std::true_type>(
        src, size, dst);
    } else {
      detail::reduce<MNN_VECTORIZE_TYPE, std::true_type, std::false_type>(
        src, size, dst);
    }
  } else {
    if (dst_aligned) {
      detail::reduce<MNN_VECTORIZE_TYPE, std::false_type, std::true_type>(
        src, size, dst);
    } else {
      detail::reduce<MNN_VECTORIZE_TYPE, std::false_type, std::false_type>(
        src, size, dst);
    }
  }
}

template <typename T>
MNN_MUST_INLINE void fill(T *dst, std::size_t size, T value) {
  detail::fill(dst, size, value);
}

}  // namespace vectorize
