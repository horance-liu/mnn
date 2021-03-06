/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <sstream>
#include <algorithm>

#include "mnn/infra/aligned_allocator.h"
#include "mnn/infra/parallel_for.h"
#include "mnn/infra/product.h"
#include "mnn/infra/random.h"

namespace mnn {

typedef size_t Label;

typedef size_t LayerSize;  // for backward compatibility

typedef std::vector<Float, AlignedAllocator<Float, 64>> Vector;

typedef std::vector<Vector> Matrix;

enum class NetPhase { TRAINING, TESTING };

enum class Padding {
  VALID,
  SAME,
};

template <typename T>
T *reverse_endian(T *p) {
  std::reverse(reinterpret_cast<char *>(p),
               reinterpret_cast<char *>(p) + sizeof(T));
  return p;
}

inline bool is_little_endian() {
  int x = 1;
  return *reinterpret_cast<char *>(&x) != 0;
}

template <typename T>
size_t max_index(const T &vec) {
  auto begin_iterator = std::begin(vec);
  return std::max_element(begin_iterator, std::end(vec)) - begin_iterator;
}

template <typename T, typename U>
U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
  U value = static_cast<U>(
    ((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
  return std::min(dst_max, std::max(value, dst_min));
}

inline void nop() {
  // do nothing
}

template <typename T>
inline T sqr(T value) {
  return value * value;
}

inline bool isfinite(Float x) { return x == x; }

template <typename Container>
inline bool has_infinite(const Container &c) {
  for (auto v : c)
    if (!isfinite(v)) return true;
  return false;
}

template <typename Container>
size_t max_size(const Container &c) {
  typedef typename Container::value_type value_t;
  const auto max_size =
    std::max_element(c.begin(), c.end(), [](const value_t &left,
                                            const value_t &right) {
      return left.size() < right.size();
    })->size();
  assert(max_size <= std::numeric_limits<size_t>::max());
  return max_size;
}

inline std::string format_str(const char *fmt, ...) {
  static char buf[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return std::string(buf);
}

template <typename T>
struct Index3d {
  Index3d(T width, T height, T depth) { reshape(width, height, depth); }

  Index3d() : width_(0), height_(0), depth_(0) {}

  void reshape(T width, T height, T depth) {
    width_  = width;
    height_ = height;
    depth_  = depth;

    if ((int64_t)width * height * depth > std::numeric_limits<T>::max())
      throw MnnError(format_str(
        "error while constructing layer: layer size too large for "
        "mnn\nWidthxHeightxChannels=%dx%dx%d >= max size of "
        "[%s](=%d)",
        width, height, depth, typeid(T).name(), std::numeric_limits<T>::max()));
  }

  T get_index(T x, T y, T channel) const {
    assert(x >= 0 && x < width_);
    assert(y >= 0 && y < height_);
    assert(channel >= 0 && channel < depth_);
    return (height_ * channel + y) * width_ + x;
  }

  T area() const { return width_ * height_; }

  T size() const { return width_ * height_ * depth_; }

  T width_;
  T height_;
  T depth_;
};

typedef Index3d<size_t> Shape3d;

template <typename T>
bool operator==(const Index3d<T> &lhs, const Index3d<T> &rhs) {
  return (lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_) &&
         (lhs.depth_ == rhs.depth_);
}

template <typename T>
bool operator!=(const Index3d<T> &lhs, const Index3d<T> &rhs) {
  return !(lhs == rhs);
}

template <typename Stream, typename T>
Stream &operator<<(Stream &s, const Index3d<T> &d) {
  s << d.width_ << "x" << d.height_ << "x" << d.depth_;
  return s;
}

template <typename T>
std::ostream &operator<<(std::ostream &s, const Index3d<T> &d) {
  s << d.width_ << "x" << d.height_ << "x" << d.depth_;
  return s;
}

template <typename Stream, typename T>
Stream &operator<<(Stream &s, const std::vector<Index3d<T>> &d) {
  s << "[";
  for (size_t i = 0; i < d.size(); i++) {
    if (i) s << ",";
    s << "[" << d[i] << "]";
  }
  s << "]";
  return s;
}

template <typename T>
std::string to_string(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T, typename Pred, typename Sum>
size_t sumif(const std::vector<T> &vec, Pred p, Sum s) {
  size_t sum = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (p(i)) sum += s(vec[i]);
  }
  return sum;
}

template <typename T, typename Pred>
std::vector<T> filter(const std::vector<T> &vec, Pred p) {
  std::vector<T> res;
  for (size_t i = 0; i < vec.size(); i++) {
    if (p(i)) res.push_back(vec[i]);
  }
  return res;
}

template <typename Result, typename T, typename Pred>
std::vector<Result> map_(const std::vector<T> &vec, Pred p) {
  std::vector<Result> res(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    res[i] = p(vec[i]);
  }
  return res;
}

enum class VectorType : int32_t {
  // 0x0001XXX : in/out data
  DATA = 0x0001000,  // input/output data, fed by other layer or input channel

  // 0x0002XXX : trainable parameters, updated for each back propagation
  WEIGHT = 0x0002000,
  BIAS   = 0x0002001,

  LABEL = 0x0004000,
  AUX   = 0x0010000  // layer-specific storage
};

inline std::ostream &operator<<(std::ostream &os, VectorType vtype) {
  os << to_string(vtype);
  return os;
}

inline VectorType operator&(VectorType lhs, VectorType rhs) {
  return (VectorType)(static_cast<int32_t>(lhs) & static_cast<int32_t>(rhs));
}
//
inline bool is_trainable_weight(VectorType vtype) {
  return (vtype & VectorType::WEIGHT) == VectorType::WEIGHT;
}

inline std::vector<VectorType> std_input_order(bool has_bias) {
  if (has_bias) {
    return {VectorType::DATA, VectorType::WEIGHT, VectorType::BIAS};
  } else {
    return {VectorType::DATA, VectorType::WEIGHT};
  }
}

inline void fill_tensor(Matrix &tensor, Float value) {
  for (auto &t : tensor) {
    vectorize::fill(&t[0], t.size(), value);
  }
}

inline void fill_tensor(Matrix &tensor, Float value, size_t size) {
  for (auto &t : tensor) {
    t.resize(size, value);
  }
}

inline size_t conv_out_length(size_t in_length,
                              size_t window_size,
                              size_t stride,
                              size_t dilation,
                              Padding pad_type) {
  size_t output_length;

  if (pad_type == Padding::SAME) {
    output_length = in_length;
  } else if (pad_type == Padding::VALID) {
    output_length = in_length - dilation * window_size + dilation;
  } else {
    throw MnnError("Not recognized pad_type.");
  }
  return (output_length + stride - 1) / stride;
}

inline size_t pool_out_length(size_t in_length,
                              size_t window_size,
                              size_t stride,
                              bool ceil_mode,
                              Padding pad_type) {
  size_t output_length;

  if (pad_type == Padding::SAME) {
    output_length = in_length;
  } else if (pad_type == Padding::VALID) {
    output_length = in_length - window_size + 1;
  } else {
    throw MnnError("Not recognized pad_type.");
  }

  float tmp = static_cast<float>((output_length + stride - 1)) / stride;
  return static_cast<int>(ceil_mode ? ceil(tmp) : floor(tmp));
}
}  // namespace mnn
