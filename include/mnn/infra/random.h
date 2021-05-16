/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <limits>
#include <random>
#include <type_traits>

#include "mnn/infra/config.h"
#include "mnn/infra/mnn_error.h"

namespace mnn {
//
class RandomGenerator {
 public:
  static RandomGenerator &get_instance() {
    static RandomGenerator instance;
    return instance;
  }

  std::mt19937 &operator()() { return gen_; }

  void set_seed(unsigned int seed) { gen_.seed(seed); }

 private:
  // avoid gen_(0) for MSVC known issue
  // https://connect.microsoft.com/VisualStudio/feedback/details/776456
  RandomGenerator() : gen_(1) {}
  std::mt19937 gen_;
};
//
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_int_distribution<T> dst(min, max);
  return dst(RandomGenerator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_real_distribution<T> dst(min, max);
  return dst(RandomGenerator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
gaussian_rand(T mean, T sigma) {
  std::normal_distribution<T> dst(mean, sigma);
  return dst(RandomGenerator::get_instance()());
}

template <typename Container>
inline int uniform_idx(const Container &t) {
  return uniform_rand(0, static_cast<int>(t.size() - 1));
}

template <typename Iter>
void uniform_rand(Iter begin, Iter end, Float min, Float max) {
  for (Iter it = begin; it != end; ++it) *it = uniform_rand(min, max);
}

template <typename Iter>
void gaussian_rand(Iter begin, Iter end, Float mean, Float sigma) {
  for (Iter it = begin; it != end; ++it) *it = gaussian_rand(mean, sigma);
}

}  // namespace mnn
