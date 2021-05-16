/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <vector>
#include <algorithm>

namespace mnn {

template <typename U = Float, typename Storage = std::vector<U>>
class Tensor {
  typedef U *UPtr;

 public:
  Tensor() {}
  explicit Tensor(std::vector<size_t> const &shape) : storage_(shape) {}
  explicit Tensor(std::initializer_list<size_t> const &shape)
    : storage_(shape) {}

  explicit Tensor(std::initializer_list<size_t> const &shape, U value)
    : storage_(shape, value) {}

  Tensor<U> &operator=(const Tensor<U> &T) {
    storage_ = T.storage_;
    return *this;
  }

  const size_t size() const { return storage_.size(); }

  Tensor &fill(U value) {
    std::fill(storage_.begin(), storage_.end(), value);
    return *this;
  }

  Tensor operator[](size_t index) { return Tensor(storage_[index]); }

 private:
  Storage storage_;
};

}  // namespace mnn
