/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/util.h"

namespace mnn {
namespace weight_init {

class Function {
 public:
  virtual ~Function() {}
  virtual void fill(Vector *weight, size_t fan_in, size_t fan_out) = 0;
};

class Scalable : public Function {
 public:
  explicit Scalable(Float value) : scale_(value) {}

  void scale(Float value) { scale_ = value; }

 protected:
  Float scale_;
};

class Xavier : public Scalable {
 public:
  Xavier() : Scalable(Float(6)) {}
  explicit Xavier(Float value) : Scalable(value) {}

  void fill(Vector *weight, size_t fan_in, size_t fan_out) override {
    const Float weight_base = std::sqrt(scale_ / (fan_in + fan_out));

    uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
  }
};

class Constant : public Scalable {
 public:
  Constant() : Scalable(Float{0}) {}
  explicit Constant(Float value) : Scalable(value) {}

  void fill(Vector *weight, size_t fan_in, size_t fan_out) override {
    MNN_UNREFERENCED_PARAMETER(fan_in);
    MNN_UNREFERENCED_PARAMETER(fan_out);

    vectorize::fill(&(*weight)[0], weight->size(), scale_);
  }
};

}  // namespace weight_init
}  // namespace mnn
