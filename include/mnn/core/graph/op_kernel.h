/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mnn/core/params/conv_params.h"
#include "mnn/infra/backend.h"

namespace mnn {

class OpKernel;  // delared below

class OpKernelConstruction {
 public:
  OpKernelConstruction() {}
  explicit OpKernelConstruction(Params *params) : params_(params) {}

  Params *params() const { return params_; }

 private:
  Params *params_ = nullptr;
};

class layer;

class OpKernelContext {
 public:
  struct OpParams {
    OpKernel *op_kernel_ptr = nullptr;
    layer *layer_ptr_ = nullptr;
    Params *params_ptr_ = nullptr;
    bool parallelize = false;
    backend_t engine = default_engine();
  };

  OpKernelContext()
    : in_data_(nullptr),
      out_data_(nullptr),
      out_grad_(nullptr),
      in_grad_(nullptr) {
    op_params_ = std::unique_ptr<OpParams>(new OpParams());
  }

  void set_in_out(const std::vector<tensor_t *> &in_data,
                  std::vector<tensor_t *> &out_data) {
    in_data_  = const_cast<std::vector<tensor_t *> *>(&in_data);
    out_data_ = &out_data;
  }

  void set_in_out(const std::vector<tensor_t *> &in_data,
                  const std::vector<tensor_t *> &out_data,
                  std::vector<tensor_t *> &out_grad,
                  std::vector<tensor_t *> &in_grad) {
    in_data_  = const_cast<std::vector<tensor_t *> *>(&in_data);
    out_data_ = const_cast<std::vector<tensor_t *> *>(&out_data);
    out_grad_ = &out_grad;
    in_grad_  = &in_grad;
  }

  tensor_t &input(const int idx) { return *(*in_data_)[idx]; }
  const tensor_t &input(const int idx) const { return *(*in_data_)[idx]; }

  tensor_t &output(const int idx) { return *(*out_data_)[idx]; }
  const tensor_t &output(const int idx) const { return *(*out_data_)[idx]; }

  tensor_t &input_grad(const int idx) { return *(*in_grad_)[idx]; }
  const tensor_t &input_grad(const int idx) const { return *(*in_grad_)[idx]; }

  tensor_t &output_grad(const int idx) { return *(*out_grad_)[idx]; }
  const tensor_t &output_grad(const int idx) const {
    return *(*out_grad_)[idx];
  }

  void setParams(Params *params) { op_params_->params_ptr_ = params; }
  Params *params() const { return op_params_->params_ptr_; }

  void setParallelize(const bool parallelize) {
    op_params_->parallelize = parallelize;
  }

  bool parallelize() const { return op_params_->parallelize; }
//  void setLayer(layer *layer) { op_params_->layer_ptr_ = layer; }
//
//  layer *Layer() const { return op_params_->layer_ptr_; }

  backend_t engine() const { return op_params_->engine; }

  void setEngine(const backend_t engine) { op_params_->engine = engine; }

 private:
  std::vector<tensor_t *> *in_data_;
  std::vector<tensor_t *> *out_data_;
  std::vector<tensor_t *> *out_grad_;
  std::vector<tensor_t *> *in_grad_;

  std::unique_ptr<OpParams> op_params_;
};

class OpKernel {
 public:
  OpKernel() {}
  explicit OpKernel(const OpKernelConstruction &context)
    : params_(context.params()) {}

  virtual ~OpKernel() {}

  virtual void compute(OpKernelContext &context) = 0;

 protected:
  Params *params_ = nullptr;
};

}  // namespace mnn
