/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/kernel/cpu/fully_connected_op_cpu.h"

namespace mnn {
namespace kernels {

void fully_connected_op_internal(const Matrix &in_data,
                                        const Vector &W,
                                        const Vector &bias,
                                        Matrix &out_data,
                                        const FullyParams &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const Vector &in = in_data[sample];
    Vector &out      = out_data[sample];

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = Float{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }
  });
}

void fully_connected_op_internal(const Matrix &prev_out,
                                        const Vector &W,
                                        Matrix &dW,
                                        Matrix &db,
                                        Matrix &curr_delta,
                                        Matrix &prev_delta,
                                        const FullyParams &params,
                                        const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const BlockedRange &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (params.has_bias_) {
        // Vector& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
  }
}

}  // namespace kernels
}  // namespace mnn
