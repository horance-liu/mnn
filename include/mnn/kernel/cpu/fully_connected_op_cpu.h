/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/util.h"
#include "mnn/core/params/fully_params.h"

namespace mnn {
namespace kernels {

void fully_connected_op_internal(const Matrix &in_data,
                                        const Vector &W,
                                        const Vector &bias,
                                        Matrix &out_data,
                                        const FullyParams &params,
                                        const bool layer_parallelize);


void fully_connected_op_internal(const Matrix &prev_out,
                                        const Vector &W,
                                        Matrix &dW,
                                        Matrix &db,
                                        Matrix &curr_delta,
                                        Matrix &prev_delta,
                                        const FullyParams &params,
                                        const bool layer_parallelize);
}  // namespace kernels
}  // namespace mnn
