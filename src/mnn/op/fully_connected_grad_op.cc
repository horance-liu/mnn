/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/op/fully_connected_grad_op.h"
#include "mnn/kernel/cpu/fully_connected_op_cpu.h"

namespace mnn {

FullyConnectedGradOp::FullyConnectedGradOp(const OpKernelConstruction &context) : OpKernel(
        context)
{
}

void FullyConnectedGradOp::compute(OpKernelContext &context)
{
    auto params = OpKernel::params_->fully();

    // incoming/outcoming data
    const tensor_t &prev_out = context.input(0);
    const tensor_t &W = context.input(1);
    tensor_t &dW = context.input_grad(1);
    tensor_t *db = params.has_bias_ ? &context.input_grad(2) : nullptr;
    tensor_t &prev_delta = context.input_grad(0);
    tensor_t &curr_delta = context.output_grad(0);
    tensor_t dummy;  // need lvalue for non-const reference

    fill_tensor(prev_delta, float_t { 0 });
    const backend_t engine = context.engine();

    if (engine == backend_t::cpu) {
        kernels::fully_connected_op_internal(prev_out, W[0], dW,
                params.has_bias_ ? *db : dummy, curr_delta, prev_delta, params,
                context.parallelize());
    } else {
        throw nn_error("Not supported engine: " + to_string(engine));
    }
}

}  // namespace mnn
