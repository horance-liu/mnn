/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/op/fully_connected_op.h"
#include "mnn/kernel/cpu/fully_connected_op_cpu.h"

namespace mnn {

FullyConnectedOp::FullyConnectedOp(const OpKernelConstruction &context) : OpKernel(
        context)
{
}

void FullyConnectedOp::compute(OpKernelContext &context)
{
    auto params = OpKernel::params_->fully();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W = context.input(1);
    const tensor_t *bias = params.has_bias_ ? &context.input(2) : nullptr;
    tensor_t &out_data = context.output(0);

    fill_tensor(out_data, float_t { 0 });
    const backend_t engine = context.engine();

    if (engine == backend_t::cpu) {
        kernels::fully_connected_op_internal(in_data, W[0],
                params.has_bias_ ? (*bias)[0] : vec_t(), out_data, params,
                context.parallelize());
    } else {
        throw nn_error("Not supported engine: " + to_string(engine));
    }
}

}  // namespace mnn
