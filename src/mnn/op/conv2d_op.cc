/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/op/conv2d_op.h"
#include "mnn/kernel/cpu/conv2d_op_cpu.h"

namespace mnn {

Conv2dOp::Conv2dOp(const OpKernelConstruction &context) : OpKernel(context)
{
}

void Conv2dOp::compute(OpKernelContext &context)
{
    auto params = OpKernel::params_->conv();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W = context.input(1);
    const tensor_t &bias = context.input(2);
    tensor_t &out_data = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t { 0 });
    const backend_t engine = context.engine();

    if (engine == backend_t::cpu) {
        kernels::conv2d_op_internal(in_data, W[0], bias[0], out_data, params,
                context.parallelize());
    } else {
        throw nn_error("Not supported engine: " + to_string(engine));
    }
}

}
// namespace mnn
