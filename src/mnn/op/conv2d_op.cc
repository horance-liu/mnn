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
    const Matrix &in_data = context.input(0);
    const Matrix &W = context.input(1);
    const Matrix &bias = context.input(2);
    Matrix &out_data = context.output(0);

    // initialize outputs
    fill_tensor(out_data, Float { 0 });
    const BackendType engine = context.engine();

    if (engine == BackendType::CPU) {
        kernels::conv2d_op_internal(in_data, W[0], bias[0], out_data, params,
                context.parallelize());
    } else {
        throw MnnError("Not supported engine: " + to_string(engine));
    }
}

}
// namespace mnn
