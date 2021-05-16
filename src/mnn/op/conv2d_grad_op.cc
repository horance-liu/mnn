/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/op/conv2d_grad_op.h"
#include "mnn/kernel/cpu/conv2d_op_cpu.h"

namespace mnn {

Conv2dGradOp::Conv2dGradOp(const OpKernelConstruction &context) : OpKernel(
        context)
{
}

void Conv2dGradOp::compute(OpKernelContext &context)
{
    auto params = OpKernel::params_->conv();

    // incoming/outcoming data
    const Matrix &prev_out = context.input(0);
    const Matrix &W = context.input(1);
    Matrix &dW = context.input_grad(1);
    Matrix &db = context.input_grad(2);
    Matrix &prev_delta = context.input_grad(0);
    Matrix &curr_delta = context.output_grad(0);

    fill_tensor(prev_delta, Float { 0 });
    const BackendType engine = context.engine();

    if (engine == BackendType::CPU) {
        kernels::conv2d_op_internal(prev_out, W[0], dW, db, curr_delta,
                prev_delta, params, context.parallelize());
    } else {
        throw MnnError("Not supported engine: " + to_string(engine));
    }
}

}
// namespace mnn
