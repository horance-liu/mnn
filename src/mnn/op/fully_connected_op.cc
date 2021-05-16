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
    const Matrix &in_data = context.input(0);
    const Matrix &W = context.input(1);
    const Matrix *bias = params.has_bias_ ? &context.input(2) : nullptr;
    Matrix &out_data = context.output(0);

    fill_tensor(out_data, Float { 0 });
    const BackendType engine = context.engine();

    if (engine == BackendType::CPU) {
        kernels::fully_connected_op_internal(in_data, W[0],
                params.has_bias_ ? (*bias)[0] : Vector(), out_data, params,
                context.parallelize());
    } else {
        throw MnnError("Not supported engine: " + to_string(engine));
    }
}

}  // namespace mnn
