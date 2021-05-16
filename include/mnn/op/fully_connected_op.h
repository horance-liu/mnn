/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/op_kernel.h"

namespace mnn {

class FullyConnectedOp: public OpKernel {
public:
    explicit FullyConnectedOp(const OpKernelConstruction &context);
    void compute(OpKernelContext &context) override;
};

}  // namespace mnn
