/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/optimizer/optimizer.h"

namespace mnn {

struct GradientDescent: public Optimizer {
    GradientDescent();

    void update(const Vector &dW, Vector &W, bool parallelize);

private:
    Float alpha;   // learning rate
    Float lambda;  // weight decay
};

}  // namespace mnn
