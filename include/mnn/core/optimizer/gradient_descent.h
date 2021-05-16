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

struct gradient_descent: public optimizer {
    gradient_descent();

    void update(const vec_t &dW, vec_t &W, bool parallelize);

private:
    float_t alpha;   // learning rate
    float_t lambda;  // weight decay
};

}  // namespace mnn
