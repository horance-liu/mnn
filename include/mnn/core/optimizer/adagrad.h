/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/optimizer/stateful_optimizer.h"

namespace mnn {

struct adagrad: public stateful_optimizer<1> {
    adagrad();

    void update(const vec_t &dW, vec_t &W, bool parallelize);

    float_t alpha;

private:
    float_t eps;
};

}  // namespace mnn
