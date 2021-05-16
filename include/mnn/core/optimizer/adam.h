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

struct adam: public stateful_optimizer<2> {
    adam();
    void update(const vec_t &dW, vec_t &W, bool parallelize);

    float_t alpha;  // learning rate
    float_t b1;     // decay term
    float_t b2;     // decay term
    float_t b1_t;   // decay term power t
    float_t b2_t;   // decay term power t

private:
    float_t eps;
};

}  // namespace mnn
