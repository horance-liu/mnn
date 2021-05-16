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

struct Adam: public StatefulOptimizer<2> {
    Adam();
    void update(const Vector &dW, Vector &W, bool parallelize);

    Float alpha;  // learning rate
    Float b1;     // decay term
    Float b2;     // decay term
    Float b1_t;   // decay term power t
    Float b2_t;   // decay term power t

private:
    Float eps;
};

}  // namespace mnn
