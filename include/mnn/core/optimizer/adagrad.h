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

struct Adagrad: public StatefulOptimizer<1> {
    Adagrad();

    void update(const Vector &dW, Vector &W, bool parallelize);

    Float alpha;

private:
    Float eps;
};

}  // namespace mnn
