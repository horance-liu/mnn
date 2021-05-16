/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <unordered_map>
#include "mnn/infra/util.h"

namespace mnn {

struct optimizer {
    virtual void update(const vec_t &dW, vec_t &W, bool parallelize) = 0;
    virtual void reset() {}
    virtual ~optimizer() {}
};

}  // namespace mnn
