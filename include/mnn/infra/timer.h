/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <chrono>
#include "mnn/infra/config.h"

namespace mnn {

class timer {
public:
    timer();

    float_t elapsed();

    void restart();
    void start();
    void stop();

    float_t total();

private:
    std::chrono::high_resolution_clock::time_point t1, t2;
};

}  // namespace mnn
