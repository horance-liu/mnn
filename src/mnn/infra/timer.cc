/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/infra/timer.h"

namespace mnn {

timer::timer() : t1(std::chrono::high_resolution_clock::now())
{
}

float_t timer::elapsed()
{
    return std::chrono::duration_cast<std::chrono::duration<float_t>>(
            std::chrono::high_resolution_clock::now() - t1).count();
}
void timer::restart()
{
    t1 = std::chrono::high_resolution_clock::now();
}
void timer::start()
{
    t1 = std::chrono::high_resolution_clock::now();
}
void timer::stop()
{
    t2 = std::chrono::high_resolution_clock::now();
}

float_t timer::total()
{
    stop();
    return std::chrono::duration_cast<std::chrono::duration<float_t>>(t2 - t1).count();
}

}  // namespace mnn
