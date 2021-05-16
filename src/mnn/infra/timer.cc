/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/infra/timer.h"

namespace mnn {

Timer::Timer() : t1(std::chrono::high_resolution_clock::now())
{
}

Float Timer::elapsed()
{
    return std::chrono::duration_cast<std::chrono::duration<Float>>(
            std::chrono::high_resolution_clock::now() - t1).count();
}
void Timer::restart()
{
    t1 = std::chrono::high_resolution_clock::now();
}
void Timer::start()
{
    t1 = std::chrono::high_resolution_clock::now();
}
void Timer::stop()
{
    t2 = std::chrono::high_resolution_clock::now();
}

Float Timer::total()
{
    stop();
    return std::chrono::duration_cast<std::chrono::duration<Float>>(t2 - t1).count();
}

}  // namespace mnn
