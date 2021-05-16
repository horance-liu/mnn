/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <cstddef>
#include <cstdint>


#define MNN_USE_EXCEPTIONS
#define MNN_USE_STDOUT

#ifdef MNN_USE_OMP
#define MNN_TASK_SIZE 100
#else
#define MNN_TASK_SIZE 8
#endif

namespace mnn {

#ifdef MNN_USE_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif

}  // namespace mnn
