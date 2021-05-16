/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#define MNN_UNREFERENCED_PARAMETER(x) (void)(x)

#if defined(__GNUC__) || defined(__clang__) || defined(__ICC)
#define MNN_MUST_INLINE __attribute__((always_inline)) inline
#else
#define MNN_MUST_INLINE inline
#endif
