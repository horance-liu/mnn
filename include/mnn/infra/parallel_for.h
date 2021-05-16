/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mnn/infra/config.h"
#include "mnn/infra/aligned_allocator.h"
#include "mnn/infra/mnn_error.h"

#ifdef MNN_USE_TBB
#ifndef NOMINMAX
#define NOMINMAX  // tbb includes windows.h in tbb/machine/windows_api.h
#endif
#include <tbb/task_group.h>
#include <tbb/tbb.h>
#endif

#if !defined(MNN_USE_OMP) && !defined(MNN_SINGLE_THREAD)
#include <future>  // NOLINT
#include <thread>  // NOLINT
#endif

#if defined(MNN_USE_GCD) && !defined(MNN_SINGLE_THREAD)
#include <dispatch/dispatch.h>
#endif

namespace mnn {

#ifdef MNN_USE_TBB

static tbb::task_scheduler_init tbbScheduler(
  tbb::task_scheduler_init::automatic);  // tbb::task_scheduler_init::deferred);

typedef tbb::BlockedRange<size_t> BlockedRange;

template <typename Func>
void parallel_for(size_t begin, size_t end, const Func &f, size_t grainsize) {
  assert(end >= begin);
  tbb::parallel_for(
    BlockedRange(begin, end, end - begin > grainsize ? grainsize : 1), f);
}

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  f(BlockedRange(begin, end, 100));
}

#else

struct BlockedRange {
  typedef size_t const_iterator;

  BlockedRange(size_t begin, size_t end) : begin_(begin), end_(end) {}
  BlockedRange(int begin, int end) : begin_(begin), end_(end) {}

  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }

 private:
  size_t begin_;
  size_t end_;
};

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  BlockedRange r(begin, end);
  f(r);
}

#if defined(MNN_USE_OMP)

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  assert(end >= begin);
// unsigned index isn't allowed in OpenMP 2.0
#pragma omp parallel for
  for (int i = static_cast<int>(begin); i < static_cast<int>(end); ++i)
    f(BlockedRange(i, i + 1));
}

#elif defined(MNN_USE_GCD)

template <typename Func>
void parallel_for(size_t begin, size_t end, const Func &f, size_t grainsize) {
  assert(end >= begin);
  size_t count     = end - begin;
  size_t blockSize = grainsize;
  if (count < blockSize || blockSize == 0) {
    blockSize = 1;
  }
  size_t blockCount = (count + blockSize - 1) / blockSize;
  assert(blockCount > 0);

  dispatch_apply(blockCount, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
                 ^(size_t block) {
                   size_t blockStart = block * blockSize;
                   size_t blockEnd   = blockStart + blockSize;
                   if (blockEnd > end) {
                     blockEnd = end;
                   }
                   assert(blockStart < blockEnd);

                   f(BlockedRange(blockStart, blockEnd));
                 });
}

#elif defined(MNN_SINGLE_THREAD)

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  xparallel_for(begin, end, f);
}

#else

template <typename Func>
void parallel_for(size_t begin,
                  size_t end,
                  const Func &f,
                  size_t /*grainsize*/) {
  assert(end >= begin);
  size_t nthreads  = std::thread::hardware_concurrency();
  size_t blockSize = (end - begin) / nthreads;
  if (blockSize * nthreads < end - begin) blockSize++;

  std::vector<std::future<void> > futures;

  size_t blockBegin            = begin;
  size_t blockEnd              = blockBegin + blockSize;
  if (blockEnd > end) blockEnd = end;

  for (size_t i = 0; i < nthreads; i++) {
    futures.push_back(
      std::move(std::async(std::launch::async, [blockBegin, blockEnd, &f] {
        f(BlockedRange(blockBegin, blockEnd));
      })));

    blockBegin += blockSize;
    blockEnd = blockBegin + blockSize;
    if (blockBegin >= end) break;
    if (blockEnd > end) blockEnd = end;
  }

  for (auto &future : futures) future.wait();
}

#endif

#endif  // MNN_USE_TBB

template <typename T, typename U>
bool value_representation(U const &value) {
  return static_cast<U>(static_cast<T>(value)) == value;
}

template <typename T, typename Func>
inline void for_(
  bool parallelize, size_t begin, T end, Func f, size_t grainsize = 100) {
  static_assert(std::is_integral<T>::value, "end must be integral type");
  parallelize = parallelize && value_representation<size_t>(end);
  parallelize ? parallel_for(begin, end, f, grainsize)
              : xparallel_for(begin, end, f);
}

template <typename T, typename Func>
inline void for_i(bool parallelize, T size, Func f, size_t grainsize = 100u) {
#ifdef MNN_SINGLE_THREAD
  for (size_t i = 0; i < size; ++i) {
    f(i);
  }
#else  // #ifdef MNN_SINGLE_THREAD
  for_(parallelize, 0u, size,
       [&](const BlockedRange &r) {
#ifdef MNN_USE_OMP
#pragma omp parallel for
         for (int i = static_cast<int>(r.begin());
              i < static_cast<int>(r.end()); i++) {
           f(i);
         }
#else
         for (size_t i = r.begin(); i < r.end(); i++) {
           f(i);
         }
#endif
       },
       grainsize);
#endif  // #ifdef MNN_SINGLE_THREAD
}

template <typename T, typename Func>
inline void for_i(T size, Func f, size_t grainsize = 100) {
  for_i(true, size, f, grainsize);
}

}  // namespace mnn
