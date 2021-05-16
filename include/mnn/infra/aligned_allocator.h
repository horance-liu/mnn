/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <stdlib.h>
#include <string>
#include <utility>
#include "mnn/infra/mnn_error.h"

namespace mnn {

template <typename T, std::size_t alignment>
class AlignedAllocator {
 public:
  typedef T value_type;
  typedef T *pointer;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T &reference;
  typedef const T &const_reference;
  typedef const T *const_pointer;

  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U, alignment> other;
  };

  AlignedAllocator() {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, alignment> &) {}

  const_pointer address(const_reference value) const {
    return std::addressof(value);
  }

  pointer address(reference value) const { return std::addressof(value); }

  pointer allocate(size_type size, const void * = nullptr) {
    void *p = aligned_alloc(alignment, sizeof(T) * size);
    if (!p && size > 0) throw MnnError("failed to allocate");
    return static_cast<pointer>(p);
  }

  size_type max_size() const {
    return ~static_cast<std::size_t>(0) / sizeof(T);
  }

  void deallocate(pointer ptr, size_type) { aligned_free(ptr); }

  template <class U, class V>
  void construct(U *ptr, const V &value) {
    void *p = ptr;
    ::new (p) U(value);
  }

  template <class U, class... Args>
  void construct(U *ptr, Args &&... args) {
    void *p = ptr;
    ::new (p) U(std::forward<Args>(args)...);
  }

  template <class U>
  void construct(U *ptr) {
    void *p = ptr;
    ::new (p) U();
  }

  template <class U>
  void destroy(U *ptr) {
    ptr->~U();
  }

 private:
  void *aligned_alloc(size_type align, size_type size) const {
    void *p;
    if (::posix_memalign(&p, align, size) != 0) {
      p = 0;
    }
    return p;
  }

  void aligned_free(pointer ptr) {
    ::free(ptr);
  }
};

template <typename T1, typename T2, std::size_t alignment>
inline bool operator==(const AlignedAllocator<T1, alignment> &,
                       const AlignedAllocator<T2, alignment> &) {
  return true;
}

template <typename T1, typename T2, std::size_t alignment>
inline bool operator!=(const AlignedAllocator<T1, alignment> &,
                       const AlignedAllocator<T2, alignment> &) {
  return false;
}

}  // namespace mnn
