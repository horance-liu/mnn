/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <memory>
#include <vector>

namespace mnn {


class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

class node : public std::enable_shared_from_this<node> {
 public:
  node(size_t in_size, size_t out_size);
  virtual ~node() {}

  const std::vector<edgeptr_t> &prev() const;
  const std::vector<edgeptr_t> &next() const;

  size_t prev_port(const edge &e) const;
  size_t next_port(const edge &e) const;

  std::vector<node *> prev_nodes() const;
  std::vector<node *> next_nodes() const;

 protected:
  node() = delete;

  friend void connect(layer *head,
                      layer *tail,
                      size_t head_index,
                      size_t tail_index);

  mutable std::vector<edgeptr_t> prev_;
  mutable std::vector<edgeptr_t> next_;
};

}  // namespace mnn
