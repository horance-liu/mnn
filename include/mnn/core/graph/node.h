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


class Node;
class Layer;
class Edge;

typedef std::shared_ptr<Edge> edgeptr_t;

class Node : public std::enable_shared_from_this<Node> {
 public:
  Node(size_t in_size, size_t out_size);
  virtual ~Node() {}

  const std::vector<edgeptr_t> &prev() const;
  const std::vector<edgeptr_t> &next() const;

  size_t prev_port(const Edge &e) const;
  size_t next_port(const Edge &e) const;

  std::vector<Node *> prev_nodes() const;
  std::vector<Node *> next_nodes() const;

 protected:
  Node() = delete;

  friend void connect(Layer *head,
                      Layer *tail,
                      size_t head_index,
                      size_t tail_index);

  mutable std::vector<edgeptr_t> prev_;
  mutable std::vector<edgeptr_t> next_;
};

}  // namespace mnn
