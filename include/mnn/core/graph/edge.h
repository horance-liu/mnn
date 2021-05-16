/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <memory>
#include "mnn/infra/util.h"

namespace mnn {

class Node;
class Layer;
class Edge;

class Edge {
 public:
  Edge(Node *prev, const Shape3d &shape, VectorType vtype);

  void merge_grads(Vector *dst);
  void clear_grads();

  Matrix *get_data();
  const Matrix *get_data() const;
  Matrix *get_gradient();
  const Matrix *get_gradient() const;
  const std::vector<Node *> &next() const;
  Node *prev();
  const Node *prev() const;
  const Shape3d &shape() const;
  VectorType vtype() const;
  void add_next_node(Node *next);

 private:
  Shape3d shape_;
  VectorType vtype_;
  Matrix data_;
  Matrix grad_;
  Node *prev_;                // previous node, "producer" of this tensor
  std::vector<Node *> next_;  // next nodes, "consumers" of this tensor
};

}  // namespace mnn
