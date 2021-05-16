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

class node;
class layer;
class edge;

class edge {
 public:
  edge(node *prev, const shape3d &shape, vector_type vtype);

  void merge_grads(vec_t *dst);
  void clear_grads();

  tensor_t *get_data();
  const tensor_t *get_data() const;
  tensor_t *get_gradient();
  const tensor_t *get_gradient() const;
  const std::vector<node *> &next() const;
  node *prev();
  const node *prev() const;
  const shape3d &shape() const;
  vector_type vtype() const;
  void add_next_node(node *next);

 private:
  shape3d shape_;
  vector_type vtype_;
  tensor_t data_;
  tensor_t grad_;
  node *prev_;                // previous node, "producer" of this tensor
  std::vector<node *> next_;  // next nodes, "consumers" of this tensor
};

}  // namespace mnn
