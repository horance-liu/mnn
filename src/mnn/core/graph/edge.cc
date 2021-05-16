/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/edge.h"

namespace mnn {

Edge::Edge(Node *prev, const Shape3d &shape, VectorType vtype) : shape_(shape), vtype_(
        vtype), data_( { Vector(shape.size()) }), grad_(
        { Vector(shape.size()) }), prev_(prev)
{
}

void Edge::merge_grads(Vector *dst)
{
    assert(!grad_.empty());
    const auto &grad_head = grad_[0];
    size_t sz = grad_head.size();
    dst->resize(sz);
    Float *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.size(); sample < sample_count;
            ++sample) {
        // dst += grad_[sample]
        vectorize::reduce < Float > (&grad_[sample][0], sz, pdst);
    }
}

void Edge::clear_grads()
{
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
            ++sample) {
        auto &g = grad_[sample];
        vectorize::fill(&g[0], g.size(), Float { 0 });
    }
}

Matrix* Edge::get_data()
{
    return &data_;
}

const Matrix* Edge::get_data() const
{
    return &data_;
}

Matrix* Edge::get_gradient()
{
    return &grad_;
}

const Matrix* Edge::get_gradient() const
{
    return &grad_;
}

const std::vector<Node*>& Edge::next() const
{
    return next_;
}
Node* Edge::prev()
{
    return prev_;
}
const Node* Edge::prev() const
{
    return prev_;
}

const Shape3d& Edge::shape() const
{
    return shape_;
}

VectorType Edge::vtype() const
{
    return vtype_;
}

void Edge::add_next_node(Node *next)
{
    next_.push_back(next);
}

}  // namespace mnn
