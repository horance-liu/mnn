/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/edge.h"

namespace mnn {

edge::edge(node *prev, const shape3d &shape, vector_type vtype) : shape_(shape), vtype_(
        vtype), data_( { vec_t(shape.size()) }), grad_(
        { vec_t(shape.size()) }), prev_(prev)
{
}

void edge::merge_grads(vec_t *dst)
{
    assert(!grad_.empty());
    const auto &grad_head = grad_[0];
    size_t sz = grad_head.size();
    dst->resize(sz);
    float_t *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.size(); sample < sample_count;
            ++sample) {
        // dst += grad_[sample]
        vectorize::reduce < float_t > (&grad_[sample][0], sz, pdst);
    }
}

void edge::clear_grads()
{
    for (size_t sample = 0, sample_count = grad_.size(); sample < sample_count;
            ++sample) {
        auto &g = grad_[sample];
        vectorize::fill(&g[0], g.size(), float_t { 0 });
    }
}

tensor_t* edge::get_data()
{
    return &data_;
}

const tensor_t* edge::get_data() const
{
    return &data_;
}

tensor_t* edge::get_gradient()
{
    return &grad_;
}

const tensor_t* edge::get_gradient() const
{
    return &grad_;
}

const std::vector<node*>& edge::next() const
{
    return next_;
}
node* edge::prev()
{
    return prev_;
}
const node* edge::prev() const
{
    return prev_;
}

const shape3d& edge::shape() const
{
    return shape_;
}

vector_type edge::vtype() const
{
    return vtype_;
}

void edge::add_next_node(node *next)
{
    next_.push_back(next);
}

}  // namespace mnn
