/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/node.h"
#include "mnn/core/graph/edge.h"

namespace mnn {

node::node(size_t in_size, size_t out_size) : prev_(in_size), next_(out_size)
{
}

const std::vector<edgeptr_t>& node::prev() const
{
    return prev_;
}
const std::vector<edgeptr_t>& node::next() const
{
    return next_;
}

size_t node::prev_port(const edge &e) const
{
    auto it = std::find_if(prev_.begin(), prev_.end(),
            [&](edgeptr_t ep) {return ep.get() == &e;});
    return (size_t) std::distance(prev_.begin(), it);
}

size_t node::next_port(const edge &e) const
{
    auto it = std::find_if(next_.begin(), next_.end(),
            [&](edgeptr_t ep) {return ep.get() == &e;});
    return (size_t) std::distance(next_.begin(), it);
}

std::vector<node*> node::prev_nodes() const
{
    std::vector<node*> vecs;
    for (auto &e : prev_) {
        if (e && e->prev()) {
            vecs.insert(vecs.end(), e->prev());
        }
    }
    return vecs;
}

std::vector<node*> node::next_nodes() const
{
    std::vector<node*> vecs;
    for (auto &e : next_) {
        if (e) {
            auto n = e->next();
            vecs.insert(vecs.end(), n.begin(), n.end());
        }
    }
    return vecs;
}

}  // namespace mnn
