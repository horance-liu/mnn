/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/layer/layer.h"
#include "mnn/core/graph/edge.h"
#include "mnn/core/graph/node.h"
#include "mnn/infra/backend.h"

#include <iomanip>

namespace mnn {

void layer::alloc_input(size_t i) const
{
    prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
}

void layer::alloc_output(size_t i) const
{
    next_[i] = std::make_shared<edge>(const_cast<layer*>(this),
            out_shape()[i], out_type_[i]);
}

edgeptr_t layer::ith_in_node(size_t i)
{
    // in case that the  edge doesn't exist, we create it
    if (!prev_[i])
        alloc_input(i);
    return prev()[i];
}

edgeptr_t layer::ith_out_node(size_t i)
{
    // in case that the  edge doesn't exist, we create it
    if (!next_[i])
        alloc_output(i);
    return next()[i];
}
edgeptr_t layer::ith_out_node(size_t i) const
{
    return next()[i];
}

vec_t* layer::get_weight_data(size_t i)
{
    assert(is_trainable_weight(in_type_[i]));
    return &(*(ith_in_node(i)->get_data()))[0];
}

const vec_t* layer::get_weight_data(size_t i) const
{
    assert(is_trainable_weight(in_type_[i]));
    return &(*(const_cast<layer*>(this)->ith_in_node(i)->get_data()))[0];
}

layer::layer(const std::vector<vector_type> &in_type,
        const std::vector<vector_type> &out_type) : node(in_type.size(),
        out_type.size()), initialized_(false), parallelize_(true), in_channels_(
        in_type.size()), out_channels_(out_type.size()), in_type_(in_type), out_type_(
        out_type), backend_type_(backend_t::cpu)
{
    weight_init_ = std::make_shared<weight_init::xavier>();
    bias_init_ = std::make_shared<weight_init::constant>();
    trainable_ = true;
}

void layer::set_parallelize(bool parallelize)
{
    parallelize_ = parallelize;
}

void layer::set_backend_type(backend_t backend_type)
{
    backend_type_ = backend_type;
}

bool layer::parallelize() const
{
    return parallelize_;
}
backend_t layer::engine() const
{
    return backend_type_;
}

size_t layer::in_channels() const
{
    return in_channels_;
}
size_t layer::out_channels() const
{
    return out_channels_;
}

size_t layer::in_data_size() const
{
    return sumif(in_shape(), [&](size_t i) {  // NOLINT
                return in_type_[i] == vector_type::data;
            }, [](const shape3d &s) {return s.size();});
}

size_t layer::out_data_size() const
{
    return sumif(out_shape(), [&](size_t i) {  // NOLINT
                return out_type_[i] == vector_type::data;
            }, [](const shape3d &s) {return s.size();});
}

std::vector<shape3d> layer::in_data_shape()
{
    return filter(in_shape(), [&](size_t i) {  // NOLINT
                return in_type_[i] == vector_type::data;
            });
}

std::vector<shape3d> layer::out_data_shape()
{
    return filter(out_shape(), [&](size_t i) {  // NOLINT
                return out_type_[i] == vector_type::data;
            });
}

std::vector<const vec_t*> layer::weights() const
{
    std::vector<const vec_t*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<vec_t*> layer::weights()
{
    std::vector<vec_t*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<tensor_t*> layer::weights_grads()
{
    std::vector<tensor_t*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(ith_in_node(i)->get_gradient());
        }
    }
    return v;
}

std::vector<edgeptr_t> layer::inputs()
{
    std::vector<edgeptr_t> nodes(in_channels_);
    for (size_t i = 0; i < in_channels_; i++) {
        nodes[i] = ith_in_node(i);
    }
    return nodes;
}

std::vector<edgeptr_t> layer::outputs()
{
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
        nodes[i] = ith_out_node(i);
    }
    return nodes;
}

std::vector<edgeptr_t> layer::outputs() const
{
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
        nodes[i] = const_cast<layer*>(this)->ith_out_node(i);
    }
    return nodes;
}

void layer::set_out_grads(const std::vector<const vec_t*> *grad, size_t cnt)
{
    MNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] != vector_type::data)
            continue;
        tensor_t &dst_grad = *ith_out_node(i)->get_gradient();
        assert(n < cnt);
        const auto &src_grad = grad[n++];
        size_t sz = src_grad.size();
        dst_grad.resize(sz);
        for (size_t j = 0; j < sz; ++j) {
            assert(dst_grad[j].size() == src_grad[j]->size());
            dst_grad[j] = *src_grad[j];
        }
    }
}

void layer::set_in_data(const std::vector<const vec_t*> *data, size_t cnt)
{
    MNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < in_channels_; i++) {
        if (in_type_[i] != vector_type::data)
            continue;
        tensor_t &dst_data = *ith_in_node(i)->get_data();
        size_t in_size = ith_in_node(i)->shape().size();
        assert(n < cnt);
        const auto &src_data = data[n++];
        size_t sz = src_data.size();
        dst_data.resize(sz);

        MNN_UNREFERENCED_PARAMETER(in_size);

        for (size_t j = 0; j < sz; ++j) {
            assert(src_data[j]->size() == in_size); // checking if training data is consistent with layer shape
            dst_data[j] = *src_data[j];
        }
    }
}

void layer::output(std::vector<const tensor_t*> &out) const
{
    out.clear();
    for (size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] == vector_type::data) {
            out.push_back(ith_out_node(i)->get_data());
        }
    }
}

void layer::set_in_shape(const shape3d &in_shape)
{
    MNN_UNREFERENCED_PARAMETER(in_shape);
    throw nn_error("Can't set shape. Shape inferring not applicable for this "
            "layer (yet).");
}

size_t layer::fan_in_size() const
{
    return in_shape()[0].width_;
}

size_t layer::fan_in_size(size_t) const
{
    return fan_in_size();  // fallback to single weight matrix.
}

size_t layer::fan_out_size() const
{
    return out_shape()[0].width_;
}

size_t layer::fan_out_size(size_t) const
{
    return fan_out_size();  // fallback to single weight matrix
}

std::vector<tensor_t> layer::backward(const std::vector<tensor_t> &out_grads)
{  // for test
    setup(false);

    std::vector<std::vector<const vec_t*>> grads2;
    grads2.resize(out_grads.size());
    for (size_t i = 0; i < out_grads.size(); ++i) {
        grads2[i].resize(out_grads[i].size());
        for (size_t j = 0; j < out_grads[i].size(); ++j) {
            grads2[i][j] = &out_grads[i][j];
        }
    }

    set_out_grads(&grads2[0], grads2.size());
    backward();
    return map_<tensor_t>(inputs(),
            [](edgeptr_t e) {return *e->get_gradient();});
}

void layer::forward()
{
    fwd_in_data_.resize(in_channels_);
    fwd_out_data_.resize(out_channels_);

    for (size_t i = 0; i < in_channels_; i++) {
        fwd_in_data_[i] = ith_in_node(i)->get_data();
    }

    set_sample_count(fwd_in_data_[0]->size());

    for (size_t i = 0; i < out_channels_; i++) {
        fwd_out_data_[i] = ith_out_node(i)->get_data();
        ith_out_node(i)->clear_grads();
    }

    forward_propagation(fwd_in_data_, fwd_out_data_);
}

void layer::backward()
{
    bwd_in_data_.resize(in_channels_);
    bwd_in_grad_.resize(in_channels_);
    bwd_out_data_.resize(out_channels_);
    bwd_out_grad_.resize(out_channels_);

    // organize input/output vectors from storage
    for (size_t i = 0; i < in_channels_; i++) {
        const auto &nd = ith_in_node(i);
        bwd_in_data_[i] = nd->get_data();
        bwd_in_grad_[i] = nd->get_gradient();
    }
    for (size_t i = 0; i < out_channels_; i++) {
        const auto &nd = ith_out_node(i);
        bwd_out_data_[i] = nd->get_data();
        bwd_out_grad_[i] = nd->get_gradient();
    }
    back_propagation(bwd_in_data_, bwd_out_data_, bwd_out_grad_, bwd_in_grad_);
}

void layer::setup(bool reset_weight)
{
    if (in_shape().size() != in_channels_
            || out_shape().size() != out_channels_) {
        throw nn_error("Connection mismatch at setup layer");
    }

    for (size_t i = 0; i < out_channels_; i++) {
        if (!next_[i]) {
            next_[i] = std::make_shared<edge>(this, out_shape()[i],
                    out_type_[i]);
        }
    }

    if (reset_weight || !initialized_) {
        init_weight();
    }
}

void layer::init_weight()
{
    if (!trainable_) {
        initialized_ = true;
        return;
    }

    for (size_t i = 0; i < in_channels_; i++) {
        switch (in_type_[i]) {
        case vector_type::weight:
            weight_init_->fill(get_weight_data(i), fan_in_size(i),
                    fan_out_size(i));
            break;
        case vector_type::bias:
            bias_init_->fill(get_weight_data(i), fan_in_size(i),
                    fan_out_size(i));
            break;
        default:
            break;
        }
    }
    initialized_ = true;
}

void layer::clear_grads()
{
    for (size_t i = 0; i < in_type_.size(); i++) {
        ith_in_node(i)->clear_grads();
    }
}

void layer::update_weight(optimizer *o)
{
    auto &diff = weights_diff_;
    for (size_t i = 0; i < in_type_.size(); i++) {
        if (trainable() && is_trainable_weight(in_type_[i])) {
            vec_t &target = *get_weight_data(i);
            ith_in_node(i)->merge_grads(&diff);
            float_t rcp_batch_size = float_t(1.0)
                    / float_t(ith_in_node(i)->get_data()->size());
            for (size_t j = 0; j < diff.size(); ++j) {
                diff[j] *= rcp_batch_size;
            }
            bool parallelize = (target.size() >= 512);
            o->update(diff, target, parallelize);
        }
    }
    clear_grads();
    post_update();
}

bool layer::has_same_weights(const layer &rhs, float_t eps) const
{
    auto w1 = weights();
    auto w2 = rhs.weights();
    if (w1.size() != w2.size())
        return false;

    for (size_t i = 0; i < w1.size(); i++) {
        if (w1[i]->size() != w2[i]->size())
            return false;

        for (size_t j = 0; j < w1[i]->size(); j++) {
            if (std::abs(w1[i]->at(j) - w2[i]->at(j)) > eps)
                return false;
        }
    }
    return true;
}

void layer::set_sample_count(size_t sample_count)
{
    auto resize = [sample_count](tensor_t *tensor) {
        tensor->resize(sample_count, (*tensor)[0]);
    };

    for (size_t i = 0; i < in_channels_; i++) {
        if (!is_trainable_weight(in_type_[i])) {
            resize(ith_in_node(i)->get_data());
        }
        resize(ith_in_node(i)->get_gradient());
    }

    for (size_t i = 0; i < out_channels_; i++) {
        if (!is_trainable_weight(out_type_[i])) {
            resize(ith_out_node(i)->get_data());
        }
        resize(ith_out_node(i)->get_gradient());
    }
}

std::vector<vector_type> layer::in_types() const { return in_type_; }
std::vector<vector_type> layer::out_types() const { return out_type_; }

void layer::set_trainable(bool trainable) { trainable_ = trainable; }

bool layer::trainable() const { return trainable_; }

std::pair<float_t, float_t> layer::out_value_range() const {
  return {float_t{0.0}, float_t{1.0}};
}

void connect(layer *head, layer *tail, size_t head_index = 0,
        size_t tail_index = 0)
{
    auto out_shape = head->out_shape()[head_index];
    auto in_shape = tail->in_shape()[tail_index];

    head->setup(false);

    if (in_shape.size() == 0) {
        tail->set_in_shape(out_shape);
        in_shape = out_shape;
    }

    if (out_shape.size() != in_shape.size()) {
        connection_mismatch(*head, *tail);
    }

    if (!head->next_[head_index]) {
        throw nn_error("output edge must not be null");
    }

    tail->prev_[tail_index] = head->next_[head_index];
    tail->prev_[tail_index]->add_next_node(tail);
}

layer& operator<<(layer &lhs, layer &rhs)
{
    connect(&lhs, &rhs);
    return rhs;
}

void data_mismatch(const layer &layer, const vec_t &data)
{
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << "\n";
    os << "network dimension: " << layer.in_data_size() << "("
            << layer.layer_type() << ":" << layer.in_shape() << ")\n";

    std::string detail_info = os.str();

    throw nn_error("input dimension mismatch!" + detail_info);
}

void pooling_size_mismatch(size_t in_width, size_t in_height,
        size_t pooling_size_x, size_t pooling_size_y)
{
    std::ostringstream os;

    os << std::endl;
    os << "WxH:" << in_width << "x" << in_height << std::endl;
    os << "pooling-size:" << pooling_size_x << "x" << pooling_size_y
            << std::endl;

    std::string detail_info = os.str();

    throw nn_error("width/height not multiple of pooling size" + detail_info);
}

void connection_mismatch(const layer &from, const layer &to)
{
    std::ostringstream os;

    os << std::endl;
    os << "output size of Nth layer must be equal to input of (N+1)th layer\n";

    os << "layerN:   " << std::setw(12) << from.layer_type() << " in:"
            << from.in_data_size() << "(" << from.in_shape() << "), " << "out:"
            << from.out_data_size() << "(" << from.out_shape() << ")\n";

    os << "layerN+1: " << std::setw(12) << to.layer_type() << " in:"
            << to.in_data_size() << "(" << to.in_shape() << "), " << "out:"
            << to.out_data_size() << "(" << to.out_shape() << ")\n";

    os << from.out_data_size() << " != " << to.in_data_size() << std::endl;
    std::string detail_info = os.str();

    throw nn_error("layer dimension mismatch!" + detail_info);
}

}  // namespace mnn
