/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/edge.h"
#include "mnn/core/graph/node.h"
#include "mnn/core/layer/layer.h"
#include "mnn/infra/backend.h"

#include <iomanip>

namespace mnn {

void Layer::alloc_input(size_t i) const
{
    prev_[i] = std::make_shared<Edge>(nullptr, in_shape()[i], in_type_[i]);
}

void Layer::alloc_output(size_t i) const
{
    next_[i] = std::make_shared<Edge>(const_cast<Layer*>(this),
            out_shape()[i], out_type_[i]);
}

edgeptr_t Layer::ith_in_node(size_t i)
{
    // in case that the  edge doesn't exist, we create it
    if (!prev_[i])
        alloc_input(i);
    return prev()[i];
}

edgeptr_t Layer::ith_out_node(size_t i)
{
    // in case that the  edge doesn't exist, we create it
    if (!next_[i])
        alloc_output(i);
    return next()[i];
}
edgeptr_t Layer::ith_out_node(size_t i) const
{
    return next()[i];
}

Vector* Layer::get_weight_data(size_t i)
{
    assert(is_trainable_weight(in_type_[i]));
    return &(*(ith_in_node(i)->get_data()))[0];
}

const Vector* Layer::get_weight_data(size_t i) const
{
    assert(is_trainable_weight(in_type_[i]));
    return &(*(const_cast<Layer*>(this)->ith_in_node(i)->get_data()))[0];
}

Layer::Layer(const std::vector<VectorType> &in_type,
        const std::vector<VectorType> &out_type) : Node(in_type.size(),
        out_type.size()), initialized_(false), parallelize_(true), in_channels_(
        in_type.size()), out_channels_(out_type.size()), in_type_(in_type), out_type_(
        out_type), backend_type_(BackendType::CPU)
{
    weight_init_ = std::make_shared<weight_init::Xavier>();
    bias_init_ = std::make_shared<weight_init::Constant>();
    trainable_ = true;
}

void Layer::set_parallelize(bool parallelize)
{
    parallelize_ = parallelize;
}

void Layer::set_backend_type(BackendType backend_type)
{
    backend_type_ = backend_type;
}

bool Layer::parallelize() const
{
    return parallelize_;
}
BackendType Layer::engine() const
{
    return backend_type_;
}

size_t Layer::in_channels() const
{
    return in_channels_;
}
size_t Layer::out_channels() const
{
    return out_channels_;
}

size_t Layer::in_data_size() const
{
    return sumif(in_shape(), [&](size_t i) {  // NOLINT
                return in_type_[i] == VectorType::DATA;
            }, [](const Shape3d &s) {return s.size();});
}

size_t Layer::out_data_size() const
{
    return sumif(out_shape(), [&](size_t i) {  // NOLINT
                return out_type_[i] == VectorType::DATA;
            }, [](const Shape3d &s) {return s.size();});
}

std::vector<Shape3d> Layer::in_data_shape()
{
    return filter(in_shape(), [&](size_t i) {  // NOLINT
                return in_type_[i] == VectorType::DATA;
            });
}

std::vector<Shape3d> Layer::out_data_shape()
{
    return filter(out_shape(), [&](size_t i) {  // NOLINT
                return out_type_[i] == VectorType::DATA;
            });
}

std::vector<const Vector*> Layer::weights() const
{
    std::vector<const Vector*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<Vector*> Layer::weights()
{
    std::vector<Vector*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(get_weight_data(i));
        }
    }
    return v;
}

std::vector<Matrix*> Layer::weights_grads()
{
    std::vector<Matrix*> v;
    for (size_t i = 0; i < in_channels_; i++) {
        if (is_trainable_weight(in_type_[i])) {
            v.push_back(ith_in_node(i)->get_gradient());
        }
    }
    return v;
}

std::vector<edgeptr_t> Layer::inputs()
{
    std::vector<edgeptr_t> nodes(in_channels_);
    for (size_t i = 0; i < in_channels_; i++) {
        nodes[i] = ith_in_node(i);
    }
    return nodes;
}

std::vector<edgeptr_t> Layer::outputs()
{
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
        nodes[i] = ith_out_node(i);
    }
    return nodes;
}

std::vector<edgeptr_t> Layer::outputs() const
{
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
        nodes[i] = const_cast<Layer*>(this)->ith_out_node(i);
    }
    return nodes;
}

void Layer::set_out_grads(const std::vector<const Vector*> *grad, size_t cnt)
{
    MNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] != VectorType::DATA)
            continue;
        Matrix &dst_grad = *ith_out_node(i)->get_gradient();
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

void Layer::set_in_data(const std::vector<const Vector*> *data, size_t cnt)
{
    MNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < in_channels_; i++) {
        if (in_type_[i] != VectorType::DATA)
            continue;
        Matrix &dst_data = *ith_in_node(i)->get_data();
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

void Layer::output(std::vector<const Matrix*> &out) const
{
    out.clear();
    for (size_t i = 0; i < out_channels_; i++) {
        if (out_type_[i] == VectorType::DATA) {
            out.push_back(ith_out_node(i)->get_data());
        }
    }
}

void Layer::set_in_shape(const Shape3d &in_shape)
{
    MNN_UNREFERENCED_PARAMETER(in_shape);
    throw MnnError("Can't set shape. Shape inferring not applicable for this "
            "layer (yet).");
}

size_t Layer::fan_in_size() const
{
    return in_shape()[0].width_;
}

size_t Layer::fan_in_size(size_t) const
{
    return fan_in_size();  // fallback to single weight matrix.
}

size_t Layer::fan_out_size() const
{
    return out_shape()[0].width_;
}

size_t Layer::fan_out_size(size_t) const
{
    return fan_out_size();  // fallback to single weight matrix
}

std::vector<Matrix> Layer::backward(const std::vector<Matrix> &out_grads)
{  // for test
    setup(false);

    std::vector<std::vector<const Vector*>> grads2;
    grads2.resize(out_grads.size());
    for (size_t i = 0; i < out_grads.size(); ++i) {
        grads2[i].resize(out_grads[i].size());
        for (size_t j = 0; j < out_grads[i].size(); ++j) {
            grads2[i][j] = &out_grads[i][j];
        }
    }

    set_out_grads(&grads2[0], grads2.size());
    backward();
    return map_<Matrix>(inputs(),
            [](edgeptr_t e) {return *e->get_gradient();});
}

void Layer::forward()
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

void Layer::backward()
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

void Layer::setup(bool reset_weight)
{
    if (in_shape().size() != in_channels_
            || out_shape().size() != out_channels_) {
        throw MnnError("Connection mismatch at setup layer");
    }

    for (size_t i = 0; i < out_channels_; i++) {
        if (!next_[i]) {
            next_[i] = std::make_shared<Edge>(this, out_shape()[i],
                    out_type_[i]);
        }
    }

    if (reset_weight || !initialized_) {
        init_weight();
    }
}

void Layer::init_weight()
{
    if (!trainable_) {
        initialized_ = true;
        return;
    }

    for (size_t i = 0; i < in_channels_; i++) {
        switch (in_type_[i]) {
        case VectorType::WEIGHT:
            weight_init_->fill(get_weight_data(i), fan_in_size(i),
                    fan_out_size(i));
            break;
        case VectorType::BIAS:
            bias_init_->fill(get_weight_data(i), fan_in_size(i),
                    fan_out_size(i));
            break;
        default:
            break;
        }
    }
    initialized_ = true;
}

void Layer::clear_grads()
{
    for (size_t i = 0; i < in_type_.size(); i++) {
        ith_in_node(i)->clear_grads();
    }
}

void Layer::update_weight(Optimizer *o)
{
    auto &diff = weights_diff_;
    for (size_t i = 0; i < in_type_.size(); i++) {
        if (trainable() && is_trainable_weight(in_type_[i])) {
            Vector &target = *get_weight_data(i);
            ith_in_node(i)->merge_grads(&diff);
            Float rcp_batch_size = Float(1.0)
                    / Float(ith_in_node(i)->get_data()->size());
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

bool Layer::has_same_weights(const Layer &rhs, Float eps) const
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

void Layer::set_sample_count(size_t sample_count)
{
    auto resize = [sample_count](Matrix *tensor) {
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

std::vector<VectorType> Layer::in_types() const { return in_type_; }
std::vector<VectorType> Layer::out_types() const { return out_type_; }

void Layer::set_trainable(bool trainable) { trainable_ = trainable; }

bool Layer::trainable() const { return trainable_; }

std::pair<Float, Float> Layer::out_value_range() const {
  return {Float{0.0}, Float{1.0}};
}

void connect(Layer *head, Layer *tail, size_t head_index = 0,
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
        throw MnnError("output edge must not be null");
    }

    tail->prev_[tail_index] = head->next_[head_index];
    tail->prev_[tail_index]->add_next_node(tail);
}

Layer& operator<<(Layer &lhs, Layer &rhs)
{
    connect(&lhs, &rhs);
    return rhs;
}

void data_mismatch(const Layer &layer, const Vector &data)
{
    std::ostringstream os;

    os << std::endl;
    os << "data dimension:    " << data.size() << "\n";
    os << "network dimension: " << layer.in_data_size() << "("
            << layer.layer_type() << ":" << layer.in_shape() << ")\n";

    std::string detail_info = os.str();

    throw MnnError("input dimension mismatch!" + detail_info);
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

    throw MnnError("width/height not multiple of pooling size" + detail_info);
}

void connection_mismatch(const Layer &from, const Layer &to)
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

    throw MnnError("layer dimension mismatch!" + detail_info);
}

}  // namespace mnn
