/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 * 
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#ifndef H216BE1CA_7DD2_49B6_9599_32B32814F109
#define H216BE1CA_7DD2_49B6_9599_32B32814F109


#include "mnn/core/graph/node_list.h"

namespace mnn {

class Sequential: public NodeList {
private:
    void backward(const std::vector<Matrix> &first) override;
    std::vector<Matrix> forward(const std::vector<Matrix> &first) override;

public:
    template<typename T>
    void add(T &&layer)
    {
        push_back(std::forward<T>(layer));

        if (nodes_.size() != 1) {
            auto head = nodes_[nodes_.size() - 2];
            auto tail = nodes_[nodes_.size() - 1];
            connect(head, tail, 0, 0);
            auto out = head->outputs();
            auto in = tail->inputs();
        }
        check_connectivity();
    }

private:
    friend class NodeList;

    void check_connectivity();
    std::vector<Matrix> normalize_out(const std::vector<const Matrix*> &out);
};

} // namespace mnn

#endif /* H216BE1CA_7DD2_49B6_9599_32B32814F109 */
