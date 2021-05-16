/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 * 
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#ifndef H3FF2C778_4348_42CA_B4E6_CE406A4876E4
#define H3FF2C778_4348_42CA_B4E6_CE406A4876E4

#include "mnn/core/optimizer/optimizer.h"

namespace mnn {

template<int N>
struct stateful_optimizer: public optimizer {
    void reset() override
    {
        for (auto &e : E_)
            e.clear();
    }

protected:
    template<int Index>
    vec_t& get(const vec_t &key)
    {
        if (E_[Index][&key].empty())
            E_[Index][&key].resize(key.size(), float_t());
        return E_[Index][&key];
    }

    std::unordered_map<const vec_t*, vec_t> E_[N];
};

} // namespace mnn

#endif /* H3FF2C778_4348_42CA_B4E6_CE406A4876E4 */
