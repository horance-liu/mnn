/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/infra/text_progress.h"

namespace mnn {

text_progress::text_progress(size_t expected_count_, std::ostream &os,
        const std::string &s1, // leading strings
        const std::string &s2, const std::string &s3) : m_os(os), m_s1(s1), m_s2(
        s2), m_s3(s3)
{
    restart(expected_count_);
}

void text_progress::restart(size_t expected_count_)
{
    _count = _next_tic_count = _tic = 0;
    _expected_count = expected_count_;

    m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
            << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
            << std::endl  // endl implies flush, which ensures display
            << m_s3;
    if (!_expected_count)
        _expected_count = 1;  // prevent divide by zero
}                                             // restart

size_t text_progress::operator+=(size_t increment)
{
    if ((_count += increment) >= _next_tic_count) {
        display_tic();
    }
    return _count;
}

size_t text_progress::operator++()
{
    return operator+=(1);
}
size_t text_progress::count() const
{
    return _count;
}
size_t text_progress::expected_count() const
{
    return _expected_count;
}

void text_progress::display_tic()
{
    size_t tics_needed = static_cast<size_t>((static_cast<double>(_count)
            / _expected_count) * 50.0);
    do {
        m_os << '*' << std::flush;
    } while (++_tic < tics_needed);
    _next_tic_count = static_cast<size_t>((_tic / 50.0) * _expected_count);
    if (_count == _expected_count) {
        if (_tic < 51)
            m_os << '*';
        m_os << std::endl;
    }
}

}  // namespace mnn
