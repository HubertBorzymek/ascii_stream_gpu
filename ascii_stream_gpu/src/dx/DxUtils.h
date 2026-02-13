#pragma once

#include <Windows.h>
#include <stdexcept>

inline void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr))
        throw std::runtime_error(msg);
}
