// src/dx/dx_context.cpp

#include "DxContext.h"

#include <stdexcept>
#include <iostream>

#include "../dx/DxUtils.h"

DxContext CreateDxContext()
{
    DxContext out{};

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };

    D3D_FEATURE_LEVEL chosen = D3D_FEATURE_LEVEL_11_0;

    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // use GPU
        nullptr,
        flags,
        featureLevels,
        _countof(featureLevels),
        D3D11_SDK_VERSION,
        out.device.GetAddressOf(),
        &chosen,
        out.context.GetAddressOf()
    );

    ThrowIfFailed(hr, "D3D11CreateDevice failed");

    // Optional: debug print (will be visible only if you have a console/debug output attached).
    std::cout << "D3D11 device created. Feature level = 0x"
        << std::hex << chosen << std::dec << "\n";

    return out;
}
