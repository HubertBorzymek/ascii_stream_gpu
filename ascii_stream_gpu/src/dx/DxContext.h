#pragma once

#include <d3d11.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// DxContext
// Owns the D3D11 device and immediate context used by the application.
struct DxContext
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
};

// Creates a D3D11 device + immediate context (hardware GPU).
// BGRA support is enabled for interoperability with capture/render formats.
// Throws std::runtime_error on failure.
DxContext CreateDxContext();
