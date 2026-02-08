#pragma once

#include <d3d11.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// IEffect
// Common interface for all GPU effects in the processing pipeline.
class IEffect
{
public:
    virtual ~IEffect() = default;

    // Called once after D3D device/context creation.
    virtual void Initialize(ComPtr<ID3D11Device> device,
                            ComPtr<ID3D11DeviceContext> context) = 0;

    // Called on shutdown to release internal resources.
    virtual void Shutdown() = 0;

    // Enable/disable effect processing.
    virtual void SetEnabled(bool enabled) = 0;
    virtual bool IsEnabled() const = 0;

    // Process one frame. If effect is disabled, it should return inputTex.
    // Returned pointer must remain valid at least until the caller finishes rendering this frame.
    virtual ID3D11Texture2D* Process(ID3D11Texture2D* inputTex) = 0;
};
