#pragma once

#include <d3d11.h>
#include <wrl.h>
#include <memory>

#include "../effects/IEffect.h"

using Microsoft::WRL::ComPtr;

// FrameProcessor
// Owns a selected effect and applies it to incoming D3D11 textures.
class FrameProcessor
{
public:
    FrameProcessor() = default;
    ~FrameProcessor() = default;

    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;

    // Stores D3D11 device/context and creates the default effect (ASCII).
    void Initialize(ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> context);

    // Releases effect and internal state/resources.
    void Shutdown();

    // Enables/disables the whole processing stage.
    void SetEnabled(bool enabled);
    bool IsEnabled() const { return enabled; }

    // Enables/disables the active effect (if any).
    void SetEffectEnabled(bool enabled);
    bool IsEffectEnabled() const;

    // Process one frame. Returns a texture to be rendered.
    // If processing is disabled, returns inputTex.
    ID3D11Texture2D* Process(ID3D11Texture2D* inputTex);

private:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;

    std::unique_ptr<IEffect> effect;

    bool enabled = true;
    bool initialized = false;
};
