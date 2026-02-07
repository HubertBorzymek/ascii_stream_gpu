#pragma once

#include <d3d11.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// FrameProcessor
// A thin processing layer that can transform a D3D11 texture into another texture.
// For now it can run in passthrough mode (output == input).
class FrameProcessor
{
public:
    FrameProcessor() = default;
    ~FrameProcessor() = default;

    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;

    // Stores D3D11 device/context used by effects and internal resources.
    void Initialize(ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> context);

    // Releases internal state/resources.
    void Shutdown();

    // Enables/disables processing (when disabled, Process returns input as output).
    void SetEnabled(bool enabled);

    bool IsEnabled() const { return m_enabled; }

    // Process one frame. Returns a texture to be rendered.
    // For now (passthrough), this returns inputTex.
    ID3D11Texture2D* Process(ID3D11Texture2D* inputTex);

private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;

    bool m_enabled = true;
    bool m_initialized = false;
};
