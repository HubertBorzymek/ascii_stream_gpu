#pragma once

#include "../IEffect.h"

#include <wrl.h>

using Microsoft::WRL::ComPtr;

// AsciiEffect
// GPU effect that will apply ASCII stylization to the input frame.
// For now it creates an output texture and copies input -> output (passthrough via copy).
class AsciiEffect final : public IEffect
{
public:
    AsciiEffect() = default;
    ~AsciiEffect() override = default;

    AsciiEffect(const AsciiEffect&) = delete;
    AsciiEffect& operator=(const AsciiEffect&) = delete;

    void Initialize(ComPtr<ID3D11Device> device,
        ComPtr<ID3D11DeviceContext> context) override;

    void Shutdown() override;

    void SetEnabled(bool enabled) override;
    bool IsEnabled() const override;

    ID3D11Texture2D* Process(ID3D11Texture2D* inputTex) override;

private:
    void EnsureOutputMatches(ID3D11Texture2D* inputTex);

private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;

    // Output owned by this effect (will be rendered by D3DRenderer).
    ComPtr<ID3D11Texture2D> m_outputTex;
    bool m_outputDescValid = false;
    D3D11_TEXTURE2D_DESC m_outputDesc{};

    bool m_enabled = true;
    bool m_initialized = false;
};
