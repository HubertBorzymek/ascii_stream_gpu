#pragma once

#include "../EffectBase.h"

#include <d3d11.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// Forward declaration to avoid including CUDA headers in .h
struct cudaGraphicsResource;

// AsciiEffect
// GPU effect that will apply ASCII stylization to the input frame.
// Next step: D3D11 <-> CUDA interop (passthrough first), then ASCII kernel.
class AsciiEffect final : public EffectBase
{
public:
    AsciiEffect() = default;
    ~AsciiEffect() override = default;

    AsciiEffect(const AsciiEffect&) = delete;
    AsciiEffect& operator=(const AsciiEffect&) = delete;

protected:
    void OnShutdown() override;

    ID3D11Texture2D* ProcessImpl(ID3D11Texture2D* inputTex) override;

private:
    void EnsureInputMatches(ID3D11Texture2D* inputTex);
    void EnsureOutputMatches(ID3D11Texture2D* inputTex);

    // CUDA interop helpers (implemented later in .cpp)
    void ReleaseCudaResources();
    void EnsureCudaRegistered(ID3D11Texture2D* inputTex);

private:
    // Internal input copy owned by this effect (interop-friendly).
    ComPtr<ID3D11Texture2D> m_inputTex;
    bool m_inputDescValid = false;
    D3D11_TEXTURE2D_DESC m_inputDesc{};

    // Output owned by this effect (will be rendered by D3DRenderer).
    ComPtr<ID3D11Texture2D> m_outputTex;
    bool m_outputDescValid = false;
    D3D11_TEXTURE2D_DESC m_outputDesc{};

    // CUDA interop state
    cudaGraphicsResource* m_cudaInputRes = nullptr;
    cudaGraphicsResource* m_cudaOutputRes = nullptr;

    bool m_cudaReady = false;
};
