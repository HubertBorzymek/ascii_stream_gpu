#include "AsciiEffect.h"

#include <stdexcept>

void AsciiEffect::Initialize(ComPtr<ID3D11Device> device,
    ComPtr<ID3D11DeviceContext> context)
{
    if (!device)
        throw std::runtime_error("AsciiEffect::Initialize: device is null");
    if (!context)
        throw std::runtime_error("AsciiEffect::Initialize: context is null");

    m_device = device;
    m_context = context;

    m_outputTex.Reset();
    m_outputDescValid = false;

    m_initialized = true;
}

void AsciiEffect::Shutdown()
{
    m_outputTex.Reset();
    m_outputDescValid = false;

    m_context.Reset();
    m_device.Reset();

    m_initialized = false;
}

void AsciiEffect::SetEnabled(bool enabled)
{
    m_enabled = enabled;
}

bool AsciiEffect::IsEnabled() const
{
    return m_enabled;
}

void AsciiEffect::EnsureOutputMatches(ID3D11Texture2D* inputTex)
{
    if (!inputTex)
        return;

    D3D11_TEXTURE2D_DESC inDesc{};
    inputTex->GetDesc(&inDesc);

    // We want output to match input dimensions and format for now.
    // Later, we may choose a fixed output format (e.g., BGRA8) depending on CUDA kernel needs.
    const bool needCreate =
        !m_outputTex ||
        !m_outputDescValid ||
        m_outputDesc.Width != inDesc.Width ||
        m_outputDesc.Height != inDesc.Height ||
        m_outputDesc.Format != inDesc.Format;

    if (!needCreate)
        return;

    // Release old output.
    m_outputTex.Reset();

    D3D11_TEXTURE2D_DESC outDesc = inDesc;

    // Output is a GPU-only texture we can write to via CopyResource now,
    // and later via CUDA interop or compute.
    outDesc.BindFlags = 0;              // not sampled directly; renderer will copy into its stableTex anyway
    outDesc.CPUAccessFlags = 0;
    outDesc.Usage = D3D11_USAGE_DEFAULT;
    outDesc.MiscFlags = 0;
    outDesc.MipLevels = 1;
    outDesc.ArraySize = 1;
    outDesc.SampleDesc.Count = 1;
    outDesc.SampleDesc.Quality = 0;

    // Some captured textures may have BindFlags = 0 already; that's fine.
    // We keep it simple: just allocate a DEFAULT texture for output.
    HRESULT hr = m_device->CreateTexture2D(&outDesc, nullptr, &m_outputTex);
    if (FAILED(hr))
        throw std::runtime_error("AsciiEffect::EnsureOutputMatches: CreateTexture2D failed");

    m_outputDesc = outDesc;
    m_outputDescValid = true;
}

ID3D11Texture2D* AsciiEffect::Process(ID3D11Texture2D* inputTex)
{
    if (!m_initialized || !m_enabled)
        return inputTex;

    if (!inputTex)
        return nullptr;

    EnsureOutputMatches(inputTex);

    // GPU -> GPU copy to output (passthrough via copy for now).
    m_context->CopyResource(m_outputTex.Get(), inputTex);

    return m_outputTex.Get();
}
