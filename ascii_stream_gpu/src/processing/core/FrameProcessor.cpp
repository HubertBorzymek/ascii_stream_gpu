#include "FrameProcessor.h"

#include <stdexcept>

void FrameProcessor::Initialize(ComPtr<ID3D11Device> device,
    ComPtr<ID3D11DeviceContext> context)
{
    if (!device)
        throw std::runtime_error("FrameProcessor::Initialize: device is null");
    if (!context)
        throw std::runtime_error("FrameProcessor::Initialize: context is null");

    m_device = device;
    m_context = context;
    m_initialized = true;
}

void FrameProcessor::Shutdown()
{
    // Release references to D3D objects.
    m_context.Reset();
    m_device.Reset();

    m_initialized = false;
}

void FrameProcessor::SetEnabled(bool enabled)
{
    m_enabled = enabled;
}

ID3D11Texture2D* FrameProcessor::Process(ID3D11Texture2D* inputTex)
{
    // No processing if not initialized or disabled.
    if (!m_initialized || !m_enabled)
        return inputTex;

    // No input -> no output.
    if (!inputTex)
        return nullptr;

    // Passthrough for now.
    // Later this is where effects (e.g. ASCII) will run.
    return inputTex;
}
