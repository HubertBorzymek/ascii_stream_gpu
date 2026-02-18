#include "FrameProcessor.h"

#include <stdexcept>

#include "../effects/ASCII/AsciiEffect.h"

void FrameProcessor::Initialize(ComPtr<ID3D11Device> deviceIn,
    ComPtr<ID3D11DeviceContext> contextIn)
{
    if (!deviceIn)
        throw std::runtime_error("FrameProcessor::Initialize: device is null");
    if (!contextIn)
        throw std::runtime_error("FrameProcessor::Initialize: context is null");

    device = deviceIn;
    context = contextIn;

    // Create default effect: ASCII.
    effect = std::make_unique<AsciiEffect>();
    effect->Initialize(device, context);

    initialized = true;
}

void FrameProcessor::Shutdown()
{
    if (effect)
    {
        effect->Shutdown();
        effect.reset();
    }

    context.Reset();
    device.Reset();

    initialized = false;
}

void FrameProcessor::SetEnabled(bool enabledIn)
{
    enabled = enabledIn;
}

void FrameProcessor::SetEffectEnabled(bool enabledIn)
{
    if (effect)
        effect->SetEnabled(enabledIn);
}

bool FrameProcessor::IsEffectEnabled() const
{
    return effect ? effect->IsEnabled() : false;
}

void FrameProcessor::ApplyAsciiSettings(const AsciiSettings& settings)
{
    // First call: always apply
    const bool firstApply = !m_hasPrevAsciiSettings;

    // Enabled
    if (firstApply || (settings.enabled != m_prevAsciiSettings.enabled))
    {
        SetEffectEnabled(settings.enabled);
    }

    // Tint (BGR8)
    const bool tintChanged =
        firstApply ||
        (settings.tintB != m_prevAsciiSettings.tintB) ||
        (settings.tintG != m_prevAsciiSettings.tintG) ||
        (settings.tintR != m_prevAsciiSettings.tintR);

    if (tintChanged)
    {
        // Forward only if the active effect is AsciiEffect.
        // (Later, when you have multiple effects, this will be routed differently.)
        if (auto* ascii = dynamic_cast<AsciiEffect*>(effect.get()))
        {
            ascii->SetTintBgr(settings.tintB, settings.tintG, settings.tintR);
        }
    }

    // Edge params (0..1)
    const bool edgeParamsChanged =
        firstApply ||
        (settings.edgeThreshold != m_prevAsciiSettings.edgeThreshold) ||
        (settings.coherenceThreshold != m_prevAsciiSettings.coherenceThreshold);

    if (edgeParamsChanged)
    {
        if (auto* ascii = dynamic_cast<AsciiEffect*>(effect.get()))
        {
            ascii->SetEdgeParams(settings.edgeThreshold, settings.coherenceThreshold);
        }
    }

    m_prevAsciiSettings = settings;
    m_hasPrevAsciiSettings = true;
}

ID3D11Texture2D* FrameProcessor::Process(ID3D11Texture2D* inputTex)
{
    if (!initialized || !enabled)
        return inputTex;

    if (!inputTex)
        return nullptr;

    if (!effect)
        return inputTex;

    return effect->Process(inputTex);
}
