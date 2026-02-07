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
