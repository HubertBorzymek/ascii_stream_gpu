#include "EffectBase.h"

#include <stdexcept>

void EffectBase::Initialize(ComPtr<ID3D11Device> deviceIn,
                            ComPtr<ID3D11DeviceContext> contextIn)
{
    if (!deviceIn)
        throw std::runtime_error("EffectBase::Initialize: device is null");
    if (!contextIn)
        throw std::runtime_error("EffectBase::Initialize: context is null");

    device = deviceIn;
    context = contextIn;

    enabled = true;
    initialized = true;

    OnInitialize();
}

void EffectBase::Shutdown()
{
    if (!initialized)
        return;

    OnShutdown();

    context.Reset();
    device.Reset();

    initialized = false;
}

ID3D11Texture2D* EffectBase::Process(ID3D11Texture2D* inputTex)
{
    if (!initialized || !enabled)
        return inputTex;

    if (!inputTex)
        return nullptr;

    return ProcessImpl(inputTex);
}
