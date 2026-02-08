#pragma once

#include "IEffect.h"

#include <wrl.h>

using Microsoft::WRL::ComPtr;

// EffectBase
// Convenience base class for GPU effects.
// Implements common lifetime, enable flags, and the standard Process() guard flow.
// Derived effects only implement ProcessImpl() (and optionally OnInitialize/OnShutdown).
class EffectBase : public IEffect
{
public:
    EffectBase() = default;
    ~EffectBase() override = default;

    EffectBase(const EffectBase&) = delete;
    EffectBase& operator=(const EffectBase&) = delete;

    // Implemented in EffectBase.cpp
    void Initialize(ComPtr<ID3D11Device> deviceIn,
        ComPtr<ID3D11DeviceContext> contextIn) override;

    // Implemented in EffectBase.cpp
    void Shutdown() override;

    // Small common methods stay inline
    void SetEnabled(bool enabledIn) override { enabled = enabledIn; }
    bool IsEnabled() const override { return enabled; }

    // Implemented in EffectBase.cpp
    ID3D11Texture2D* Process(ID3D11Texture2D* inputTex) override;

protected:
    // Optional hooks for derived classes.
    virtual void OnInitialize() {}
    virtual void OnShutdown() {}

    // Derived classes implement the actual effect here.
    virtual ID3D11Texture2D* ProcessImpl(ID3D11Texture2D* inputTex) = 0;

protected:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;

    bool enabled = true;
    bool initialized = false;
};
