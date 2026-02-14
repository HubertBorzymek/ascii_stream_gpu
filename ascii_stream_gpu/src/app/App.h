#pragma once

#include "IWindowMessageHandler.h"

#include "AppState.h"

#include <Windows.h>
#include <stdint.h>

#include "../dx/DxContext.h"
#include "../render/D3dRenderer.h"
#include "../render/SwapChainRenderTarget.h"
#include "../capture/ScreenCapture.h"
#include "../processing/core/FrameProcessor.h"

// App
// Owns the application lifetime:
// - creates windows
// - initializes D3D device/context
// - initializes renderers, capture, processing, ImGui
// - runs tick loop
// - handles Win32 messages via IWindowMessageHandler (Option 2 routing)
class App final : public IWindowMessageHandler
{
public:
    App() = default;
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    void Initialize(HINSTANCE hInst);
    void Shutdown();

    bool Running() const;
    void PumpMessages();
    void Tick();

    // IWindowMessageHandler
    bool HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, LRESULT& outResult) override;

private:
    void EnsureCaptureSupportedOrThrow();

    void CreateWindows(HINSTANCE hInst);
    void InitHotkeys();
    void InitDx();
    void InitMainRenderer();
    void InitPanelSwapchain();
    void InitImGui();
    void InitCapture();
    void InitFrameProcessor();

    void ShutdownImGui();
    void ShutdownCapture();
    void ShutdownFrameProcessor();
    void ShutdownHotkeys();

    void RenderMain();
    void RenderPanel();
    void UpdateFpsTitle();

    void OnMainResize(int w, int h);
    void OnPanelResize(int w, int h);

private:
    // Windows
    HWND m_hwndMain = nullptr;
    HWND m_hwndPanel = nullptr;

    // Core systems
    DxContext m_dx{};
    D3DRenderer m_mainRenderer{};
    SwapChainRenderTarget m_panelRT{};

    ScreenCapture m_capture{};
    FrameProcessor m_frameProcessor{};

    // Timing / debug
    uint64_t m_frameCount = 0;
    uint64_t m_lastTickMs = 0;

    // State
    AppState m_state{};
    bool m_prevAsciiEnabled = true;
    bool m_initialized = false;
};
