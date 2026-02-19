#pragma once

#include <Windows.h>
#include <stdint.h>

#include "window/IWindowMessageHandler.h"
#include "window/OverlayController.h"
#include "appState/AppState.h"

#include "ui/ControlPanel.h"
#include "dx/DxContext.h"
#include "render/D3dRenderer.h"
#include "render/SwapChainRenderTarget.h"
#include "capture/ScreenCapture.h"
#include "processing/core/FrameProcessor.h"

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
    void RestartCaptureSelected();
    void InitFrameProcessor();

    void ShutdownImGui();
    void ShutdownCapture();
    void ShutdownFrameProcessor();
    void ShutdownHotkeys();

    void RenderMain();
    void RenderPanel();
    void UpdateFpsTitle();

    void ApplyOverlayFromState();

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

    // UI + window controllers
    ControlPanel m_controlPanel{};
    ControlPanel::Callbacks m_uiCallbacks{};
    OverlayController m_overlay{};

    // Timing / debug
    uint64_t m_frameCount = 0;
    uint64_t m_lastTickMs = 0;

    // State
    AppState m_state{};
    bool m_initialized = false;

    // Runtime selection (App-owned, not UI-owned)
    HMONITOR m_selectedMonitor = nullptr;
};
