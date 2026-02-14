#include "App.h"

#include "Window.h"
#include "WindowRole.h"
#include "Hotkeys.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_win32.h"
#include "../imgui/imgui_impl_dx11.h"

#include <stdexcept>
#include <cstring>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>

#pragma comment(lib, "windowsapp.lib")

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ------------------------------------------------------------
// Destructor
// ------------------------------------------------------------
App::~App()
{
    if (m_initialized)
        Shutdown();
}

// ------------------------------------------------------------
// Initialize
// ------------------------------------------------------------
void App::Initialize(HINSTANCE hInst)
{
    winrt::init_apartment();

    EnsureCaptureSupportedOrThrow();

    CreateWindows(hInst);
    InitHotkeys();
    InitDx();
    InitMainRenderer();
    InitPanelSwapchain();
    InitImGui();
    InitCapture();
    InitFrameProcessor();

    // Sync initial UI state -> backend
    m_frameProcessor.SetEffectEnabled(m_state.asciiEnabled);
    m_prevAsciiEnabled = m_state.asciiEnabled;

    m_lastTickMs = GetTickCount64();
    m_initialized = true;
}

// ------------------------------------------------------------
// Shutdown
// ------------------------------------------------------------
void App::Shutdown()
{
    ShutdownImGui();
    ShutdownFrameProcessor();
    ShutdownCapture();
    ShutdownHotkeys();

    m_initialized = false;
}

// ------------------------------------------------------------
// Running
// ------------------------------------------------------------
bool App::Running() const
{
    return AppIsRunning();
}

// ------------------------------------------------------------
// Pump Windows messages (call regularly, e.g., once per frame)
// ------------------------------------------------------------
void App::PumpMessages()
{
    MSG msg{};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        if (Hotkeys::HandleMessage(msg))
            continue;

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

// ------------------------------------------------------------
// Tick (called every frame)
// ------------------------------------------------------------
void App::Tick()
{
    if (!m_initialized)
        return;

    PumpMessages();

    // Apply UI state -> backend only when changed
    if (m_state.asciiEnabled != m_prevAsciiEnabled)
    {
        m_frameProcessor.SetEffectEnabled(m_state.asciiEnabled);
        m_prevAsciiEnabled = m_state.asciiEnabled;
    }

    RenderMain();
    RenderPanel();
    UpdateFpsTitle();
}

// ------------------------------------------------------------
// Message routing
// ------------------------------------------------------------
bool App::HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, LRESULT& outResult)
{
    // Feed Win32 messages to ImGui for the PANEL window.
    if (hwnd == m_hwndPanel)
    {
        if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
        {
            outResult = 1;
            return true; // message handled
        }
    }
    switch (msg)
    {
    case WM_SIZE:
    {
        int w = LOWORD(lParam);
        int h = HIWORD(lParam);

        if (hwnd == m_hwndMain)
            OnMainResize(w, h);
        else if (hwnd == m_hwndPanel)
            OnPanelResize(w, h);

        return false; // allow fallback (window.cpp) to also update winW/winH
    }

    default:
        break;
    }

    return false;
}

// ------------------------------------------------------------
// Private helpers
// ------------------------------------------------------------
void App::EnsureCaptureSupportedOrThrow()
{
    if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported())
        throw std::runtime_error("Windows Graphics Capture is not supported.");
}

void App::CreateWindows(HINSTANCE hInst)
{
    constexpr int WIN_W = 1280;
    constexpr int WIN_H = 720;

    m_hwndMain = CreateAppWindow(hInst, WindowRole::Main, WIN_W, WIN_H, 100, 100);
    m_hwndPanel = CreateAppWindow(hInst, WindowRole::Panel, 520, WIN_H, 100 + WIN_W + 10, 100);

    SetWindowMessageHandler(m_hwndMain, this);
    SetWindowMessageHandler(m_hwndPanel, this);
}

// ------------------------------------------------------------
// Initialization helpers
// ------------------------------------------------------------
void App::InitHotkeys()
{
    Hotkeys::Register(m_hwndPanel);
}

void App::InitDx()
{
    m_dx = CreateDxContext();
}

void App::InitMainRenderer()
{
    m_mainRenderer.Initialize(m_hwndMain, m_dx.device, m_dx.context);
}

void App::InitPanelSwapchain()
{
    m_panelRT.Initialize(m_hwndPanel, m_dx.device, m_dx.context);
}

void App::InitImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(m_hwndPanel);
    ImGui_ImplDX11_Init(m_dx.device.Get(), m_dx.context.Get());
}

void App::InitCapture()
{
    m_capture.Start(m_dx.device);
}

void App::InitFrameProcessor()
{
    m_frameProcessor.Initialize(m_dx.device, m_dx.context);
}

// ------------------------------------------------------------
// Shutdown helpers
// ------------------------------------------------------------
void App::ShutdownImGui()
{
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

void App::ShutdownCapture()
{
    m_capture.Stop();
}

void App::ShutdownFrameProcessor()
{
    m_frameProcessor.Shutdown();
}

void App::ShutdownHotkeys()
{
    Hotkeys::Unregister();
}

// ------------------------------------------------------------
// Rendering
// ------------------------------------------------------------
void App::RenderMain()
{
    auto tex = m_capture.LatestTexture();
    ID3D11Texture2D* processed = m_frameProcessor.Process(tex.Get());
    m_mainRenderer.RenderFrame(processed);
}

void App::RenderPanel()
{
    float clearPanel[4] = { 0.05f, 0.05f, 0.05f, 1.0f };
    m_panelRT.Begin(clearPanel);

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Control Panel");

    ImGui::Text("Hello from ImGui!");

    // --- AppState-driven controls ---
    ImGui::Checkbox("Enable ASCII effect", &m_state.asciiEnabled);
    ImGui::Checkbox("Show ImGui demo window", &m_state.showImGuiDemo);

    ImGui::End();

    if (m_state.showImGuiDemo)
        ImGui::ShowDemoWindow(&m_state.showImGuiDemo);

    ImGui::Render();
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    m_panelRT.Present(1);
}

void App::UpdateFpsTitle()
{
    m_frameCount++;

    uint64_t now = GetTickCount64();
    if (now - m_lastTickMs >= 1000)
    {
        wchar_t title[128];
        swprintf_s(title, L"ASCII Stream GPU  |  FPS: %llu", m_frameCount);
        SetWindowText(m_hwndMain, title);

        m_frameCount = 0;
        m_lastTickMs = now;
    }
}

// ------------------------------------------------------------
// Resize
// ------------------------------------------------------------
void App::OnMainResize(int w, int h)
{
    m_mainRenderer.OnResize(w, h);
}

void App::OnPanelResize(int w, int h)
{
    m_panelRT.OnResize(w, h);
}
