#include "App.h"

#include "Window.h"
#include "WindowRole.h"
#include "Hotkeys.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_win32.h"
#include "../imgui/imgui_impl_dx11.h"

#include "monitor/MonitorEnumerator.h"

#include <stdexcept>
#include <cstring>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>

#pragma comment(lib, "windowsapp.lib")

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace
{
    // UI state for monitor selection (kept in App.cpp to avoid touching App.h in this step).
    static std::vector<MonitorInfo> g_monitors;
    static std::vector<std::string> g_monitorLabelsUtf8;
    static int g_selectedMonitorIndex = 0;
    static int g_appliedMonitorIndex = -1;
    static bool g_excludeAppWindows = true;
    static bool g_appliedExcludeAppWindows = true;

    static std::string WideToUtf8(const std::wstring& w)
    {
        if (w.empty())
            return {};

        int sizeNeeded = WideCharToMultiByte(
            CP_UTF8, 0,
            w.data(), (int)w.size(),
            nullptr, 0,
            nullptr, nullptr
        );

        if (sizeNeeded <= 0)
            return {};

        std::string out;
        out.resize(sizeNeeded);

        WideCharToMultiByte(
            CP_UTF8, 0,
            w.data(), (int)w.size(),
            out.data(), sizeNeeded,
            nullptr, nullptr
        );

        return out;
    }

    static void RefreshMonitorList()
    {
        g_monitors = MonitorEnumerator::Enumerate();
        g_monitorLabelsUtf8.clear();
        g_monitorLabelsUtf8.reserve(g_monitors.size());

        for (const auto& m : g_monitors)
            g_monitorLabelsUtf8.push_back(WideToUtf8(m.label));

        // Clamp selection
        if (g_monitors.empty())
            g_selectedMonitorIndex = 0;
        else if (g_selectedMonitorIndex < 0)
            g_selectedMonitorIndex = 0;
        else if (g_selectedMonitorIndex >= (int)g_monitors.size())
            g_selectedMonitorIndex = (int)g_monitors.size() - 1;
    }
}

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
    
    RefreshMonitorList();
    g_appliedMonitorIndex = g_selectedMonitorIndex;

    InitHotkeys();
    InitDx();
    InitMainRenderer();
    InitPanelSwapchain();
    InitImGui();
    InitCapture();
    InitFrameProcessor();

    // Sync initial UI state -> backend (delegated to FrameProcessor)
    m_frameProcessor.ApplyAsciiSettings(m_state.ascii);

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

    // Apply UI state -> backend (FrameProcessor handles change detection)
    m_frameProcessor.ApplyAsciiSettings(m_state.ascii);

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
    RestartCaptureSelected();
}

void App::RestartCaptureSelected()
{
    // Apply exclude option first (affects window attributes).
    if (g_excludeAppWindows)
        m_capture.SetExcludedWindows(m_hwndMain, m_hwndPanel);
    else
        m_capture.SetExcludedWindows(nullptr, nullptr);

    // Restart capture on selected monitor (fallback to primary if list empty).
    if (!g_monitors.empty())
        m_capture.Start(m_dx.device, g_monitors[g_selectedMonitorIndex].handle);
    else
        m_capture.Start(m_dx.device);

    // Sync applied state (prevents re-starting every frame).
    g_appliedMonitorIndex = g_selectedMonitorIndex;
    g_appliedExcludeAppWindows = g_excludeAppWindows;
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

    // --- Capture controls ---
    RenderCaptureSettings();

    // --- Effect controls ---
    RenderAsciiSettings();
    //ImGui::Checkbox("Show ImGui demo window", &m_state.showImGuiDemo);

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

void App::RenderCaptureSettings()
{
    ImGui::Separator();
    ImGui::Text("Capture");

    if (ImGui::Button("Refresh monitors"))
    {
        RefreshMonitorList();
        g_appliedMonitorIndex = g_selectedMonitorIndex; // keep applied in sync after refresh
    }

    if (g_monitors.empty())
    {
        ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "No monitors detected.");
        return;
    }

    ImGui::SetNextItemWidth(300.0f);

    // Build array of const char* for ImGui.
    std::vector<const char*> items;
    items.reserve(g_monitorLabelsUtf8.size());
    for (auto& s : g_monitorLabelsUtf8)
        items.push_back(s.c_str());

    ImGui::Checkbox("Exclude app windows from capture", &g_excludeAppWindows);

    ImGui::Combo("Monitor source", &g_selectedMonitorIndex, items.data(), (int)items.size());

    // Safety clamp
    if (g_selectedMonitorIndex < 0) g_selectedMonitorIndex = 0;
    if (g_selectedMonitorIndex >= (int)g_monitors.size()) g_selectedMonitorIndex = (int)g_monitors.size() - 1;

    const bool needRestart =
        (g_selectedMonitorIndex != g_appliedMonitorIndex) ||
        (g_excludeAppWindows != g_appliedExcludeAppWindows);

    if (needRestart)
    {
        RestartCaptureSelected();
    }

    // Optional debug info
    const auto& sel = g_monitors[g_selectedMonitorIndex];
    ImGui::Text("Selected: %S", sel.label.c_str());
}

void App::RenderAsciiSettings()
{
    // Enable
    ImGui::Checkbox("Enable ASCII effect", &m_state.ascii.enabled);

    // Colors
    ImGui::Separator();
    ImGui::Text("ASCII Color (RGB)");

    float rgb[3] = {
        m_state.ascii.tintR / 255.0f,
        m_state.ascii.tintG / 255.0f,
        m_state.ascii.tintB / 255.0f
    };

    if (ImGui::ColorEdit3("Tint", rgb))
    {
        auto clamp01 = [](float v) {
            return (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
            };

        rgb[0] = clamp01(rgb[0]);
        rgb[1] = clamp01(rgb[1]);
        rgb[2] = clamp01(rgb[2]);

        m_state.ascii.tintR = static_cast<uint8_t>(rgb[0] * 255.0f + 0.5f);
        m_state.ascii.tintG = static_cast<uint8_t>(rgb[1] * 255.0f + 0.5f);
        m_state.ascii.tintB = static_cast<uint8_t>(rgb[2] * 255.0f + 0.5f);
    }

    // Edges
    ImGui::Separator();
    ImGui::Text("ASCII Edges");

    const float step = 0.05f;
    const float stepFast = 0.10f;


    ImGui::TextUnformatted("Edge threshold (0.2)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("##EdgeThr", &m_state.ascii.edgeThreshold, step, stepFast, "%.2f");

    ImGui::TextUnformatted("Coherence threshold (0.5)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("##CohThr", &m_state.ascii.coherenceThreshold, step, stepFast, "%.2f");

    // Clamp to [0..1]
    auto clamp01 = [](float& v) {
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        };

    clamp01(m_state.ascii.edgeThreshold);
    clamp01(m_state.ascii.coherenceThreshold);
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
