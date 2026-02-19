#pragma once

#include <Windows.h>
#include <functional>
#include <vector>
#include <string>

#include "monitor/MonitorInfo.h"
#include "appState/AppState.h"

// ControlPanel
// Renders the ImGui control panel (the 2nd window UI).
// Owns UI-only state (monitor list, selected indices, etc.).
// Communicates user intent to the App via callbacks.
class ControlPanel
{
public:
    struct Callbacks
    {
        std::function<void(HMONITOR)> onMonitorChanged;
        std::function<void(bool)> onExcludeWindowsChanged;
        std::function<void()> onOverlaySettingsChanged;
    };

public:
    ControlPanel() = default;
    ~ControlPanel() = default;

    ControlPanel(const ControlPanel&) = delete;
    ControlPanel& operator=(const ControlPanel&) = delete;

    // Render the control panel UI.
    // AppState is modified directly by UI controls (e.g., ASCII settings).
    void Render(AppState& state, const Callbacks& cb);

    // UI-only: refresh monitor enumeration (e.g., on button press).
    void RefreshMonitors();

private:
    void RenderCaptureSection(const Callbacks& cb);
    void RenderOverlaySection(AppState& state, const Callbacks& cb);
    void RenderAsciiSection(AppState& state);

private:
    // UI-only state: monitor list + labels for ImGui combo.
    std::vector<MonitorInfo> m_monitors{};
    std::vector<std::string> m_monitorLabelsUtf8{};

    int m_selectedMonitorIndex = 0;
    int m_appliedMonitorIndex = -1;

    bool m_excludeAppWindows = true;
    bool m_appliedExcludeAppWindows = true;
};
