#include "ControlPanel.h"
#include "imgui/imgui.h"
#include "monitor/MonitorEnumerator.h"

#include <algorithm>

// ------------------------------------------------------------
// Helpers (local)
// ------------------------------------------------------------
static std::string WideToUtf8(const std::wstring& w)
{
    if (w.empty())
        return {};

    const int needed = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    if (needed <= 0)
        return {};

    std::string out;
    out.resize((size_t)needed);

    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), needed, nullptr, nullptr);
    return out;
}

// ------------------------------------------------------------
// ControlPanel
// ------------------------------------------------------------
void ControlPanel::RefreshMonitors()
{
    m_monitors = MonitorEnumerator::Enumerate();

    // Optional: bring primary to the front (UI convenience).
    std::stable_sort(m_monitors.begin(), m_monitors.end(),
        [](const MonitorInfo& a, const MonitorInfo& b)
        {
            if (a.isPrimary != b.isPrimary)
                return a.isPrimary > b.isPrimary;
            return a.label < b.label;
        });

    m_monitorLabelsUtf8.clear();
    m_monitorLabelsUtf8.reserve(m_monitors.size());
    for (const auto& m : m_monitors)
        m_monitorLabelsUtf8.push_back(WideToUtf8(m.label));

    // Clamp indices.
    if (m_monitors.empty())
    {
        m_selectedMonitorIndex = 0;
        m_appliedMonitorIndex = -1;
        return;
    }

    if (m_selectedMonitorIndex < 0) m_selectedMonitorIndex = 0;
    if (m_selectedMonitorIndex >= (int)m_monitors.size()) m_selectedMonitorIndex = 0;

    if (m_appliedMonitorIndex >= (int)m_monitors.size())
        m_appliedMonitorIndex = -1;
}

void ControlPanel::Render(AppState& state, const Callbacks& cb)
{
    // Lazy init.
    if (m_monitors.empty())
        RefreshMonitors();

    ImGui::Begin("Control Panel");

    RenderCaptureSection(cb);
    RenderOverlaySection(state, cb);
    RenderAsciiSection(state);

    ImGui::End();
}

void ControlPanel::RenderCaptureSection(const Callbacks& cb)
{
    ImGui::Separator();
    ImGui::Text("Capture");

    if (ImGui::Button("Refresh monitors"))
        RefreshMonitors();

    if (m_monitors.empty())
    {
        ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "No monitors detected.");
        return;
    }

    // Exclude app windows
    if (ImGui::Checkbox("Exclude app windows from capture", &m_excludeAppWindows))
    {
        if (cb.onExcludeWindowsChanged)
            cb.onExcludeWindowsChanged(m_excludeAppWindows);
    }

    ImGui::SetNextItemWidth(300.0f);

    // Build array of const char* for ImGui.
    std::vector<const char*> items;
    items.reserve(m_monitorLabelsUtf8.size());
    for (auto& s : m_monitorLabelsUtf8)
        items.push_back(s.c_str());

    ImGui::Combo("Monitor source", &m_selectedMonitorIndex, items.data(), (int)items.size());

    // Apply if changed.
    const bool monitorChanged = (m_selectedMonitorIndex != m_appliedMonitorIndex);
    const bool excludeChanged = (m_excludeAppWindows != m_appliedExcludeAppWindows);

    if (monitorChanged || excludeChanged)
    {
        // Apply exclude first (so App can set window affinity before starting capture)
        if (excludeChanged && cb.onExcludeWindowsChanged)
            cb.onExcludeWindowsChanged(m_excludeAppWindows);

        if (monitorChanged && cb.onMonitorChanged)
            cb.onMonitorChanged(m_monitors[m_selectedMonitorIndex].handle);

        // Update applied state
        if (monitorChanged)
            m_appliedMonitorIndex = m_selectedMonitorIndex;

        if (excludeChanged)
            m_appliedExcludeAppWindows = m_excludeAppWindows;
    }

    // Optional debug info
    const auto& sel = m_monitors[m_selectedMonitorIndex];
    ImGui::Text("Selected: %S", sel.label.c_str());
}

void ControlPanel::RenderOverlaySection(AppState& state, const Callbacks& cb)
{
    ImGui::Separator();
    ImGui::Text("Overlay");

    bool changed = false;

    changed |= ImGui::Checkbox("Enable overlay (borderless fullscreen)", &state.overlayEnabled);

    // Policy: click-through only makes sense in overlay mode
    if (!state.overlayEnabled)
        state.overlayClickThrough = false;

    changed |= ImGui::Checkbox("Click-through (pass input to underlying app)", &state.overlayClickThrough);

    if (changed)
    {
        if (cb.onOverlaySettingsChanged)
            cb.onOverlaySettingsChanged();
    }
}

void ControlPanel::RenderAsciiSection(AppState& state)
{
    ImGui::Separator();
    ImGui::Text("ASCII");

    // Enable
    ImGui::Checkbox("Enable ASCII effect", &state.ascii.enabled);

    // Colors
    ImGui::Separator();
    ImGui::Text("ASCII Color (RGB)");

    float rgb[3] = {
        state.ascii.tintR / 255.0f,
        state.ascii.tintG / 255.0f,
        state.ascii.tintB / 255.0f
    };

    if (ImGui::ColorEdit3("Tint", rgb))
    {
        auto clamp01 = [](float v) {
            return (v < 0.0f) ? 0.0f : (v > 1.0f) ? 1.0f : v;
            };

        rgb[0] = clamp01(rgb[0]);
        rgb[1] = clamp01(rgb[1]);
        rgb[2] = clamp01(rgb[2]);

        state.ascii.tintR = static_cast<uint8_t>(rgb[0] * 255.0f + 0.5f);
        state.ascii.tintG = static_cast<uint8_t>(rgb[1] * 255.0f + 0.5f);
        state.ascii.tintB = static_cast<uint8_t>(rgb[2] * 255.0f + 0.5f);
    }

    // Edges
    ImGui::Separator();
    ImGui::Text("ASCII Edges");

    const float step = 0.05f;
    const float stepFast = 0.10f;

    ImGui::TextUnformatted("Edge threshold (0.2)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("##EdgeThr", &state.ascii.edgeThreshold, step, stepFast, "%.2f");

    ImGui::TextUnformatted("Coherence threshold (0.5)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("##CohThr", &state.ascii.coherenceThreshold, step, stepFast, "%.2f");

    // Clamp to [0..1]
    auto clamp01ref = [](float& v) {
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        };

    clamp01ref(state.ascii.edgeThreshold);
    clamp01ref(state.ascii.coherenceThreshold);
}
