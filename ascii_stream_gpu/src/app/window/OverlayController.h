#pragma once

#include <Windows.h>

// OverlayController
// Owns the Win32 window style/position logic for "overlay mode" (borderless fullscreen + optional click-through).
// Does NOT render UI. Does NOT know about ImGui. Does NOT own monitor enumeration.
// App calls Apply() when overlay settings or target monitor changes.
class OverlayController
{
public:
    struct Settings
    {
        bool enabled = false;       // borderless fullscreen overlay
        bool clickThrough = false;  // WS_EX_TRANSPARENT (usually with WS_EX_LAYERED)
        bool topMost = true;        // HWND_TOPMOST vs HWND_NOTOPMOST
    };

public:
    OverlayController() = default;
    explicit OverlayController(HWND hwnd);

    void SetTargetWindow(HWND hwnd);

    // Applies overlay settings to the target window.
    // targetMonitor may be nullptr -> controller will pick monitor nearest to window.
    void Apply(const Settings& s, HMONITOR targetMonitor);

    bool IsApplied() const { return m_applied; }

private:
    RECT ResolveTargetRect(HMONITOR targetMonitor) const;

    void SaveRestoreStateIfNeeded();
    void RestoreIfApplied();

    void ApplyBorderlessStyle();
    void ApplyClickThrough(bool enable);
    void ApplyTopMost(bool enable);

    void MoveResizeToRect(const RECT& r);

private:
    HWND m_hwnd = nullptr;

    bool m_applied = false;

    // Stored only for restoring back to normal windowed mode.
    RECT m_restoreRect{};
    LONG_PTR m_restoreStyle = 0;
    LONG_PTR m_restoreExStyle = 0;
};
