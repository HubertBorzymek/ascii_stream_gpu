#include "OverlayController.h"

#include <stdexcept>

OverlayController::OverlayController(HWND hwnd)
{
    SetTargetWindow(hwnd);
}

void OverlayController::SetTargetWindow(HWND hwnd)
{
    m_hwnd = hwnd;
}

RECT OverlayController::ResolveTargetRect(HMONITOR targetMonitor) const
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        throw std::runtime_error("OverlayController::ResolveTargetRect: invalid hwnd");

    HMONITOR hmon = targetMonitor;
    if (!hmon)
        hmon = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST);

    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);

    if (!GetMonitorInfo(hmon, &mi))
        throw std::runtime_error("OverlayController::ResolveTargetRect: GetMonitorInfo failed");

    return mi.rcMonitor;
}

void OverlayController::SaveRestoreStateIfNeeded()
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    if (m_applied)
        return;

    GetWindowRect(m_hwnd, &m_restoreRect);
    m_restoreStyle = GetWindowLongPtr(m_hwnd, GWL_STYLE);
    m_restoreExStyle = GetWindowLongPtr(m_hwnd, GWL_EXSTYLE);
}

void OverlayController::RestoreIfApplied()
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    if (!m_applied)
        return;

    SetWindowLongPtr(m_hwnd, GWL_STYLE, m_restoreStyle);
    SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, m_restoreExStyle);

    const int w = m_restoreRect.right - m_restoreRect.left;
    const int h = m_restoreRect.bottom - m_restoreRect.top;

    SetWindowPos(
        m_hwnd,
        HWND_NOTOPMOST,
        m_restoreRect.left,
        m_restoreRect.top,
        w,
        h,
        SWP_FRAMECHANGED | SWP_SHOWWINDOW
    );

    m_applied = false;
}

void OverlayController::ApplyBorderlessStyle()
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    LONG_PTR style = GetWindowLongPtr(m_hwnd, GWL_STYLE);

    // Remove standard window chrome and switch to popup.
    style &= ~(WS_OVERLAPPEDWINDOW);
    style |= WS_POPUP;

    SetWindowLongPtr(m_hwnd, GWL_STYLE, style);
}

void OverlayController::ApplyClickThrough(bool enable)
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    LONG_PTR exStyle = GetWindowLongPtr(m_hwnd, GWL_EXSTYLE);

    if (enable)
    {
        // Layered allows advanced hit-test behaviors in some cases; Transparent enables click-through.
        exStyle |= (WS_EX_LAYERED | WS_EX_TRANSPARENT);
        SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, exStyle);

        // Keep fully opaque; we still render normally, we just want input to pass through.
        SetLayeredWindowAttributes(m_hwnd, 0, 255, LWA_ALPHA);
    }
    else
    {
        // Only remove the click-through flag. Keep layered as-is to avoid side effects.
        exStyle &= ~(WS_EX_TRANSPARENT);
        SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, exStyle);
    }
}

void OverlayController::ApplyTopMost(bool enable)
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    SetWindowPos(
        m_hwnd,
        enable ? HWND_TOPMOST : HWND_NOTOPMOST,
        0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
    );
}

void OverlayController::MoveResizeToRect(const RECT& r)
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    const int w = r.right - r.left;
    const int h = r.bottom - r.top;

    SetWindowPos(
        m_hwnd,
        nullptr, // do not change Z order here (ApplyTopMost handles it)
        r.left,
        r.top,
        w,
        h,
        SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOZORDER
    );
}

void OverlayController::Apply(const Settings& s, HMONITOR targetMonitor)
{
    if (!m_hwnd || !IsWindow(m_hwnd))
        return;

    if (!s.enabled)
    {
        RestoreIfApplied();
        return;
    }

    // Enabled -> compute target rect, save restore snapshot once, apply overlay state.
    RECT targetRect{};
    try
    {
        targetRect = ResolveTargetRect(targetMonitor);
    }
    catch (...)
    {
        // If monitor resolution fails, do not partially apply.
        return;
    }

    SaveRestoreStateIfNeeded();

    ApplyBorderlessStyle();

    // Click-through makes sense only in overlay mode.
    ApplyClickThrough(s.clickThrough);

    ApplyTopMost(s.topMost);

    MoveResizeToRect(targetRect);

    m_applied = true;
}
