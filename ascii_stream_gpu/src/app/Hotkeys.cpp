#include "Hotkeys.h"

namespace
{
    static HWND g_panelHwnd = nullptr;
    static const int kHotkeyIdTogglePanel = 1;
}

namespace Hotkeys
{
    bool Register(HWND panelHwnd)
    {
        g_panelHwnd = panelHwnd;

        // Register F1 with no modifiers.
        // NULL hWnd => WM_HOTKEY is posted to the thread message queue.
        if (!RegisterHotKey(nullptr, kHotkeyIdTogglePanel, 0, VK_F1))
            return false;

        return true;
    }

    void Unregister()
    {
        UnregisterHotKey(nullptr, kHotkeyIdTogglePanel);
        g_panelHwnd = nullptr;
    }

    bool HandleMessage(const MSG& msg)
    {
        if (msg.message != WM_HOTKEY)
            return false;

        if (static_cast<int>(msg.wParam) != kHotkeyIdTogglePanel)
            return false;

        if (!g_panelHwnd || !IsWindow(g_panelHwnd))
            return true; // handled, but nothing to toggle

        const BOOL visible = IsWindowVisible(g_panelHwnd);
        ShowWindow(g_panelHwnd, visible ? SW_HIDE : SW_SHOW);

        // Optional: bring to front when showing
        if (!visible)
            SetForegroundWindow(g_panelHwnd);

        return true;
    }
}
