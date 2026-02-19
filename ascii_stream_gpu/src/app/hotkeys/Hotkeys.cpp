#include "Hotkeys.h"

namespace
{
    static HWND g_panelHwnd = nullptr;

    static const int kHotkeyIdTogglePanel = 1;        // F1
    static const int kHotkeyIdToggleOverlay = 2;      // F2
    static const int kHotkeyIdToggleClickThrough = 3; // F3

    static Hotkeys::ActionCallback g_callback = nullptr;
}

namespace Hotkeys
{
    void SetCallback(ActionCallback cb)
    {
        g_callback = cb;
    }

    bool Register(HWND panelHwnd)
    {
        g_panelHwnd = panelHwnd;

        // NULL hWnd => WM_HOTKEY is posted to the thread message queue.
        bool ok = true;

        ok &= (RegisterHotKey(nullptr, kHotkeyIdTogglePanel, 0, VK_F1) != 0);
        ok &= (RegisterHotKey(nullptr, kHotkeyIdToggleOverlay, 0, VK_F2) != 0);
        ok &= (RegisterHotKey(nullptr, kHotkeyIdToggleClickThrough, 0, VK_F3) != 0);

        return ok;
    }

    void Unregister()
    {
        UnregisterHotKey(nullptr, kHotkeyIdTogglePanel);
        UnregisterHotKey(nullptr, kHotkeyIdToggleOverlay);
        UnregisterHotKey(nullptr, kHotkeyIdToggleClickThrough);

        g_panelHwnd = nullptr;
        g_callback = nullptr;
    }

    bool HandleMessage(const MSG& msg)
    {
        if (msg.message != WM_HOTKEY)
            return false;

        const int id = static_cast<int>(msg.wParam);

        switch (id)
        {
        case kHotkeyIdTogglePanel:
        {
            if (g_panelHwnd && IsWindow(g_panelHwnd))
            {
                const BOOL visible = IsWindowVisible(g_panelHwnd);
                ShowWindow(g_panelHwnd, visible ? SW_HIDE : SW_SHOW);

                if (!visible)
                    SetForegroundWindow(g_panelHwnd);
            }

            if (g_callback)
                g_callback(Hotkeys::Action::TogglePanel);

            return true;
        }

        case kHotkeyIdToggleOverlay:
            if (g_callback)
                g_callback(Hotkeys::Action::ToggleOverlay);
            return true;

        case kHotkeyIdToggleClickThrough:
            if (g_callback)
                g_callback(Hotkeys::Action::ToggleClickThrough);
            return true;

        default:
            return false;
        }
    }

}
