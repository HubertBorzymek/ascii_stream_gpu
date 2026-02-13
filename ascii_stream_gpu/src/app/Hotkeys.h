#pragma once
#include <Windows.h>

namespace Hotkeys
{
    // Registers hotkeys for the application (e.g., F1).
    // panelHwnd is the window that will be toggled by the hotkey.
    bool Register(HWND panelHwnd);

    // Unregisters all hotkeys registered by this module.
    void Unregister();

    // Handles WM_HOTKEY message. Returns true if handled.
    bool HandleMessage(const MSG& msg);
}
