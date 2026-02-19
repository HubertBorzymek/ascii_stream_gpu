#pragma once
#include <Windows.h>

namespace Hotkeys
{
    // High-level actions triggered by hotkeys.
    enum class Action
    {
        TogglePanel,        // F1
        ToggleOverlay,      // F2
        ToggleClickThrough  // F3
    };

    // Callback invoked when a registered hotkey fires.
    using ActionCallback = void(*)(Action action);

    // Registers hotkeys for the application (F1/F2/F3).
    // panelHwnd is the window that will be toggled by F1.
    bool Register(HWND panelHwnd);

    // Unregisters all hotkeys registered by this module.
    void Unregister();

    // Set by App: called when a hotkey fires.
    void SetCallback(ActionCallback cb);

    // Handles WM_HOTKEY message. Returns true if handled.
    bool HandleMessage(const MSG& msg);
}
