#pragma once

#include "AsciiSettings.h"

struct AppState
{
    // Per-effect settings
    AsciiSettings ascii{};

    // UI-only globals
    bool showImGuiDemo = false;

    // Overlay / display
    bool overlayEnabled = false;        // borderless fullscreen overlay on selected monitor
    bool overlayClickThrough = false;   // pass mouse/keyboard to underlying app (WS_EX_TRANSPARENT)

};
