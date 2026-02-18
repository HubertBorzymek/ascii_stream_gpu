#pragma once

#include "AsciiSettings.h"

struct AppState
{
    // Per-effect settings
    AsciiSettings ascii{};

    // UI-only globals
    bool showImGuiDemo = false;
};
