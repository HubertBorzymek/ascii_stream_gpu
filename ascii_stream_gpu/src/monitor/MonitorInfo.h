#pragma once

#include <Windows.h>
#include <string>

// MonitorInfo
// Runtime description of a single physical monitor detected by the system.
// This struct contains only data (no logic).
struct MonitorInfo
{
    // Native Win32 monitor handle (used later for capture).
    HMONITOR handle = nullptr;

    // Full monitor rectangle in virtual screen coordinates.
    RECT rect{};

    // True if this monitor is marked as primary in Windows settings.
    bool isPrimary = false;

    // Preformatted human-readable label for UI display.
    // Example: "Monitor 1 (Primary) - 1920x1080"
    std::wstring label;
};
