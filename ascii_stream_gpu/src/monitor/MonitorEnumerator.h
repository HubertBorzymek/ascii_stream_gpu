#pragma once

#include <vector>

#include "MonitorInfo.h"

// MonitorEnumerator
// Provides runtime enumeration of physical monitors available in the system.
namespace MonitorEnumerator
{
    // Enumerate all monitors currently connected to the system.
    // The returned vector order is stable within a single enumeration call,
    // but can change between runs depending on OS/driver configuration.
    std::vector<MonitorInfo> Enumerate();
}
