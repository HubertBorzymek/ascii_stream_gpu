#include "MonitorEnumerator.h"

#include <Windows.h>
#include <vector>
#include <string>
#include <cstdio>

namespace
{
    struct EnumContext
    {
        std::vector<MonitorInfo> monitors;
        int index = 0; // 1-based for labels
    };

    static std::wstring BuildLabel(int idx, bool isPrimary, const RECT& rc)
    {
        const int w = rc.right - rc.left;
        const int h = rc.bottom - rc.top;

        wchar_t buf[256]{};

        if (isPrimary)
        {
            std::swprintf(
                buf, _countof(buf),
                L"Monitor %d (Primary) - %dx%d",
                idx, w, h
            );
        }
        else
        {
            std::swprintf(
                buf, _countof(buf),
                L"Monitor %d - %dx%d",
                idx, w, h
            );
        }

        return std::wstring(buf);
    }

    static BOOL CALLBACK EnumMonitorsProc(HMONITOR hMon, HDC, LPRECT, LPARAM lParam)
    {
        auto* ctx = reinterpret_cast<EnumContext*>(lParam);
        if (!ctx || !hMon)
            return TRUE;

        MONITORINFO mi{};
        mi.cbSize = sizeof(mi);
        if (!GetMonitorInfo(hMon, &mi))
            return TRUE;

        ctx->index++;

        MonitorInfo info{};
        info.handle = hMon;
        info.rect = mi.rcMonitor;
        info.isPrimary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
        info.label = BuildLabel(ctx->index, info.isPrimary, info.rect);

        ctx->monitors.push_back(std::move(info));
        return TRUE;
    }
}

namespace MonitorEnumerator
{
    std::vector<MonitorInfo> Enumerate()
    {
        EnumContext ctx{};

        // EnumDisplayMonitors enumerates monitors in a system-defined order.
        // Assign labels in that order (Monitor 1, Monitor 2, ...).
        EnumDisplayMonitors(nullptr, nullptr, EnumMonitorsProc, reinterpret_cast<LPARAM>(&ctx));

        return ctx.monitors;
    }
}
