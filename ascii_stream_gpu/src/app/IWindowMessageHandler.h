#pragma once

#include <Windows.h>

// Interface used by Window.cpp to route Win32 messages to an owning object (e.g., App).
// Returning true means: "message handled, use outResult as WndProc return value".
// Returning false means: "not handled, Window.cpp will apply its default behavior".
class IWindowMessageHandler
{
public:
    virtual ~IWindowMessageHandler() = default;

    virtual bool HandleMessage(HWND hwnd,
        UINT msg,
        WPARAM wParam,
        LPARAM lParam,
        LRESULT& outResult) = 0;
};
