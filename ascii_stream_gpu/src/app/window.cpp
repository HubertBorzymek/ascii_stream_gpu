#include "window.h"
#include <atomic>

static std::atomic<bool> s_running{ true };
static int s_winW = 1280;
static int s_winH = 720;

// Main window procedure (message handler).
LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_SIZE:
        s_winW = LOWORD(lParam);
        s_winH = HIWORD(lParam);
        return 0;

    case WM_DESTROY:
        s_running = false;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Creates and shows the application window.
HWND CreateAppWindow(HINSTANCE hInst, int width, int height)
{
    WNDCLASS wc{};
    wc.lpfnWndProc = AppWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"AsciiStreamGpuWindow";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClass(&wc);

    RECT rc{ 0, 0, width, height };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowEx(
        0,
        wc.lpszClassName,
        L"ascii_stream_gpu (preview)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr, nullptr, hInst, nullptr
    );

    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

// Getters
bool AppIsRunning()
{
    return s_running.load();
}

void GetAppWindowSize(int& outW, int& outH)
{
    outW = s_winW;
    outH = s_winH;
}
