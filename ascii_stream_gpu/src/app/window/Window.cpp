#include "window.h"
#include "IWindowMessageHandler.h"

#include <atomic>

static std::atomic<bool> s_running{ true };

static ResizeCallback s_onResize = nullptr;

struct WindowState
{
    WindowRole role = WindowRole::Main;
    ResizeCallback onResize = nullptr;
    IWindowMessageHandler* handler = nullptr;

    int winW = 0;
    int winH = 0;
};

static WindowState* GetState(HWND hwnd)
{
    return reinterpret_cast<WindowState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
}

// Main window procedure (message handler).
LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_NCCREATE)
    {
        CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        WindowState* st = reinterpret_cast<WindowState*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(st));
        return TRUE;
    }

    // From now on, we can route messages to a handler (if set).
    if (WindowState* st = GetState(hwnd))
    {
        if (st->handler)
        {
            LRESULT routedResult = 0;
            if (st->handler->HandleMessage(hwnd, msg, wParam, lParam, routedResult))
                return routedResult;
        }
    }

    switch (msg)
    {
    case WM_SIZE:
    {
        WindowState* st = GetState(hwnd);
        if (st)
        {
            st->winW = LOWORD(lParam);
            st->winH = HIWORD(lParam);

            if (st->onResize)
                st->onResize(st->winW, st->winH);
        }
        return 0;
    }

    case WM_CLOSE:
    {
        WindowState* st = GetState(hwnd);
        if (st && st->role == WindowRole::Panel)
        {
            // Panel close hides window instead of destroying the app.
            ShowWindow(hwnd, SW_HIDE);
            return 0;
        }

        DestroyWindow(hwnd);
        return 0;
    }

    case WM_DESTROY:
    {
        WindowState* st = GetState(hwnd);

        // IMPORTANT: decide app lifetime BEFORE deleting st
        if (st && st->role == WindowRole::Main)
        {
            s_running = false;
            PostQuitMessage(0);
        }

        // Free per-window state
        if (st)
        {
            SetWindowLongPtr(hwnd, GWLP_USERDATA, 0);
            delete st;
        }

        return 0;
    }
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Creates and shows the application window.
HWND CreateAppWindow(HINSTANCE hInst, WindowRole role, int width, int height, int x, int y)
{
    // Register class once.
    static bool registered = false;
    if (!registered)
    {
        WNDCLASS wc{};
        wc.lpfnWndProc = AppWndProc;
        wc.hInstance = hInst;
        wc.lpszClassName = L"AsciiStreamGpuWindow";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClass(&wc);
        registered = true;
    }

    RECT rc{ 0, 0, width, height };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    // Allocate state per window.
    WindowState* st = new WindowState();
    st->role = role;
    st->winW = width;
    st->winH = height;

    const wchar_t* title = (role == WindowRole::Main)
        ? L"ascii_stream_gpu (main)"
        : L"ascii_stream_gpu (panel)";

    HWND hwnd = CreateWindowEx(
        0,
        L"AsciiStreamGpuWindow",
        title,
        WS_OVERLAPPEDWINDOW,
        x, y,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr, nullptr, hInst,
        st // passed into WM_NCCREATE
    );

    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

// Set resize callback for a specific window.
void SetResizeCallback(HWND hwnd, ResizeCallback cb)
{
    WindowState* st = GetState(hwnd);
    if (st)
        st->onResize = cb;
}

void SetWindowMessageHandler(HWND hwnd, IWindowMessageHandler* handler)
{
    WindowState* st = GetState(hwnd);
    if (st)
        st->handler = handler;
}

// Getters
bool AppIsRunning()
{
    return s_running.load();
}

void GetAppWindowSize(HWND hwnd, int& outW, int& outH)
{
    WindowState* st = GetState(hwnd);
    if (st)
    {
        outW = st->winW;
        outH = st->winH;
        return;
    }

    // Fallback: query client rect
    RECT rc{};
    GetClientRect(hwnd, &rc);
    outW = rc.right - rc.left;
    outH = rc.bottom - rc.top;
}

WindowRole GetAppWindowRole(HWND hwnd)
{
    WindowState* st = GetState(hwnd);
    return st ? st->role : WindowRole::Main;
}
