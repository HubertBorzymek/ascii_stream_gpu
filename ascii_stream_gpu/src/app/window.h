#pragma once
#include <Windows.h>

#include "WindowRole.h"

// Resize callback signature.
// w, h are the new client area size in pixels.
using ResizeCallback = void(*)(int w, int h);

// Main window procedure (handles resize and close)
LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Creates and shows the application window.
// role decides behavior (Main vs Panel).
// x,y are initial top-left position on screen.
HWND CreateAppWindow(HINSTANCE hInst, WindowRole role, int width, int height, int x, int y);

// Setter
void SetResizeCallback(HWND hwnd, ResizeCallback cb);

// Getters
bool AppIsRunning();
void GetAppWindowSize(int& outW, int& outH);
WindowRole GetAppWindowRole(HWND hwnd);


