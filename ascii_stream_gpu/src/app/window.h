#pragma once
#include <Windows.h>

// Resize callback signature.
// w, h are the new client area size in pixels.
using ResizeCallback = void(*)(int w, int h);

// Main window procedure (handles resize and close)
LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Creates and shows the application window.
// Returns the HWND handle.
HWND CreateAppWindow(HINSTANCE hInst, int width, int height);

// Setter
void SetResizeCallback(ResizeCallback cb);

// Getters
bool AppIsRunning();
void GetAppWindowSize(int& outW, int& outH);

