#pragma once
#include <Windows.h>

// Main window procedure (handles resize and close)
LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Creates and shows the application window.
// Returns the HWND handle.
HWND CreateAppWindow(HINSTANCE hInst, int width, int height);

// Getters
bool AppIsRunning();
void GetAppWindowSize(int& outW, int& outH);
