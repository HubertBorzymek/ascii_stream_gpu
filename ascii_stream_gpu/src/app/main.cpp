// ascii_stream_gpu_preview.cpp
// Minimal Win32 + D3D11 window that displays Windows Graphics Capture frames (GPU-only path).

#include "Window.h"
#include "WindowRole.h"
#include "Hotkeys.h"
#include "../capture/ScreenCapture.h"
#include "../render/D3dRenderer.h"
#include "../dx/DxContext.h"
#include "../processing/core/FrameProcessor.h"

#include <stdint.h>
#include <Windows.h>
#include <iostream>
#include <string>
#include <cstring>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>


#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

# define WIN_W 1280
# define WIN_H 720

static D3DRenderer* g_renderer = nullptr;

// Trampoline
static void OnAppResize(int w, int h)
{
    if (g_renderer)
        g_renderer->OnResize(w, h);
}

// --------------------------------------------------------
// wWinMain
// Program entry: creates window, D3D, capture session, and runs the render loop.
// --------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    try
    {
        winrt::init_apartment();

        if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported())
        {
            MessageBox(nullptr, L"Windows Graphics Capture is not supported.", L"Error", MB_OK);
            return 0;
        }

        // Create two windows
        HWND hwndMain = CreateAppWindow(hInst, WindowRole::Main, WIN_W, WIN_H, 100, 100);
        HWND hwndPanel = CreateAppWindow(hInst, WindowRole::Panel, 520, WIN_H, 100 + WIN_W + 10, 100); 

		// Register hotkeys (e.g., F1 to toggle panel visibility).
        Hotkeys::Register(hwndPanel);
        
        (void)hwndPanel; // panel unused for now

        // Create D3D device/context (GPU).
        DxContext dx = CreateDxContext();

        // Create render system (swapchain + shader pipeline) - MAIN ONLY for now.
        D3DRenderer renderer;
        renderer.Initialize(hwndMain, dx.device, dx.context);
        g_renderer = &renderer;

        // Resize callback bound to MAIN window
        SetResizeCallback(hwndMain, &OnAppResize);

        // Create screen capure (WGC).
        ScreenCapture capture;
        capture.Start(dx.device);

        // Create frame processor
        FrameProcessor frameProcessor;
        frameProcessor.Initialize(dx.device, dx.context);

		// FPS counter (debug)
        uint64_t frameCount = 0;
        uint64_t lastTick = GetTickCount64();

        // Main message + render loop.
        MSG msg{};
        while (AppIsRunning())
        {
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                if (Hotkeys::HandleMessage(msg))
                    continue;

                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            auto tex = capture.LatestTexture();
            ID3D11Texture2D* processed = frameProcessor.Process(tex.Get());
            renderer.RenderFrame(processed);
            //renderer.RenderFrame(tex.Get());

            // FPS counter
            frameCount++;
            uint64_t now = GetTickCount64();
            if (now - lastTick >= 1000)
            {
                wchar_t title[128];
                swprintf_s(title, L"ASCII Stream GPU  |  FPS: %llu", frameCount);
                SetWindowText(hwndMain, title);
                frameCount = 0;
                lastTick = now;
            }
			// FPS counter end
        }

        frameProcessor.Shutdown();
        capture.Stop();
        Hotkeys::Unregister();
        return 0;
    }
    catch (const std::exception& e)
    {
        Hotkeys::Unregister();

        std::wstring wmsg = L"Fatal error: ";
        wmsg += std::wstring(e.what(), e.what() + strlen(e.what()));
        MessageBox(nullptr, wmsg.c_str(), L"Exception", MB_OK | MB_ICONERROR);
        return 0;
    }
}
