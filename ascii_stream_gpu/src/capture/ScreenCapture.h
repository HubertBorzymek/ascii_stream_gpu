#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <wrl.h>

#include <mutex>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

using Microsoft::WRL::ComPtr;

// ScreenCapture
// Captures the primary monitor using Windows Graphics Capture (WGC) and exposes
// the latest frame as an ID3D11Texture2D (GPU-only).
class ScreenCapture
{
public:
    ScreenCapture() = default;
    ~ScreenCapture();

    ScreenCapture(const ScreenCapture&) = delete;
    ScreenCapture& operator=(const ScreenCapture&) = delete;

    // Starts capture of the primary monitor using the given D3D11 device.
    void Start(ComPtr<ID3D11Device> device);

    // Starts capture of a specific monitor using the given D3D11 device.
    void Start(ComPtr<ID3D11Device> device, HMONITOR monitor);

    // Stops capture and releases WinRT resources.
    void Stop();

    // Thread-safe snapshot of the latest captured texture (may be null).
    ComPtr<ID3D11Texture2D> LatestTexture();

    // Size of the capture item (valid after Start).
    int Width() const { return m_width; }
    int Height() const { return m_height; }

    // Sets application windows that should be excluded from capture.
    // Pass nullptr to clear.
    void SetExcludedWindows(HWND mainHwnd, HWND panelHwnd);

private:
    // Internal callback: called by WGC when a new frame arrives.
    void OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& pool);

private:
    std::mutex m_mutex;
    ComPtr<ID3D11Texture2D> m_latestTex;

    int m_width = 0;
    int m_height = 0;

    // WinRT capture objects (kept alive while capturing).
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{ nullptr };

    // Windows to exclude from capture.
    HWND m_excludeMainHwnd = nullptr;
    HWND m_excludePanelHwnd = nullptr;
};
