#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// SwapChainRenderTarget
// Owns: swapchain + RTV for a single HWND.
// Responsibilities:
// - CreateSwapChainForHwnd
// - CreateRTV from backbuffer
// - ResizeBuffers + RTV recreation
// - Begin(): OMSetRenderTargets + viewport + clear
// - Present()
class SwapChainRenderTarget
{
public:
    SwapChainRenderTarget() = default;
    ~SwapChainRenderTarget() = default;

    SwapChainRenderTarget(const SwapChainRenderTarget&) = delete;
    SwapChainRenderTarget& operator=(const SwapChainRenderTarget&) = delete;

    void Initialize(HWND hwnd, ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> ctx);

    // Called when the window client size changes (WM_SIZE).
    void OnResize(int newW, int newH);

    // Bind RTV + viewport and clear.
    void Begin(const float clearColor[4]);

    // Present swapchain.
    void Present(int vsync);

    int Width() const { return m_winW; }
    int Height() const { return m_winH; }

private:
    void CreateSwapchain(HWND hwnd);
    void CreateRTV();

private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_ctx;

    ComPtr<IDXGISwapChain1> m_swap;
    ComPtr<ID3D11RenderTargetView> m_rtv;

    DXGI_FORMAT m_backbufferFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

    int m_winW = 0;
    int m_winH = 0;
};
