#include "SwapChainRenderTarget.h"

#include <stdexcept>

#include "../dx/DxUtils.h"

void SwapChainRenderTarget::Initialize(HWND hwnd, ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> ctx)
{
    if (!hwnd) throw std::runtime_error("SwapChainRenderTarget::Initialize: hwnd is null");
    if (!device) throw std::runtime_error("SwapChainRenderTarget::Initialize: device is null");
    if (!ctx) throw std::runtime_error("SwapChainRenderTarget::Initialize: context is null");

    m_device = device;
    m_ctx = ctx;

    RECT rc{};
    GetClientRect(hwnd, &rc);
    m_winW = (rc.right - rc.left);
    m_winH = (rc.bottom - rc.top);

    CreateSwapchain(hwnd);
    CreateRTV();
}

void SwapChainRenderTarget::CreateSwapchain(HWND hwnd)
{
    // (Re)create swapchain for the given HWND.
    m_swap.Reset();

    ComPtr<IDXGIDevice> dxgiDev;
    ThrowIfFailed(m_device.As(&dxgiDev), "SwapChainRenderTarget: As IDXGIDevice failed");

    ComPtr<IDXGIAdapter> adapter;
    ThrowIfFailed(dxgiDev->GetAdapter(&adapter), "SwapChainRenderTarget: GetAdapter failed");

    ComPtr<IDXGIFactory2> factory;
    ThrowIfFailed(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)factory.GetAddressOf()),
        "SwapChainRenderTarget: GetParent IDXGIFactory2 failed");

    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = m_winW;
    desc.Height = m_winH;
    desc.Format = m_backbufferFormat;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SampleDesc.Count = 1;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

    ThrowIfFailed(factory->CreateSwapChainForHwnd(m_device.Get(), hwnd, &desc, nullptr, nullptr, &m_swap),
        "SwapChainRenderTarget: CreateSwapChainForHwnd failed");
}

void SwapChainRenderTarget::CreateRTV()
{
    // Create RTV for current backbuffer.
    m_rtv.Reset();

    ComPtr<ID3D11Texture2D> backbuf;
    ThrowIfFailed(m_swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)backbuf.GetAddressOf()),
        "SwapChainRenderTarget: GetBuffer(backbuffer) failed");

    ThrowIfFailed(m_device->CreateRenderTargetView(backbuf.Get(), nullptr, &m_rtv),
        "SwapChainRenderTarget: CreateRenderTargetView failed");
}

void SwapChainRenderTarget::OnResize(int newW, int newH)
{
    if (!m_swap || !m_device || !m_ctx)
        return;

    // Minimized (or invalid) - do not resize swapchain to 0x0.
    if (newW <= 0 || newH <= 0)
        return;

    // No change
    if (newW == m_winW && newH == m_winH)
        return;

    m_winW = newW;
    m_winH = newH;

    // Release references to the backbuffer before resizing.
    m_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    m_rtv.Reset();

    ThrowIfFailed(m_swap->ResizeBuffers(
        0,
        static_cast<UINT>(m_winW),
        static_cast<UINT>(m_winH),
        DXGI_FORMAT_UNKNOWN,
        0),
        "SwapChainRenderTarget: ResizeBuffers failed");

    CreateRTV();
}

void SwapChainRenderTarget::Begin(const float clearColor[4])
{
    if (!m_ctx || !m_rtv)
        return;

    m_ctx->OMSetRenderTargets(1, m_rtv.GetAddressOf(), nullptr);
    m_ctx->ClearRenderTargetView(m_rtv.Get(), clearColor);

    D3D11_VIEWPORT vp{};
    vp.Width = (float)m_winW;
    vp.Height = (float)m_winH;
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    m_ctx->RSSetViewports(1, &vp);
}

void SwapChainRenderTarget::Present(int vsync)
{
    if (!m_swap)
        return;

    m_swap->Present(vsync, 0);
}
