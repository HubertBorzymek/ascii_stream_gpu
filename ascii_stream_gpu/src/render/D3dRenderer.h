#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

// D3DRenderer
// Owns swapchain + RTV and a simple fullscreen shader pipeline.
// Displays a captured texture (GPU-only) by copying it into an SRV-bindable texture.
class D3DRenderer
{
public:
    D3DRenderer() = default;
    ~D3DRenderer() = default;

    D3DRenderer(const D3DRenderer&) = delete;
    D3DRenderer& operator=(const D3DRenderer&) = delete;

    // Initializes swapchain/RTV for the given window and builds fullscreen pipeline.
    void Initialize(HWND hwnd, ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> ctx);

    // Called when the window client size changes (WM_SIZE or polled in main).
    void OnResize(int newW, int newH);

    // Renders one frame. capturedTex can be null (then it just clears).
    void RenderFrame(ID3D11Texture2D* capturedTex);

private:
    void CreateSwapchainAndRTV(HWND hwnd);
    void CreateRTV();
    void CreateFullscreenPipeline();
    void EnsureStableTextureMatches(ID3D11Texture2D* captured);

private:
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_ctx;

    ComPtr<IDXGISwapChain1> m_swap;
    ComPtr<ID3D11RenderTargetView> m_rtv;

    // Fullscreen quad pipeline
    ComPtr<ID3D11VertexShader> m_vs;
    ComPtr<ID3D11PixelShader> m_ps;
    ComPtr<ID3D11InputLayout> m_il;
    ComPtr<ID3D11Buffer> m_vb;
    ComPtr<ID3D11SamplerState> m_samp;

    // Stable SRV texture
    ComPtr<ID3D11Texture2D> m_stableTex;
    ComPtr<ID3D11ShaderResourceView> m_stableSRV;

    DXGI_FORMAT m_backbufferFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

    int m_winW;
    int m_winH;
};
