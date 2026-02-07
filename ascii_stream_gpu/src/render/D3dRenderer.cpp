#include "D3dRenderer.h"

#include <stdexcept>
#include <cstring>

static void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr))
        throw std::runtime_error(msg);
}

void D3DRenderer::Initialize(HWND hwnd, ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> ctx)
{
    if (!hwnd) throw std::runtime_error("D3DRenderer::Initialize: hwnd is null");
    if (!device) throw std::runtime_error("D3DRenderer::Initialize: device is null");
    if (!ctx) throw std::runtime_error("D3DRenderer::Initialize: context is null");

    m_device = device;
    m_ctx = ctx;

    // Initialize window size from the actual client rect (no globals).
    RECT rc{};
    GetClientRect(hwnd, &rc);
    m_winW = (rc.right - rc.left);
    m_winH = (rc.bottom - rc.top);

    CreateSwapchainAndRTV(hwnd);
    CreateFullscreenPipeline();
}

void D3DRenderer::CreateSwapchainAndRTV(HWND hwnd)
{
    ComPtr<IDXGIDevice> dxgiDev;
    ThrowIfFailed(m_device.As(&dxgiDev), "As IDXGIDevice failed");

    ComPtr<IDXGIAdapter> adapter;
    ThrowIfFailed(dxgiDev->GetAdapter(&adapter), "GetAdapter failed");

    ComPtr<IDXGIFactory2> factory;
    ThrowIfFailed(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)factory.GetAddressOf()),
        "GetParent IDXGIFactory2 failed");

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
        "CreateSwapChainForHwnd failed");

    CreateRTV();
}

void D3DRenderer::CreateRTV()
{
    m_rtv.Reset();

    ComPtr<ID3D11Texture2D> backbuf;
    ThrowIfFailed(m_swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)backbuf.GetAddressOf()),
        "GetBuffer backbuffer failed");

    ThrowIfFailed(m_device->CreateRenderTargetView(backbuf.Get(), nullptr, &m_rtv),
        "CreateRenderTargetView failed");
}

void D3DRenderer::OnResize(int newW, int newH)
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

    // Resize swapchain buffers to match the new window client size.
    // Use DXGI_FORMAT_UNKNOWN to keep the existing format.
    ThrowIfFailed(m_swap->ResizeBuffers(
        0,
        static_cast<UINT>(m_winW),
        static_cast<UINT>(m_winH),
        DXGI_FORMAT_UNKNOWN,
        0),
        "ResizeBuffers failed");

    CreateRTV();
}


void D3DRenderer::CreateFullscreenPipeline()
{
    struct Vtx { float x, y, u, v; };
    Vtx verts[4] = {
        { -1.f, -1.f, 0.f, 1.f },
        { -1.f,  1.f, 0.f, 0.f },
        {  1.f, -1.f, 1.f, 1.f },
        {  1.f,  1.f, 1.f, 0.f },
    };

    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof(verts);
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = verts;

    ThrowIfFailed(m_device->CreateBuffer(&bd, &init, &m_vb), "CreateBuffer vb failed");

    const char* vsSrc = R"(
        struct VSIn { float2 pos : POSITION; float2 uv : TEXCOORD0; };
        struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
        VSOut main(VSIn i) {
            VSOut o;
            o.pos = float4(i.pos, 0, 1);
            o.uv = i.uv;
            return o;
        }
    )";

    const char* psSrc = R"(
        Texture2D tex0 : register(t0);
        SamplerState s0 : register(s0);
        float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
            return tex0.Sample(s0, uv);
        }
    )";

    ComPtr<ID3DBlob> vsBlob, psBlob, err;

    HRESULT hr = D3DCompile(vsSrc, std::strlen(vsSrc), nullptr, nullptr, nullptr,
        "main", "vs_5_0", 0, 0, &vsBlob, &err);
    if (err) { err.Reset(); }
    ThrowIfFailed(hr, "D3DCompile VS failed");

    hr = D3DCompile(psSrc, std::strlen(psSrc), nullptr, nullptr, nullptr,
        "main", "ps_5_0", 0, 0, &psBlob, &err);
    if (err) { err.Reset(); }
    ThrowIfFailed(hr, "D3DCompile PS failed");

    ThrowIfFailed(m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vs),
        "CreateVertexShader failed");
    ThrowIfFailed(m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_ps),
        "CreatePixelShader failed");

    D3D11_INPUT_ELEMENT_DESC ild[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    ThrowIfFailed(m_device->CreateInputLayout(ild, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_il),
        "CreateInputLayout failed");

    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ThrowIfFailed(m_device->CreateSamplerState(&sd, &m_samp), "CreateSamplerState failed");
}

void D3DRenderer::EnsureStableTextureMatches(ID3D11Texture2D* captured)
{
    if (!captured) return;

    D3D11_TEXTURE2D_DESC cdesc{};
    captured->GetDesc(&cdesc);

    bool need = false;
    if (!m_stableTex) need = true;
    else
    {
        D3D11_TEXTURE2D_DESC sdesc{};
        m_stableTex->GetDesc(&sdesc);
        if (sdesc.Width != cdesc.Width || sdesc.Height != cdesc.Height || sdesc.Format != cdesc.Format)
            need = true;
    }

    if (need)
    {
        m_stableSRV.Reset();
        m_stableTex.Reset();

        D3D11_TEXTURE2D_DESC td = cdesc;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE; // sampleable by pixel shader
        td.Usage = D3D11_USAGE_DEFAULT;
        td.CPUAccessFlags = 0;
        td.MiscFlags = 0;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc.Count = 1;

        ThrowIfFailed(m_device->CreateTexture2D(&td, nullptr, &m_stableTex), "CreateTexture2D stableTex failed");
        ThrowIfFailed(m_device->CreateShaderResourceView(m_stableTex.Get(), nullptr, &m_stableSRV),
            "CreateShaderResourceView stableSRV failed");
    }
}

void D3DRenderer::RenderFrame(ID3D11Texture2D* capturedTex)
{
    // Copy captured frame into stable SRV texture (GPU->GPU).
    if (capturedTex)
    {
        EnsureStableTextureMatches(capturedTex);
        m_ctx->CopyResource(m_stableTex.Get(), capturedTex);
    }

    float clear[4] = { 0.f, 0.f, 0.f, 1.f };
    m_ctx->OMSetRenderTargets(1, m_rtv.GetAddressOf(), nullptr);
    m_ctx->ClearRenderTargetView(m_rtv.Get(), clear);

    D3D11_VIEWPORT vp{};
    vp.Width = (float)m_winW;
    vp.Height = (float)m_winH;
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    m_ctx->RSSetViewports(1, &vp);

    UINT stride = 16;
    UINT offset = 0;
    m_ctx->IASetInputLayout(m_il.Get());
    m_ctx->IASetVertexBuffers(0, 1, m_vb.GetAddressOf(), &stride, &offset);
    m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    m_ctx->VSSetShader(m_vs.Get(), nullptr, 0);
    m_ctx->PSSetShader(m_ps.Get(), nullptr, 0);
    m_ctx->PSSetSamplers(0, 1, m_samp.GetAddressOf());

    if (m_stableSRV)
        m_ctx->PSSetShaderResources(0, 1, m_stableSRV.GetAddressOf());

    m_ctx->Draw(4, 0);

    ID3D11ShaderResourceView* nullSRV[1] = { nullptr };
    m_ctx->PSSetShaderResources(0, 1, nullSRV);

    m_swap->Present(1, 0);
}
