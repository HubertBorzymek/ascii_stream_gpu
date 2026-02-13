#include "D3dRenderer.h"

#include <stdexcept>
#include <cstring>

#include "../dx/DxUtils.h"

void D3DRenderer::Initialize(HWND hwnd, ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> ctx)
{
    if (!hwnd) throw std::runtime_error("D3DRenderer::Initialize: hwnd is null");
    if (!device) throw std::runtime_error("D3DRenderer::Initialize: device is null");
    if (!ctx) throw std::runtime_error("D3DRenderer::Initialize: context is null");

    m_device = device;
    m_ctx = ctx;

    // Initialize swapchain + RTV through encapsulated render target.
    m_rt.Initialize(hwnd, device, ctx);

    // Build fullscreen pipeline (quad + shaders).
    CreateFullscreenPipeline();
}

void D3DRenderer::OnResize(int newW, int newH)
{
    m_rt.OnResize(newW, newH);
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
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.CPUAccessFlags = 0;
        td.MiscFlags = 0;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.SampleDesc.Count = 1;

        ThrowIfFailed(m_device->CreateTexture2D(&td, nullptr, &m_stableTex),
            "CreateTexture2D stableTex failed");

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
    m_rt.Begin(clear);

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

    m_rt.Present(1);
}
