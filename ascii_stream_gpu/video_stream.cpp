// ascii_stream_gpu_preview.cpp
// Minimal Win32 + D3D11 window that displays Windows Graphics Capture frames (GPU-only path).

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl.h>

#include <atomic>
#include <mutex>
#include <iostream>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

using WinRTDirect3DDevice = winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
using WinRTInspectable = winrt::Windows::Foundation::IInspectable;

// ---------------------------
// Global state (simple, minimal)
// ---------------------------
static std::atomic<bool> g_running{ true };

static std::mutex g_texMutex;
static ComPtr<ID3D11Texture2D> g_latestCapturedTex; // "latest frame" as a D3D11 texture (no CPU)

static int g_winW = 1280;
static int g_winH = 720;

// ---------------------------
// Utility: HRESULT check (throws)
// ---------------------------
static void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr))
    {
        std::cerr << msg << " hr=0x" << std::hex << hr << std::dec << "\n";
        throw std::runtime_error(msg);
    }
}

// --------------------------------------------------------
// createD3DDevice
// Creates a D3D11 device + immediate context (hardware GPU).
// BGRA support is required for interoperability with capture/render formats.
// --------------------------------------------------------
static void createD3DDevice(ComPtr<ID3D11Device>& outDevice, ComPtr<ID3D11DeviceContext>& outCtx)
{
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };

    D3D_FEATURE_LEVEL chosen = D3D_FEATURE_LEVEL_11_0;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        featureLevels,
        _countof(featureLevels),
        D3D11_SDK_VERSION,
        outDevice.GetAddressOf(),
        &chosen,
        outCtx.GetAddressOf()
    );

    ThrowIfFailed(hr, "D3D11CreateDevice failed");
    std::cout << "D3D11 device created. Feature level = 0x" << std::hex << chosen << std::dec << "\n";
}

// --------------------------------------------------------
// createWinRTDirect3DDevice
// Wraps an ID3D11Device as WinRT IDirect3DDevice, needed by FramePool.
// --------------------------------------------------------
static WinRTDirect3DDevice createWinRTDirect3DDevice(ComPtr<ID3D11Device> const& d3dDevice)
{
    ComPtr<IDXGIDevice> dxgiDevice;
    ThrowIfFailed(d3dDevice.As(&dxgiDevice), "As IDXGIDevice failed");

    winrt::com_ptr<::IInspectable> inspectable;
    HRESULT hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectable.put());
    ThrowIfFailed(hr, "CreateDirect3D11DeviceFromDXGIDevice failed");

    // Convert raw COM IInspectable -> WinRT object
    WinRTDirect3DDevice winrtDevice = inspectable.as<WinRTDirect3DDevice>();
    return winrtDevice;
}


// --------------------------------------------------------
// createCaptureItemForPrimaryMonitor
// Creates a GraphicsCaptureItem bound to the primary monitor (HMONITOR).
// --------------------------------------------------------
static GraphicsCaptureItem createCaptureItemForPrimaryMonitor()
{
    POINT pt{ 0, 0 };
    HMONITOR hmon = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);
    if (!hmon)
        throw std::runtime_error("MonitorFromPoint failed");

    winrt::com_ptr<IGraphicsCaptureItemInterop> interop =
        get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();

    GraphicsCaptureItem item{ nullptr };
    HRESULT hr = interop->CreateForMonitor(
        hmon,
        guid_of<GraphicsCaptureItem>(),
        put_abi(item)
    );
    ThrowIfFailed(hr, "CreateForMonitor failed");

    return item;
}

// --------------------------------------------------------
// AsDxgiAccess
// Converts WinRT IDirect3DSurface to a COM interop interface that can expose
// the underlying DXGI / D3D11 resource (ID3D11Texture2D).
// --------------------------------------------------------
static ComPtr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>
AsDxgiAccess(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const& surface)
{
    IInspectable* insp = reinterpret_cast<IInspectable*>(winrt::get_abi(surface));
    ComPtr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access;

    HRESULT hr = insp->QueryInterface(
        __uuidof(::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess),
        reinterpret_cast<void**>(access.GetAddressOf())
    );

    if (FAILED(hr))
        access.Reset();

    return access;
}

// --------------------------------------------------------
// WndProc
// Handles basic window events. Updates size and exits cleanly.
// --------------------------------------------------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_SIZE:
        g_winW = LOWORD(lParam);
        g_winH = HIWORD(lParam);
        return 0;

    case WM_DESTROY:
        g_running = false;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// --------------------------------------------------------
// CreateAppWindow
// Creates and shows a Win32 window for rendering.
// --------------------------------------------------------
static HWND CreateAppWindow(HINSTANCE hInst)
{
    WNDCLASS wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"AsciiStreamGpuWindow";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    RegisterClass(&wc);

    RECT rc{ 0, 0, g_winW, g_winH };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowEx(
        0,
        wc.lpszClassName,
        L"ascii_stream_gpu (preview)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr, nullptr, hInst, nullptr
    );

    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

// --------------------------------------------------------
// D3DRender
// Owns swapchain + RTV and a simple fullscreen shader pipeline.
// Keeps a "stable" GPU texture with SRV and copies captured frames into it.
// --------------------------------------------------------
struct D3DRender
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> ctx;

    ComPtr<IDXGISwapChain1> swap;
    ComPtr<ID3D11RenderTargetView> rtv;

    // Fullscreen quad pipeline
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11InputLayout> il;
    ComPtr<ID3D11Buffer> vb;
    ComPtr<ID3D11SamplerState> samp;

    // Owned texture that is guaranteed to be bindable as SRV (sampling in PS).
    ComPtr<ID3D11Texture2D> stableTex;
    ComPtr<ID3D11ShaderResourceView> stableSRV;

    // Chosen to match WGC typical format (BGRA8 UNORM).
    DXGI_FORMAT backbufferFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

    // Creates swapchain for the window and builds the RTV from backbuffer.
    void CreateSwapchainAndRTV(HWND hwnd)
    {
        ComPtr<IDXGIDevice> dxgiDev;
        ThrowIfFailed(device.As(&dxgiDev), "As IDXGIDevice failed");

        ComPtr<IDXGIAdapter> adapter;
        ThrowIfFailed(dxgiDev->GetAdapter(&adapter), "GetAdapter failed");

        ComPtr<IDXGIFactory2> factory;
        ThrowIfFailed(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)factory.GetAddressOf()),
            "GetParent IDXGIFactory2 failed");

        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = g_winW;
        desc.Height = g_winH;
        desc.Format = backbufferFormat;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 2;
        desc.SampleDesc.Count = 1;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

        ThrowIfFailed(factory->CreateSwapChainForHwnd(device.Get(), hwnd, &desc, nullptr, nullptr, &swap),
            "CreateSwapChainForHwnd failed");

        CreateRTV();
    }

    // Creates RTV from swapchain's backbuffer.
    void CreateRTV()
    {
        rtv.Reset();

        ComPtr<ID3D11Texture2D> backbuf;
        ThrowIfFailed(swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)backbuf.GetAddressOf()),
            "GetBuffer backbuffer failed");

        ThrowIfFailed(device->CreateRenderTargetView(backbuf.Get(), nullptr, &rtv),
            "CreateRenderTargetView failed");
    }

    // Resizes swapchain buffers when the window size changes.
    void ResizeIfNeeded()
    {
        if (!swap) return;
        if (g_winW <= 0 || g_winH <= 0) return; // minimized

        DXGI_SWAP_CHAIN_DESC1 scDesc{};
        ThrowIfFailed(swap->GetDesc1(&scDesc), "GetDesc1 failed");

        if ((int)scDesc.Width != g_winW || (int)scDesc.Height != g_winH)
        {
            ctx->OMSetRenderTargets(0, nullptr, nullptr);
            rtv.Reset();

            ThrowIfFailed(swap->ResizeBuffers(0, g_winW, g_winH, scDesc.Format, scDesc.Flags),
                "ResizeBuffers failed");

            CreateRTV();
        }
    }

    // Builds a minimal fullscreen quad (triangle strip) + trivial texture sampling shaders.
    void CreateFullscreenPipeline()
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

        ThrowIfFailed(device->CreateBuffer(&bd, &init, &vb), "CreateBuffer vb failed");

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

        HRESULT hr = D3DCompile(vsSrc, strlen(vsSrc), nullptr, nullptr, nullptr,
            "main", "vs_5_0", 0, 0, &vsBlob, &err);
        if (err) { std::cout << (char*)err->GetBufferPointer() << "\n"; err.Reset(); }
        ThrowIfFailed(hr, "D3DCompile VS failed");

        hr = D3DCompile(psSrc, strlen(psSrc), nullptr, nullptr, nullptr,
            "main", "ps_5_0", 0, 0, &psBlob, &err);
        if (err) { std::cout << (char*)err->GetBufferPointer() << "\n"; err.Reset(); }
        ThrowIfFailed(hr, "D3DCompile PS failed");

        ThrowIfFailed(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs),
            "CreateVertexShader failed");
        ThrowIfFailed(device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps),
            "CreatePixelShader failed");

        D3D11_INPUT_ELEMENT_DESC ild[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        ThrowIfFailed(device->CreateInputLayout(ild, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &il),
            "CreateInputLayout failed");

        D3D11_SAMPLER_DESC sd{};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        ThrowIfFailed(device->CreateSamplerState(&sd, &samp), "CreateSamplerState failed");
    }

    // Ensures that stableTex exists and matches captured texture size/format, and is SRV-bindable.
    void EnsureStableTextureMatches(ID3D11Texture2D* captured)
    {
        if (!captured) return;

        D3D11_TEXTURE2D_DESC cdesc{};
        captured->GetDesc(&cdesc);

        bool need = false;
        if (!stableTex) need = true;
        else
        {
            D3D11_TEXTURE2D_DESC sdesc{};
            stableTex->GetDesc(&sdesc);
            if (sdesc.Width != cdesc.Width || sdesc.Height != cdesc.Height || sdesc.Format != cdesc.Format)
                need = true;
        }

        if (need)
        {
            stableSRV.Reset();
            stableTex.Reset();

            D3D11_TEXTURE2D_DESC td = cdesc;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE; // must be sampleable by PS
            td.Usage = D3D11_USAGE_DEFAULT;
            td.CPUAccessFlags = 0;
            td.MiscFlags = 0;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.SampleDesc.Count = 1;

            ThrowIfFailed(device->CreateTexture2D(&td, nullptr, &stableTex), "CreateTexture2D stableTex failed");
            ThrowIfFailed(device->CreateShaderResourceView(stableTex.Get(), nullptr, &stableSRV),
                "CreateShaderResourceView stableSRV failed");
        }
    }

    // Renders one frame: copy latest captured texture (GPU->GPU) and draw fullscreen quad to swapchain.
    void RenderFrame()
    {
        ResizeIfNeeded();

        // Get latest captured texture snapshot (thread-safe).
        ComPtr<ID3D11Texture2D> cap;
        {
            std::lock_guard<std::mutex> lock(g_texMutex);
            cap = g_latestCapturedTex;
        }

        // If we have a captured frame, copy it to our stable SRV texture (still GPU-only).
        if (cap)
        {
            EnsureStableTextureMatches(cap.Get());
            ctx->CopyResource(stableTex.Get(), cap.Get());
        }

        // Clear and draw.
        float clear[4] = { 0.f, 0.f, 0.f, 1.f };
        ctx->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
        ctx->ClearRenderTargetView(rtv.Get(), clear);

        D3D11_VIEWPORT vp{};
        vp.Width = (float)g_winW;
        vp.Height = (float)g_winH;
        vp.MinDepth = 0.f;
        vp.MaxDepth = 1.f;
        ctx->RSSetViewports(1, &vp);

        UINT stride = 16;
        UINT offset = 0;
        ctx->IASetInputLayout(il.Get());
        ctx->IASetVertexBuffers(0, 1, vb.GetAddressOf(), &stride, &offset);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(ps.Get(), nullptr, 0);
        ctx->PSSetSamplers(0, 1, samp.GetAddressOf());

        if (stableSRV)
            ctx->PSSetShaderResources(0, 1, stableSRV.GetAddressOf());

        ctx->Draw(4, 0);

        // Unbind SRV to avoid resource hazards later (good practice).
        ID3D11ShaderResourceView* nullSRV[1] = { nullptr };
        ctx->PSSetShaderResources(0, 1, nullSRV);

        swap->Present(1, 0);
    }
};

// --------------------------------------------------------
// wWinMain
// Program entry: creates window, D3D, capture session, and runs the render loop.
// --------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    try
    {
        init_apartment();

        if (!GraphicsCaptureSession::IsSupported())
        {
            MessageBox(nullptr, L"Windows Graphics Capture is not supported.", L"Error", MB_OK);
            return 0;
        }

        HWND hwnd = CreateAppWindow(hInst);

        // Create D3D device/context (GPU).
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> ctx;
        createD3DDevice(device, ctx);

        // Create render system (swapchain + shader pipeline).
        D3DRender renderer{};
        renderer.device = device;
        renderer.ctx = ctx;
        renderer.CreateSwapchainAndRTV(hwnd);
        renderer.CreateFullscreenPipeline();

        // Setup capture (WGC).
        WinRTDirect3DDevice winrtDevice = createWinRTDirect3DDevice(device);

        GraphicsCaptureItem item = createCaptureItemForPrimaryMonitor();
        auto capSize = item.Size();
        std::cout << "Capture item size: " << capSize.Width << " x " << capSize.Height << "\n";

        auto framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            winrtDevice,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            capSize
        );

        auto session = framePool.CreateCaptureSession(item);

        // When a frame arrives, extract ID3D11Texture2D and store it as "latest".
        framePool.FrameArrived([&](Direct3D11CaptureFramePool const& pool, WinRTInspectable const&)
            {
                auto frame = pool.TryGetNextFrame();
                if (!frame) return;

                auto surface = frame.Surface();
                auto access = AsDxgiAccess(surface);
                if (!access) return;

                ComPtr<ID3D11Texture2D> tex;
                HRESULT hr = access->GetInterface(__uuidof(ID3D11Texture2D), (void**)tex.GetAddressOf());
                if (FAILED(hr) || !tex) return;

                std::lock_guard<std::mutex> lock(g_texMutex);
                g_latestCapturedTex = tex;
            });

        session.StartCapture();

        // Main message + render loop.
        MSG msg{};
        while (g_running)
        {
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            renderer.RenderFrame();
        }

        session.Close();
        framePool.Close();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::wstring wmsg = L"Fatal error: ";
        wmsg += std::wstring(e.what(), e.what() + strlen(e.what()));
        MessageBox(nullptr, wmsg.c_str(), L"Exception", MB_OK | MB_ICONERROR);
        return 0;
    }
}
