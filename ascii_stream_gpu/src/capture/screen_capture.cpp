#include "screen_capture.h"

#include <stdexcept>
#include <iostream>

#include <dxgi1_2.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

static void ThrowIfFailed(HRESULT hr, const char* msg)
{
    if (FAILED(hr))
        throw std::runtime_error(msg);
}

// Wraps an ID3D11Device as WinRT IDirect3DDevice (required by WGC FramePool).
static IDirect3DDevice CreateWinRTDirect3DDevice(ComPtr<ID3D11Device> const& d3dDevice)
{
    ComPtr<IDXGIDevice> dxgiDevice;
    ThrowIfFailed(d3dDevice.As(&dxgiDevice), "As IDXGIDevice failed");

    winrt::com_ptr<::IInspectable> inspectable;
    HRESULT hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectable.put());
    ThrowIfFailed(hr, "CreateDirect3D11DeviceFromDXGIDevice failed");

    return inspectable.as<IDirect3DDevice>();
}

// Creates a GraphicsCaptureItem for the primary monitor.
static GraphicsCaptureItem CreateCaptureItemForPrimaryMonitor()
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

// Converts WinRT IDirect3DSurface to a COM interop interface that can expose
// the underlying DXGI/D3D11 resource.
static ComPtr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>
AsDxgiAccess(IDirect3DSurface const& surface)
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

ScreenCapture::~ScreenCapture()
{
    Stop();
}

void ScreenCapture::Start(ComPtr<ID3D11Device> device)
{
    if (!device)
        throw std::runtime_error("ScreenCapture::Start: device is null");

    if (!GraphicsCaptureSession::IsSupported())
        throw std::runtime_error("Windows Graphics Capture is not supported");

    // Create WinRT device (bridge for WGC).
    m_winrtDevice = CreateWinRTDirect3DDevice(device);

    // Create capture item (primary monitor).
    m_item = CreateCaptureItemForPrimaryMonitor();
    auto size = m_item.Size();
    m_width = size.Width;
    m_height = size.Height;

    // Create frame pool and session.
    m_framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        m_winrtDevice,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        size
    );

    m_session = m_framePool.CreateCaptureSession(m_item);

    // Register callback.
    m_framePool.FrameArrived([this](Direct3D11CaptureFramePool const& pool, winrt::Windows::Foundation::IInspectable const&)
        {
            this->OnFrameArrived(pool);
        });

    // Start capture.
    m_session.StartCapture();
}

void ScreenCapture::Stop()
{
    if (m_session)
    {
        m_session.Close();
        m_session = nullptr;
    }
    if (m_framePool)
    {
        m_framePool.Close();
        m_framePool = nullptr;
    }

    m_item = nullptr;
    m_winrtDevice = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_latestTex.Reset();
    }

    m_width = 0;
    m_height = 0;
}

ComPtr<ID3D11Texture2D> ScreenCapture::LatestTexture()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_latestTex;
}

void ScreenCapture::OnFrameArrived(Direct3D11CaptureFramePool const& pool)
{
    auto frame = pool.TryGetNextFrame();
    if (!frame)
        return;

    auto surface = frame.Surface();
    auto access = AsDxgiAccess(surface);
    if (!access)
        return;

    ComPtr<ID3D11Texture2D> tex;
    HRESULT hr = access->GetInterface(__uuidof(ID3D11Texture2D), (void**)tex.GetAddressOf());
    if (FAILED(hr) || !tex)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    m_latestTex = tex;
}
