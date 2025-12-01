#include <iostream>
#include <Windows.h>    // HMONITOR, MonitorFromPoint, Sleep

// C++/WinRT
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

// D3D11 / DXGI
#include <d3d11.h>
#include <dxgi1_2.h>

// WinRT interop headers (bridge between D3D11 and WinRT)
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <wrl.h>

using Microsoft::WRL::ComPtr;


#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

// Alias for WinRT Direct3D device type
using WinRTDirect3DDevice = winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
using WinRTInspectable = winrt::Windows::Foundation::IInspectable;


// --------------------------------------------------------
// Create a pure D3D11 device (GPU device)
// --------------------------------------------------------
com_ptr<ID3D11Device> createD3DDevice()
{
    com_ptr<ID3D11Device> device;
    com_ptr<ID3D11DeviceContext> context;

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

    D3D_FEATURE_LEVEL chosenLevel = D3D_FEATURE_LEVEL_11_0;

    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // use GPU
        nullptr,
        flags,
        featureLevels,
        _countof(featureLevels),
        D3D11_SDK_VERSION,
        device.put(),
        &chosenLevel,
        context.put()
    );

    if (FAILED(hr))
    {
        std::cout << "D3D11CreateDevice failed, hr = 0x"
            << std::hex << hr << std::dec << "\n";
        throw hresult_error(hr);
    }

    std::cout << "D3D11 device created. Feature level = 0x"
        << std::hex << chosenLevel << std::dec << "\n";

    return device;
}

// --------------------------------------------------------
// Wrap D3D11 device as WinRT IDirect3DDevice
//   (needed by Direct3D11CaptureFramePool)
// --------------------------------------------------------
WinRTDirect3DDevice createWinRTDirect3DDevice(com_ptr<ID3D11Device> const& d3dDevice)
{
    com_ptr<IDXGIDevice> dxgiDevice = d3dDevice.as<IDXGIDevice>();

    com_ptr<::IInspectable> inspectable;

    HRESULT hr = CreateDirect3D11DeviceFromDXGIDevice(
        dxgiDevice.get(),
        inspectable.put()
    );

    if (FAILED(hr))
    {
        std::cout << "CreateDirect3D11DeviceFromDXGIDevice failed, hr = 0x"
            << std::hex << hr << std::dec << "\n";
        throw hresult_error(hr);
    }

    WinRTDirect3DDevice winrtDevice = inspectable.as<WinRTDirect3DDevice>();
    return winrtDevice;
}

// --------------------------------------------------------
// Create GraphicsCaptureItem for PRIMARY monitor
// --------------------------------------------------------
GraphicsCaptureItem createCaptureItemForPrimaryMonitor()
{
    POINT pt{ 0, 0 };
    HMONITOR hmon = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);

    if (!hmon)
    {
        throw hresult_error(E_FAIL, L"MonitorFromPoint failed.");
    }

    // Get COM interop interface for GraphicsCaptureItem factory
    com_ptr<IGraphicsCaptureItemInterop> interop =
        get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();

    GraphicsCaptureItem item{ nullptr };

    // Create a capture item for the given HMONITOR
    check_hresult(interop->CreateForMonitor(
        hmon,
        guid_of<GraphicsCaptureItem>(),
        put_abi(item)
    ));

    return item;
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main()
{
    // Initialize C++/WinRT
    init_apartment();

    // Check if Windows Graphics Capture is supported
    if (!GraphicsCaptureSession::IsSupported())
    {
        std::cout << "Windows Graphics Capture is NOT supported.\n";
        return 0;
    }

    std::cout << "Windows Graphics Capture IS supported.\n";

    // 1. Create D3D11 device
    com_ptr<ID3D11Device> d3dDevice;
    try
    {
        d3dDevice = createD3DDevice();
    }
    catch (const hresult_error& e)
    {
        std::wcout << L"Failed to create D3D11 device: "
            << e.message().c_str() << L"\n";
        return 0;
    }

    // 2. Wrap it as WinRT IDirect3DDevice
    WinRTDirect3DDevice winrtDevice{ nullptr };
    try
    {
        winrtDevice = createWinRTDirect3DDevice(d3dDevice);
    }
    catch (const hresult_error& e)
    {
        std::wcout << L"Failed to create WinRT Direct3D device: "
            << e.message().c_str() << L"\n";
        return 0;
    }

    // 3. Create capture item for primary monitor
    GraphicsCaptureItem item{ nullptr };
    try
    {
        item = createCaptureItemForPrimaryMonitor();
    }
    catch (const hresult_error& e)
    {
        std::wcout << L"Failed to create capture item: "
            << e.message().c_str() << L"\n";
        return 0;
    }

    auto size = item.Size();
    std::cout << "Capture item size: "
        << size.Width << " x " << size.Height << "\n";

    // 4. Create frame pool and capture session (free-threaded for console app)
    auto framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        winrtDevice,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        1,          // number of buffered frames
        size
    );

    GraphicsCaptureSession session = framePool.CreateCaptureSession(item);

    bool gotFrame = false;
    Direct3D11CaptureFrame firstFrame{ nullptr };

    // Event handler: triggered when a new frame is available
    framePool.FrameArrived([&](Direct3D11CaptureFramePool const& pool, WinRTInspectable  const&)
        {
            if (gotFrame)
                return;

            // TryGetNextFrame is non-blocking
            firstFrame = pool.TryGetNextFrame();
            if (firstFrame)
            {
                gotFrame = true;
            }
        });

    // 5. Start capturing
    session.StartCapture();
    std::cout << "Capture session started. Waiting for first frame...\n";

    // 6. Wait until we get one frame
    while (!gotFrame)
    {
        ::Sleep(10); // simple polling; ok for test
    }

    // 7. Stop capture
    session.Close();
    framePool.Close();

    // Basic info about captured frame (WinRT size)
    auto contentSize = firstFrame.ContentSize();
    std::cout << "Captured one frame. Frame content size: "
        << contentSize.Width << " x " << contentSize.Height << "\n";

    // 8. Extract ID3D11Texture2D from frame surface
    auto surface = firstFrame.Surface();   // WinRT IDirect3DSurface

    // Get raw IInspectable* from WinRT object
    IInspectable* inspectable = reinterpret_cast<IInspectable*>(winrt::get_abi(surface));

    // Query for IDirect3DDxgiInterfaceAccess (COM interop interface)
    ComPtr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access;
    HRESULT hr = inspectable->QueryInterface(
        __uuidof(::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess),
        reinterpret_cast<void**>(access.GetAddressOf())
    );
    check_hresult(hr);

    // Now get ID3D11Texture2D from that interop interface
    ComPtr<ID3D11Texture2D> texture;
    hr = access->GetInterface(
        __uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(texture.GetAddressOf())
    );
    check_hresult(hr);

    // 9. Query texture description
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);

    std::cout << "D3D11 texture from frame:\n";
    std::cout << "  Width     = " << desc.Width << "\n";
    std::cout << "  Height    = " << desc.Height << "\n";
    std::cout << "  MipLevels = " << desc.MipLevels << "\n";
    std::cout << "  ArraySize = " << desc.ArraySize << "\n";
    std::cout << "  Format    = " << desc.Format << " (DXGI_FORMAT)\n";

    return 0;
}

