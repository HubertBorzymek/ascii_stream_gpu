#include "AsciiEffect.h"

#include <stdexcept>

// CUDA headers only in .cpp
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "../../cuda/CudaUtils.h"

extern "C" void RunAsciiKernel(cudaArray_t srcArray, cudaArray_t dstArray, int width, int height);

namespace
{
    struct CudaGraphicsMapGuard
    {
        cudaGraphicsResource** resources = nullptr;
        int count = 0;
        bool mapped = false;

        CudaGraphicsMapGuard(cudaGraphicsResource** res, int n)
            : resources(res), count(n)
        {
            ThrowIfCuda(cudaGraphicsMapResources(count, resources, 0), "cudaGraphicsMapResources failed");
            mapped = true;
        }

        ~CudaGraphicsMapGuard()
        {
            if (mapped)
            {
                // Don't throw from destructor. Best-effort cleanup.
                cudaGraphicsUnmapResources(count, resources, 0);
            }
        }

        CudaGraphicsMapGuard(const CudaGraphicsMapGuard&) = delete;
        CudaGraphicsMapGuard& operator=(const CudaGraphicsMapGuard&) = delete;
    };
}

void AsciiEffect::ReleaseCudaResources()
{
    // Note: safe to call multiple times.

    if (m_cudaInputRes)
    {
        cudaGraphicsUnregisterResource(m_cudaInputRes);
        m_cudaInputRes = nullptr;
    }

    if (m_cudaOutputRes)
    {
        cudaGraphicsUnregisterResource(m_cudaOutputRes);
        m_cudaOutputRes = nullptr;
    }

    m_cudaReady = false;
}

void AsciiEffect::EnsureCudaRegistered(ID3D11Texture2D* /*inputTex*/)
{
    if (!m_inputTex || !m_outputTex)
        return;

    if (!m_cudaInputRes)
    {
        ThrowIfCuda(cudaGraphicsD3D11RegisterResource(&m_cudaInputRes, m_inputTex.Get(), cudaGraphicsRegisterFlagsNone),
            "cudaGraphicsD3D11RegisterResource(input) failed");
    }

    if (!m_cudaOutputRes)
    {
        ThrowIfCuda(cudaGraphicsD3D11RegisterResource(&m_cudaOutputRes, m_outputTex.Get(), cudaGraphicsRegisterFlagsNone),
            "cudaGraphicsD3D11RegisterResource(output) failed");
    }

    m_cudaReady = (m_cudaInputRes != nullptr) && (m_cudaOutputRes != nullptr);
}

void AsciiEffect::OnShutdown()
{
    ReleaseCudaResources();

    m_outputTex.Reset();
    m_outputDescValid = false;

    m_inputTex.Reset();
    m_inputDescValid = false;
}

void AsciiEffect::EnsureInputMatches(ID3D11Texture2D* inputTex)
{
    if (!inputTex)
        return;

    D3D11_TEXTURE2D_DESC inDesc{};
    inputTex->GetDesc(&inDesc);

    // Contract: BGRA8
    if (inDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
        throw std::runtime_error("AsciiEffect: input texture format is not DXGI_FORMAT_B8G8R8A8_UNORM (BGRA8)");

    const bool needCreate =
        !m_inputTex ||
        !m_inputDescValid ||
        m_inputDesc.Width != inDesc.Width ||
        m_inputDesc.Height != inDesc.Height ||
        m_inputDesc.Format != inDesc.Format;

    if (!needCreate)
        return;

    // If internal input texture changes, CUDA registration must be refreshed.
    ReleaseCudaResources();

    m_inputTex.Reset();

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = inDesc.Width;
    desc.Height = inDesc.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = inDesc.Format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &m_inputTex);
    if (FAILED(hr))
        throw std::runtime_error("AsciiEffect::EnsureInputMatches: CreateTexture2D failed");

    m_inputDesc = desc;
    m_inputDescValid = true;
}

void AsciiEffect::EnsureOutputMatches(ID3D11Texture2D* inputTex)
{
    if (!inputTex)
        return;

    D3D11_TEXTURE2D_DESC inDesc{};
    inputTex->GetDesc(&inDesc);

    // Contract: this effect operates on BGRA8.
    if (inDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM)
        throw std::runtime_error("AsciiEffect: input texture format is not DXGI_FORMAT_B8G8R8A8_UNORM (BGRA8)");

    const DXGI_FORMAT outFormat = DXGI_FORMAT_B8G8R8A8_UNORM;

    const bool needCreate =
        !m_outputTex ||
        !m_outputDescValid ||
        m_outputDesc.Width != inDesc.Width ||
        m_outputDesc.Height != inDesc.Height ||
        m_outputDesc.Format != outFormat;

    if (!needCreate)
        return;

    // If output texture changes, CUDA registration (later) must be refreshed too.
    // For now, just drop any CUDA state.
    ReleaseCudaResources();

    m_outputTex.Reset();

    D3D11_TEXTURE2D_DESC outDesc{};
    outDesc.Width = inDesc.Width;
    outDesc.Height = inDesc.Height;
    outDesc.MipLevels = 1;
    outDesc.ArraySize = 1;
    outDesc.Format = outFormat;
    outDesc.SampleDesc.Count = 1;
    outDesc.SampleDesc.Quality = 0;
    outDesc.Usage = D3D11_USAGE_DEFAULT;
    outDesc.BindFlags = 0;      // renderer copies into its own SRV texture anyway
    outDesc.CPUAccessFlags = 0;
    outDesc.MiscFlags = 0;

    HRESULT hr = device->CreateTexture2D(&outDesc, nullptr, &m_outputTex);
    if (FAILED(hr))
        throw std::runtime_error("AsciiEffect::EnsureOutputMatches: CreateTexture2D failed");

    m_outputDesc = outDesc;
    m_outputDescValid = true;

    // Later: register m_outputTex with CUDA here.
}

ID3D11Texture2D* AsciiEffect::ProcessImpl(ID3D11Texture2D* inputTex)
{
    EnsureInputMatches(inputTex);
    EnsureOutputMatches(inputTex);

    // Copy capture -> internal input (interop-friendly)
    context->CopyResource(m_inputTex.Get(), inputTex);

    EnsureCudaRegistered(inputTex);

    if (m_cudaReady)
    {
        cudaGraphicsResource* resArr[2] = { m_cudaInputRes, m_cudaOutputRes };
        cudaGraphicsResource** res = resArr;

        // RAII map guard ensures unmap even if an exception is thrown below.
        CudaGraphicsMapGuard mapGuard(res, 2);

        cudaArray_t inArray = nullptr;
        cudaArray_t outArray = nullptr;

        ThrowIfCuda(cudaGraphicsSubResourceGetMappedArray(&inArray, m_cudaInputRes, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray(input) failed");
        ThrowIfCuda(cudaGraphicsSubResourceGetMappedArray(&outArray, m_cudaOutputRes, 0, 0),
            "cudaGraphicsSubResourceGetMappedArray(output) failed");

        RunAsciiKernel(inArray, outArray, (int)m_outputDesc.Width, (int)m_outputDesc.Height);

        // Kernel launch error (asynchronous)
        ThrowIfCuda(cudaGetLastError(), "DebugPassthroughKernel launch failed");

#ifdef _DEBUG
        // Debug-only: force sync so execution errors surface immediately.
        ThrowIfCuda(cudaDeviceSynchronize(), "DebugPassthroughKernel execution failed");
#endif

        return m_outputTex.Get();
    }

    // Fallback (no CUDA path)
    context->CopyResource(m_outputTex.Get(), inputTex);
    return m_outputTex.Get();
}



