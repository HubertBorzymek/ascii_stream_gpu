// DebugKernel.cu
#include <cuda_runtime.h>

// Simple BGRA8 passthrough using CUDA surfaces.
// srcArray and dstArray are expected to be 2D arrays mapped from D3D11 textures (BGRA8).
__global__ void DebugPassthroughKernel(cudaSurfaceObject_t srcSurf,
    cudaSurfaceObject_t dstSurf,
    int width,
    int height)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height)
        return;


    uchar4 p;
    surf2Dread(&p, srcSurf, x * (int)sizeof(uchar4), y);

    // Invert BGRA (leave alpha)
    p.x = 255 - p.x; // B
    p.y = 255 - p.y; // G
    p.z = 255 - p.z; // R

    surf2Dwrite(p, dstSurf, x * (int)sizeof(uchar4), y);

}

extern "C" void RunDebugPassthrough(cudaArray_t srcArray,
    cudaArray_t dstArray,
    int width,
    int height)
{
    cudaResourceDesc srcDesc{};
    srcDesc.resType = cudaResourceTypeArray;
    srcDesc.res.array.array = srcArray;

    cudaResourceDesc dstDesc{};
    dstDesc.resType = cudaResourceTypeArray;
    dstDesc.res.array.array = dstArray;

    cudaSurfaceObject_t srcSurf = 0;
    cudaSurfaceObject_t dstSurf = 0;

    cudaError_t e = cudaCreateSurfaceObject(&srcSurf, &srcDesc);
    if (e != cudaSuccess) return;

    e = cudaCreateSurfaceObject(&dstSurf, &dstDesc);
    if (e != cudaSuccess) { cudaDestroySurfaceObject(srcSurf); return; }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    DebugPassthroughKernel << <grid, block >> > (srcSurf, dstSurf, width, height);

    cudaDestroySurfaceObject(srcSurf);
    cudaDestroySurfaceObject(dstSurf);
}

