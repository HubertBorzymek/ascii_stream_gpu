#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <math.h>

#include "../../res/AsciiScale.h"
#include "../../res/AsciiEdges.h"
#include "../../cuda/CudaUtils.h"


#define BLOCK_SIZE   8
#define TILE_PIXELS  (BLOCK_SIZE * BLOCK_SIZE)

#define EDGE_THRESHOLD  0.2f   // 0..1 of 255
#define COH_THRESHOLD   0.5f
#define EPS             1e-6f
#define PI              3.1415926535f

// Read per-pixel luminance (BGRA8 -> float) using standard Rec. 709 coefficients
__device__ float readPixelLuma(const cudaTextureObject_t srcTex, const int x, const int y)
{
    uchar4 p = tex2D<uchar4>(srcTex, x + 0.5f, y + 0.5f);
    return
        0.0722f * (float)p.x +
        0.7152f * (float)p.y +
        0.2126f * (float)p.z;
}

// Decide if tile has an edge and, if so, return edge glyph index (0..3 or your mapping)
// Returns -1 if tile has no coherent edge
__device__ int pickEdgeGlyph(float sumMag, float sumGx, float sumGy) {
    // Mean magnitude and direction coherence
    float meanMag = sumMag / (float)TILE_PIXELS;
    float vecLen = sqrtf(sumGx * sumGx + sumGy * sumGy);
    float coherence = vecLen / (sumMag + EPS);

    bool hasEdge = (meanMag >= (255.0f * EDGE_THRESHOLD)) &&
        (coherence >= COH_THRESHOLD);

    if (!hasEdge) {
        return -1;
    }

    // Gradient direction (normal to edge)
    float theta = atan2f(sumGy, sumGx);   // [-pi, pi]

    // Edge direction = gradient rotated by 90 degrees
    theta += PI * 0.5f;

    // Normalize to [0, 2pi)
    if (theta < 0.0f)
        theta += 2.0f * PI;

    // Edge orientation modulo pi (line has no arrow)
    if (theta >= PI)
        theta -= PI; // [0, pi)

    // Quantize [0, pi) to 4 nearest orientations
    float t = 4.0f * theta / PI;        // 0..4
    int sector = (int)floorf(t + 0.5f); // nearest 0..4
    if (sector == 4) sector = 0;

    // Mapping kept as in working version
    int edgeGlyph = -1;
    switch (sector) {
    case 0: // horizontal
        edgeGlyph = 1; // '-'
        break;
    case 1: // "/"
        edgeGlyph = 3; // '/'
        break;
    case 2: // vertical
        edgeGlyph = 2; // '|'
        break;
    case 3: // "\"
    default:
        edgeGlyph = 0; // '\'
        break;
    }

    return edgeGlyph;
}

// Choose brightness-based ASCII glyph (0..15) from sum of luma in tile
__device__ int pickBrightnessGlyph(const float sumLuma) {
    float meanLuma = sumLuma / (float)TILE_PIXELS;
    int idx = static_cast<int>(meanLuma * 16.0f / 256.0f); // 0..15

    return idx;
}

// Monochrome shading helper: applies a colored ASCII pixel
__device__ void shadePixelMonochrome(cudaSurfaceObject_t dstSurf, const int x, const int y, const unsigned char base)
{
    // Monochrome tint (0.0f..1.0f)
    // Example: blue  = (1, 0, 0)
    //          green = (0, 1, 0)
    //          red   = (1, 0, 0)
    //          yellow = (0, 1, 1)
    //          nice blue = (0.9, 0.9, 0.7)
    //          nice orange = (0.2, 0.5, 0.9)
    //          nice yellow = (0.2, 0.7, 0.9)
    const float b = 0.9f;
    const float g = 0.9f;
    const float r = 0.5f;

    uchar4 out;
    out.x = static_cast<unsigned char>(base * b); // B
    out.y = static_cast<unsigned char>(base * g); // G
    out.z = static_cast<unsigned char>(base * r); // R
    out.w = 255; // A

    surf2Dwrite(out, dstSurf, x * sizeof(uchar4), y);
}

// Kernel: read BGRA8 via texture, write BGRA8 via surface.
// For now: invert colors to verify the pipeline end-to-end.
__global__ void AsciiKernel(cudaTextureObject_t srcTex, cudaSurfaceObject_t dstSurf, int width, int height)
{
    // 2D thread coordinates in block (0..BLOCK_SIZE-1)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Flattened thread index in tile: 0..TILE_PIXELS-1
    int tid = ty * blockDim.x + tx;

    // Shared buffers:
    //  - per-pixel luma (for brightness-based ASCII)
    //  - per-pixel gradient and magnitude (for edges)
    __shared__ float s_luma[TILE_PIXELS];
    __shared__ float s_mag[TILE_PIXELS];
    __shared__ float s_Gx[TILE_PIXELS];
    __shared__ float s_Gy[TILE_PIXELS];

    // One shared glyph index and a flag which table to use
    __shared__ int s_glyphIdx;     // ascii_scale index or ascii_edges index
    __shared__ int s_isEdgeTile;   // 0 = use ascii_scale, 1 = use ascii_edges

	// TODO: make sure if necessary!!!
	// necessary if input image res is not divisible by block size, otherwise some threads will read out of bounds?
    bool inside = (x < width && y < height);

    // -------------------------------------------------
    // 1) Per-thread luminance
    // -------------------------------------------------
    float luma = 255.0f; // default for out-of-image threads

    if (inside) {
        luma = readPixelLuma(srcTex, x, y);
    }

    s_luma[tid] = luma;

    // -------------------------------------------------
    // 2) Per-thread Sobel gradient on luminance
    // -------------------------------------------------
    float Gx = 0.0f;
    float Gy = 0.0f;
    float mag = 0.0f;

    // TODO: make sure if necessary!!!
    // maybe some optimalization with luma (1)
    bool interior = (x > 0) && (x < width - 1) &&
        (y > 0) && (y < height - 1);

    if (interior) {
        float L00 = readPixelLuma(srcTex, x - 1, y - 1);
        float L10 = readPixelLuma(srcTex, x    , y - 1);
        float L20 = readPixelLuma(srcTex, x + 1, y - 1);

        float L01 = readPixelLuma(srcTex, x - 1, y);
        float L11 = luma;
        float L21 = readPixelLuma(srcTex, x + 1, y);

        float L02 = readPixelLuma(srcTex, x - 1, y + 1);
        float L12 = readPixelLuma(srcTex, x    , y + 1);
        float L22 = readPixelLuma(srcTex, x + 1, y + 1);

        // Sobel X
        Gx = (-1.0f * L00) + (1.0f * L20)
            + (-2.0f * L01) + (2.0f * L21)
            + (-1.0f * L02) + (1.0f * L22);

        // Sobel Y
        Gy = (-1.0f * L00) + (-2.0f * L10) + (-1.0f * L20)
            + (1.0f * L02) + (2.0f * L12) + (1.0f * L22);

        // Approximate gradient magnitude
        mag = fabsf(Gx) + fabsf(Gy);
    }

    s_Gx[tid] = Gx;
    s_Gy[tid] = Gy;
    s_mag[tid] = mag;

    __syncthreads();

    // -------------------------------------------------
    // 3) Tile-wide reduction:
    //    - sum luma (for brightness)
    //    - sum mag, Gx, Gy (for edge decision)
    // -------------------------------------------------
    for (int stride = TILE_PIXELS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_luma[tid] += s_luma[tid + stride];
            s_mag[tid] += s_mag[tid + stride];
            s_Gx[tid] += s_Gx[tid + stride];
            s_Gy[tid] += s_Gy[tid + stride];
        }
        __syncthreads();
    }

    // -------------------------------------------------
    // 4) Thread 0: decide what glyph this tile uses
    // -------------------------------------------------
    if (tid == 0) {
        float sumLuma = s_luma[0];
        float sumMag = s_mag[0];
        float sumGx = s_Gx[0];
        float sumGy = s_Gy[0];

        // Try to detect an edge glyph from gradients
        int edgeGlyph = pickEdgeGlyph(sumMag, sumGx, sumGy);

        if (edgeGlyph >= 0) {
            // Edge tile: use ascii_edges[edgeGlyph]
            s_isEdgeTile = 1;
            s_glyphIdx = edgeGlyph;
        }
        else {
            // No edge: use brightness-based ASCII - ascii_scale[asciiIndex]
            int asciiIndex = pickBrightnessGlyph(sumLuma);
            s_isEdgeTile = 0;
            s_glyphIdx = asciiIndex;
        }
    }

    __syncthreads();

    // -------------------------------------------------
    // 5) Draw final glyph (edge overrides brightness)
    // -------------------------------------------------
	// TODO: check if necessary!!!
    if (!inside)
        return;

    unsigned char base = 0; // Default: black background

    // Decide which table to read from
    unsigned char rowBits = 0;
    if (s_isEdgeTile) {
        // Edge glyph from ascii_edges
        rowBits = (unsigned char)AsciiEdges[s_glyphIdx][ty];
    }
    else {
        // Brightness glyph from ascii_scale
        rowBits = (unsigned char)AsciiScale[s_glyphIdx][ty];
    }
    // Common bit extraction + mapping to pixel
    int bit = (rowBits >> tx) & 0x01;
    base = bit ? 255 : 0; // white symbol on black

    // Monochrome tinting
    shadePixelMonochrome(dstSurf, x, y, base);
}

extern "C" void RunAsciiKernel(cudaArray_t srcArray, cudaArray_t dstArray, int width, int height)
{
    // RAII for texture object
    struct TexObjGuard
    {
        cudaTextureObject_t obj = 0;
        TexObjGuard() = default;
        ~TexObjGuard()
        {
            if (obj) cudaDestroyTextureObject(obj);
        }
        TexObjGuard(const TexObjGuard&) = delete;
        TexObjGuard& operator=(const TexObjGuard&) = delete;
    };

    // RAII for surface object
    struct SurfObjGuard
    {
        cudaSurfaceObject_t obj = 0;
        SurfObjGuard() = default;
        ~SurfObjGuard()
        {
            if (obj) cudaDestroySurfaceObject(obj);
        }
        SurfObjGuard(const SurfObjGuard&) = delete;
        SurfObjGuard& operator=(const SurfObjGuard&) = delete;
    };

    if (!srcArray || !dstArray)
        throw std::runtime_error("RunAsciiKernel: srcArray or dstArray is null");
    if (width <= 0 || height <= 0)
        throw std::runtime_error("RunAsciiKernel: invalid width/height");

    // Create texture object for input array
    cudaResourceDesc srcRes{};
    srcRes.resType = cudaResourceTypeArray;
    srcRes.res.array.array = srcArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    TexObjGuard srcTex;
    ThrowIfCuda(cudaCreateTextureObject(&srcTex.obj, &srcRes, &texDesc, nullptr),
        "cudaCreateTextureObject failed");

    // Create surface object for output array
    cudaResourceDesc dstRes{};
    dstRes.resType = cudaResourceTypeArray;
    dstRes.res.array.array = dstArray;

    SurfObjGuard dstSurf;
    ThrowIfCuda(cudaCreateSurfaceObject(&dstSurf.obj, &dstRes),
        "cudaCreateSurfaceObject failed");

    // Launch
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    AsciiKernel <<< grid, block >>> (srcTex.obj, dstSurf.obj, width, height);

    // Catch async launch errors
    ThrowIfCuda(cudaGetLastError(), "AsciiKernel launch failed");

#ifdef _DEBUG
    // Debug-only: surface execution errors immediately
    ThrowIfCuda(cudaDeviceSynchronize(), "AsciiKernel execution failed");
#endif
}

