#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"

#include <stdio.h>
#include <math.h>

#include "inc/ascii_scale.h"
#include "inc/ascii_edges.h"

#define BLOCK_SIZE   8
#define TILE_PIXELS  (BLOCK_SIZE * BLOCK_SIZE)

#define EDGE_THRESHOLD  0.30f   // 0..1 of 255
#define COH_THRESHOLD   0.6f
#define EPS             1e-6f
#define PI              3.1415926535f


__global__ void asciiKernel(
    unsigned char* img_dev_in,
    unsigned char* img_dev_out,
    int width,
    int height,
    int channels
);

__device__ float readPixelLuma(const unsigned char* img_dev_in, int width, int channels, int x, int y);
__device__ float calcPixelLuma(unsigned char r, unsigned char g, unsigned char b);
__device__ int pickBrightnessGlyph(float sumLuma);
__device__ int pickEdgeGlyph(float sumMag, float sumGx, float sumGy);
__device__ void shadePixelMonochrome(unsigned char* img_dev_out, int outIdx, int channels, unsigned char base);




// Host wrapper – one block per 8x8 tile
cudaError_t runAsciiKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    asciiKernel <<<gridSize, blockSize>>> (img_dev_in, img_dev_out, width, height, channels);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "asciiKernel launch error: %s\n",
            cudaGetErrorString(status));
        return status;
    }

    return status;
}


// Unified kernel: brightness-based ASCII + edge glyph override
__global__ void asciiKernel(
    unsigned char* img_dev_in,
    unsigned char* img_dev_out,
    int width,
    int height,
    int channels
)
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
    // - per-pixel luma (for brightness-based ASCII)
    // - per-pixel gradient and magnitude (for edges)
    __shared__ float s_luma[TILE_PIXELS];
    __shared__ float s_mag[TILE_PIXELS];
    __shared__ float s_Gx[TILE_PIXELS];
    __shared__ float s_Gy[TILE_PIXELS];

    // One shared glyph index and a flag which table to use
    __shared__ int s_glyphIdx;     // ascii_scale index or ascii_edges index
    __shared__ int s_isEdgeTile;   // 0 = use ascii_scale, 1 = use ascii_edges

    bool inside = (x < width && y < height);

    // -------------------------------------------------
    // 1) Per-thread luminance
    // -------------------------------------------------
    float luma = 255.0f; // default for out-of-image threads

    if (inside) {
        luma = readPixelLuma(img_dev_in, width, channels, x, y);
    }

    s_luma[tid] = luma;

    // -------------------------------------------------
    // 2) Per-thread Sobel gradient on luminance
    // -------------------------------------------------
    float Gx = 0.0f;
    float Gy = 0.0f;
    float mag = 0.0f;

    bool interior = (x > 0) && (x < width - 1) &&
        (y > 0) && (y < height - 1);

    if (interior) {
        float L00 = readPixelLuma(img_dev_in, width, channels, x - 1, y - 1);
        float L10 = readPixelLuma(img_dev_in, width, channels, x    , y - 1);
        float L20 = readPixelLuma(img_dev_in, width, channels, x + 1, y - 1);

        float L01 = readPixelLuma(img_dev_in, width, channels, x - 1, y);
        float L11 = luma;
        float L21 = readPixelLuma(img_dev_in, width, channels, x + 1, y);

        float L02 = readPixelLuma(img_dev_in, width, channels, x - 1, y + 1);
        float L12 = readPixelLuma(img_dev_in, width, channels, x    , y + 1);
        float L22 = readPixelLuma(img_dev_in, width, channels, x + 1, y + 1);

        // Sobel X
        Gx =  (-1.0f * L00) + (1.0f * L20)
            + (-2.0f * L01) + (2.0f * L21)
            + (-1.0f * L02) + (1.0f * L22);

        // Sobel Y
        Gy = (-1.0f * L00) + (-2.0f * L10) + (-1.0f * L20)
           + ( 1.0f * L02) + ( 2.0f * L12) + ( 1.0f * L22);

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
    if (!inside)
        return;

    int outIdx = (y * width + x) * channels;
    unsigned char base = 0; // Default: black background

    // Decide which table to read from
    unsigned char rowBits = 0;
    if (s_isEdgeTile) {
        // Edge glyph from ascii_edges
        rowBits = (unsigned char)ascii_edges[s_glyphIdx][ty];
    }
    else {
        // Brightness glyph from ascii_scale
        rowBits = (unsigned char)ascii_scale[s_glyphIdx][ty];
    }
    // Common bit extraction + mapping to pixel
    int bit = (rowBits >> tx) & 0x01;
    base = bit ? 255 : 0; // white symbol on black

    // Monochrome tinting
    shadePixelMonochrome(img_dev_out, outIdx, channels, base);
}


// Read per-pixel luminance from RGB(A) image
__device__ float readPixelLuma(const unsigned char* img_dev_in, int width, int channels, int x, int y) {
    int idx = (y * width + x) * channels;
    return calcPixelLuma(
        img_dev_in[idx + 0],
        img_dev_in[idx + 1],
        img_dev_in[idx + 2]
    );
}


// Standard Rec.709 luma approximation
__device__ float calcPixelLuma(unsigned char r, unsigned char g, unsigned char b) {
    float lum = 0.2126f * (float)r +
        0.7152f * (float)g +
        0.0722f * (float)b;
    return lum;
}


// Choose brightness-based ASCII glyph (0..15) from sum of luma in tile
__device__ int pickBrightnessGlyph(float sumLuma) {
    float meanLuma = sumLuma / (float)TILE_PIXELS;
    int idx = static_cast<int>(meanLuma * 16.0f / 256.0f); // 0..15

    return idx;
}

// Decide if tile has an edge and, if so, return edge glyph index (0..3 or your mapping)
// Returns -1 if tile has no coherent edge
__device__ int pickEdgeGlyph(float sumMag, float sumGx, float sumGy) {
    // Mean magnitude and direction coherence
    float meanMag = sumMag / (float)TILE_PIXELS;
    float vecLen = sqrtf(sumGx * sumGx + sumGy * sumGy);
    float coherence = vecLen / (sumMag + EPS);

    bool hasEdge =  (meanMag >= (255.0f * EDGE_THRESHOLD)) &&
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

// Monochrome shading helper: applies a colored ASCII pixel
__device__ void shadePixelMonochrome(unsigned char* img_dev_out,
    int outIdx,
    int channels,
    unsigned char base)
{
    // Monochrome tint (0.0f..1.0f)
    // Example: red   = (1, 0, 0)
    //          green = (0, 1, 0)
    //          blue  = (0, 0, 1)
    //          yellow = (1, 1, 0)
    //          nice blue = (0.7, 0.9, 0.9)
    //          nice orange = (0.9, 0.5, 0.2)
    //          nice yellow = (0.9, 0.7, 0.2)
    const float r = 0.7f;
    const float g = 0.9f;
    const float b = 0.9f;

    img_dev_out[outIdx + 0] = static_cast<unsigned char>(base * r);
    img_dev_out[outIdx + 1] = static_cast<unsigned char>(base * g);
    img_dev_out[outIdx + 2] = static_cast<unsigned char>(base * b);

    if (channels == 4) {
        img_dev_out[outIdx + 3] = 255;
    }
}


