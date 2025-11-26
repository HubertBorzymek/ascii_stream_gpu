#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"

#include <stdio.h>
#include <math.h>

#include "inc/ascii_scale.h"

#define BLOCK_SIZE 8
#define TILE_PIXELS (BLOCK_SIZE * BLOCK_SIZE)

#define INV_CHARS 1   // currently unused, kept for future ASCII modes


__global__ void edgesKernel(unsigned char* img_dev_in,
    unsigned char* img_dev_out,
    int width, int height, int channels);

__device__ float readPixelLuma(const unsigned char* img_dev_in,
    int width, int channels,
    int x, int y);

__device__ float calcPixelLuma(unsigned char r,
    unsigned char g,
    unsigned char b);

__device__ void shadePixelMonochrome(unsigned char* img_dev_out,
    int outIdx,
    int channels,
    unsigned char base);


// Host wrapper – launch one block per 8x8 tile
cudaError_t runEdgesKernel(unsigned char* img_dev_in,
    unsigned char* img_dev_out,
    int width, int height,
    int channels)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    edgesKernel << <gridSize, blockSize >> > (img_dev_in,
        img_dev_out,
        width, height,
        channels);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "edgesKernel launch error: %s\n",
            cudaGetErrorString(status));
        return status;
    }

    return status;
}


// Main kernel: per-thread Sobel, per-tile edge decision + orientation
__global__ void edgesKernel(unsigned char* img_dev_in,
    unsigned char* img_dev_out,
    int width, int height,
    int channels)
{
    // 2D thread coordinates inside block (0..BLOCK_SIZE-1)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Flattened thread index inside the tile: 0..TILE_PIXELS-1
    int tid = ty * blockDim.x + tx;

    // Shared buffers for per-pixel gradient and magnitude
    __shared__ float s_mag[TILE_PIXELS];
    __shared__ float s_Gx[TILE_PIXELS];
    __shared__ float s_Gy[TILE_PIXELS];

    // Shared color for the whole tile (0..1)
    __shared__ float s_r;
    __shared__ float s_g;
    __shared__ float s_b;

    bool inside = (x < width && y < height);

    // Per-thread gradient (Sobel on luminance)
    float Gx = 0.0f;
    float Gy = 0.0f;
    float mag = 0.0f;

    // Only interior pixels have a full 3x3 neighborhood
    bool interior = inside &&
        (x > 0) && (x < width - 1) &&
        (y > 0) && (y < height - 1);

    if (interior) {
        // 3x3 luminance neighborhood around (x, y)
        float L00 = readPixelLuma(img_dev_in, width, channels, x - 1, y - 1);
        float L10 = readPixelLuma(img_dev_in, width, channels, x, y - 1);
        float L20 = readPixelLuma(img_dev_in, width, channels, x + 1, y - 1);

        float L01 = readPixelLuma(img_dev_in, width, channels, x - 1, y);
        float L11 = readPixelLuma(img_dev_in, width, channels, x, y);
        float L21 = readPixelLuma(img_dev_in, width, channels, x + 1, y);

        float L02 = readPixelLuma(img_dev_in, width, channels, x - 1, y + 1);
        float L12 = readPixelLuma(img_dev_in, width, channels, x, y + 1);
        float L22 = readPixelLuma(img_dev_in, width, channels, x + 1, y + 1);

        // Sobel X:
        // [-1  0  1]
        // [-2  0  2]
        // [-1  0  1]
        Gx = (-1.0f * L00) + (1.0f * L20)
            + (-2.0f * L01) + (2.0f * L21)
            + (-1.0f * L02) + (1.0f * L22);

        // Sobel Y:
        // [-1 -2 -1]
        // [ 0  0  0]
        // [ 1  2  1]
        Gy = (-1.0f * L00) + (-2.0f * L10) + (-1.0f * L20)
            + (1.0f * L02) + (2.0f * L12) + (1.0f * L22);

        // Cheap gradient magnitude approximation
        mag = fabsf(Gx) + fabsf(Gy);
    }

    // Store per-thread results to shared memory
    s_Gx[tid] = Gx;
    s_Gy[tid] = Gy;
    s_mag[tid] = mag;

    __syncthreads();

    // -----------------------------------
    // Tile reduction: sum mag, Gx, Gy
    // -----------------------------------

    for (int stride = TILE_PIXELS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mag[tid] += s_mag[tid + stride];
            s_Gx[tid] += s_Gx[tid + stride];
            s_Gy[tid] += s_Gy[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 decides if this tile is an edge tile
    // and computes the dominant edge orientation
    if (tid == 0) {
        float sumMag = s_mag[0];                 // sum of |grad| in tile
        float meanMag = sumMag / (float)TILE_PIXELS;

        float sumGx = s_Gx[0];                    // sum of Gx over tile
        float sumGy = s_Gy[0];                    // sum of Gy over tile

        // --- Edge decision parameters ---
        const float EDGE_THRESHOLD = 50.0f;      // magnitude threshold (tune)
        const float COH_THRESHOLD = 0.6f;        // direction coherence threshold (0..1, tune)
        const float EPS = 1e-6f;

        // Direction coherence:
        // C = |sum(grad)| / sum(|grad|)
        float vecLen = sqrtf(sumGx * sumGx + sumGy * sumGy);
        float coherence = (sumMag > 0.0f) ? (vecLen / (sumMag + EPS)) : 0.0f;

        bool hasEdge = (meanMag >= EDGE_THRESHOLD) &&
            (coherence >= COH_THRESHOLD);

        float baseR, baseG, baseB;

        if (!hasEdge) {
            // No strong, coherent edge: dark grey tile
            baseR = 0.2f;
            baseG = 0.2f;
            baseB = 0.2f;
        }
        else {
            // Compute dominant edge orientation from summed gradient
            const float PI = 3.1415926535f;

            // Gradient direction (normal to edge)
            float theta = atan2f(sumGy, sumGx);   // [-pi, pi]

            // Edge direction = gradient rotated by 90 degrees
            theta += PI * 0.5f;

            // Normalize to [0, 2pi)
            if (theta < 0.0f)       theta += 2.0f * PI;
            if (theta >= 2.0f * PI) theta -= 2.0f * PI;

            // Edge orientation is modulo pi (line has no arrow)
            if (theta >= PI) theta -= PI;         // now in [0, pi)

            // Quantize [0, pi) to 4 nearest orientations:
            // 0   -> horizontal
            // pi/4 -> diagonal "/"
            // pi/2 -> vertical
            // 3pi/4 -> diagonal "\"
            float t = 4.0f * theta / PI;          // 0..4
            int sector = (int)floorf(t + 0.5f);   // round to nearest 0..4
            if (sector == 4) sector = 0;         // wrap back to horizontal

            switch (sector) {
            case 0: // horizontal
                baseR = 1.0f; baseG = 0.0f; baseB = 0.0f;   // red
                break;
            case 1: // "/"
                baseR = 1.0f; baseG = 1.0f; baseB = 0.0f;   // yellow
                break;
            case 2: // vertical
                baseR = 0.0f; baseG = 1.0f; baseB = 0.0f;   // green
                break;
            case 3: // "\"
            default:
                baseR = 0.0f; baseG = 0.0f; baseB = 1.0f;   // blue
                break;
            }
        }

        // Tile brightness: full for edge tiles, dimmer for background
        float bright = hasEdge ? 1.0f : 0.5f;

        s_r = baseR * bright;
        s_g = baseG * bright;
        s_b = baseB * bright;
    }

    __syncthreads();

    // Write tile color to output image
    if (!inside)
        return;

    int outIdx = (y * width + x) * channels;

    unsigned char R = (unsigned char)fminf(s_r * 255.0f, 255.0f);
    unsigned char G = (unsigned char)fminf(s_g * 255.0f, 255.0f);
    unsigned char B = (unsigned char)fminf(s_b * 255.0f, 255.0f);

    img_dev_out[outIdx + 0] = R;
    img_dev_out[outIdx + 1] = G;
    img_dev_out[outIdx + 2] = B;

    if (channels == 4) {
        img_dev_out[outIdx + 3] = 255;
    }
}


// Read per-pixel luminance (Rec.709) from RGB image
__device__ float readPixelLuma(const unsigned char* img_dev_in,
    int width, int channels,
    int x, int y)
{
    int idx = (y * width + x) * channels;
    return calcPixelLuma(img_dev_in[idx + 0],
        img_dev_in[idx + 1],
        img_dev_in[idx + 2]);
}


// Standard Rec.709 luma approximation
__device__ float calcPixelLuma(unsigned char r,
    unsigned char g,
    unsigned char b)
{
    float lum = 0.2126f * (float)r +
        0.7152f * (float)g +
        0.0722f * (float)b;
    return lum;
}


// Monochrome shading helper (currently unused here, kept for future reuse)
__device__ void shadePixelMonochrome(unsigned char* img_dev_out,
    int outIdx,
    int channels,
    unsigned char base)
{
    const float r = 0.99f;
    const float g = 0.99f;
    const float b = 0.99f;

    img_dev_out[outIdx + 0] = (unsigned char)(base * r);
    img_dev_out[outIdx + 1] = (unsigned char)(base * g);
    img_dev_out[outIdx + 2] = (unsigned char)(base * b);

    if (channels == 4) {
        img_dev_out[outIdx + 3] = 255;
    }
}
