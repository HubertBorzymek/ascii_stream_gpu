
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"

#include <stdio.h>

#include "inc/ascii_scale.h"

#define BLOCK_SIZE 8
#define TILE_PIXELS (BLOCK_SIZE * BLOCK_SIZE)

#define INV_CHARS 1


__global__ void edgesKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels);
__device__ float readPixelLuma(const unsigned char* img_dev_in, int width, int channels, int x, int y);
__device__ float calcPixelLuma(unsigned char r, unsigned char g, unsigned char b);
__device__ void shadePixelMonochrome(unsigned char* img_dev_out, int outIdx, int channels, unsigned char base);


cudaError_t runEdgesKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels) {
	// Define block and grid sizes
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Kernel to convert image to edges
	edgesKernel << <gridSize, blockSize >> > (img_dev_in, img_dev_out, width, height, channels);

	// Check launch error
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "asciiKernel launch error: %s\n",
			cudaGetErrorString(status));
		return status;
	}

	return status;
}

__global__ void edgesKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels) {
	// 2D thread coordinates inside block (0..7)
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Global pixel coordinates
	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	// Flattened thread index inside block: 0..TILE_PIXELS-1
	int tid = ty * blockDim.x + tx;

	// Shared buffers for per-pixel data
	__shared__ float s_mag[TILE_PIXELS];
	__shared__ float s_Gx[TILE_PIXELS];
	__shared__ float s_Gy[TILE_PIXELS];

	// Shared color for the whole tile (0..1)
	__shared__ float s_r;
	__shared__ float s_g;
	__shared__ float s_b;

	// -----------------------------------
	// 1. Per-thread: compute Sobel on luma
	// -----------------------------------

	float Gx = 0.0f;
	float Gy = 0.0f;
	float mag = 0.0f;

	bool inside = (x < width && y < height);

	// For border pixels we skip Sobel (no neighbors)
	bool interior = inside &&
		(x > 0) && (x < width - 1) &&
		(y > 0) && (y < height - 1);

	if (interior) {
		// 3x3 neighborhood
		float L00 = readPixelLuma(img_dev_in, width, channels, x - 1, y - 1);
		float L10 = readPixelLuma(img_dev_in, width, channels, x	, y - 1);
		float L20 = readPixelLuma(img_dev_in, width, channels, x + 1, y - 1);

		float L01 = readPixelLuma(img_dev_in, width, channels, x - 1, y);
		float L11 = readPixelLuma(img_dev_in, width, channels, x	, y);
		float L21 = readPixelLuma(img_dev_in, width, channels, x + 1, y);

		float L02 = readPixelLuma(img_dev_in, width, channels, x - 1, y + 1);
		float L12 = readPixelLuma(img_dev_in, width, channels, x	, y + 1);
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

		// Approximate gradient magnitude (cheap)
		mag = fabsf(Gx) + fabsf(Gy);
	}

	// Store to shared memory
	s_Gx[tid] = Gx;
	s_Gy[tid] = Gy;
	s_mag[tid] = mag;

	__syncthreads();

	// -----------------------------------
	// 2. Tile reduction: sum mag, Gx, Gy
	// -----------------------------------

	// Tree reduction over TILE_PIXELS elements
	for (int stride = TILE_PIXELS / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			s_mag[tid] += s_mag[tid + stride];
			s_Gx[tid] += s_Gx[tid + stride];
			s_Gy[tid] += s_Gy[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		float sumMag = s_mag[0];
		float meanMag = sumMag / (float)TILE_PIXELS;

		float sumGx = s_Gx[0];
		float sumGy = s_Gy[0];

		// --- Binary decision: is there an edge in this tile? ---
		// Ten próg bêdziesz sobie stroi³ – na pocz¹tek rz¹d kilkudziesiêciu.
		const float EDGE_THRESHOLD = 30.0f;

		bool hasEdge = (meanMag >= EDGE_THRESHOLD) &&
			(fabsf(sumGx) > 1e-3f || fabsf(sumGy) > 1e-3f);

		float baseR, baseG, baseB;

		if (!hasEdge) {
			// No edge: dark grey tile
			baseR = 0.2f;
			baseG = 0.2f;
			baseB = 0.2f;
		}
		else {
            const float PI = 3.1415926535f;

            // gradient angle
            float theta = atan2f(sumGy, sumGx); // [-pi, pi]
            
			theta += PI / 2;

			// [0, 2pi)
			if (theta < 0.0f)      theta += 2.0f * PI;
			if (theta >= 2.0f * PI)  theta -= 2.0f * PI;

			// [0, pi)
			if (theta >= PI) theta -= PI;


			float t = 4.0f * theta / PI;
			int sector = (int)floorf(t + 0.5f);  // rounding
			if (sector == 4) sector = 0; 


			switch (sector) {
			case 0: // horizontal
				baseR = 1.0f; baseG = 0.0f; baseB = 0.0f;
				break;
			case 1: // "/"
				baseR = 1.0f; baseG = 1.0f; baseB = 0.0f;
				break;
			case 2: // vertical
				baseR = 0.0f; baseG = 1.0f; baseB = 0.0f;
				break;
			case 3: // "\"
			default:
				baseR = 0.0f; baseG = 0.0f; baseB = 1.0f;
				break;
			}
        }


		// Brightness: binary – edges full bright, background dark
		float bright = hasEdge ? 1.0f : 0.5f;

		s_r = baseR * bright;
		s_g = baseG * bright;
		s_b = baseB * bright;
	}

	__syncthreads();

	// --- Write tile color ---
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



	return;
}

__device__ float readPixelLuma(const unsigned char* img_dev_in, int width, int channels, int x, int y) {
	int idx = (y * width + x) * channels;
	return calcPixelLuma(img_dev_in[idx + 0],
		img_dev_in[idx + 1],
		img_dev_in[idx + 2]);
}

__device__ float calcPixelLuma(unsigned char r, unsigned char g, unsigned char b) {
	// Standard Rec.709 luma approximation
	float lum = 0.2126f * (float)r + 0.7152f * (float)g + 0.0722f * (float)b;
	return lum;
}

// Monochrome shading helper: applies a colored ASCII pixel
__device__ void shadePixelMonochrome(unsigned char* img_dev_out, int outIdx, int channels, unsigned char base) {
	// Monochrome tint (0.0f..1.0f)
	// Example: red   = (1, 0, 0)
	//          green = (0, 1, 0)
	//          blue  = (0, 0, 1)
	//          yellow = (1, 1, 0)
	//          nice blue = (0.7, 0.9, 0.9)
	//          nice orange = (0.9, 0.5, 0.2)
	//			nice yellow = (0.9, 0.7, 0.2)
	const float r = 0.99f;
	const float g = 0.99f;
	const float b = 0.99f;

	img_dev_out[outIdx + 0] = static_cast<unsigned char>(base * r);
	img_dev_out[outIdx + 1] = static_cast<unsigned char>(base * g);
	img_dev_out[outIdx + 2] = static_cast<unsigned char>(base * b);

	if (channels == 4) {
		img_dev_out[outIdx + 3] = 255;
	}
}


