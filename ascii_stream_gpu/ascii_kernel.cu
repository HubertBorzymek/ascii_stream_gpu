
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"

#include <stdio.h>

#include "inc/ascii_scale.h"

#define BLOCK_SIZE 8

#define INV_CHARS 1


__global__ void asciiKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels);
__device__ float calcPixelLuma(unsigned char r, unsigned char g, unsigned char b);
__device__ void shadePixelMonochrome(unsigned char* img_dev_out, int outIdx, int channels, unsigned char base);


cudaError_t runAsciiKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels) {
	// Define block and grid sizes
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Kernel to convert image to ASCII
	asciiKernel <<<gridSize, blockSize>>> (img_dev_in, img_dev_out, width, height, channels);

	// Check launch error
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "asciiKernel launch error: %s\n",
			cudaGetErrorString(status));
		return status;
	}

	return status;
}

__global__ void asciiKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels) {
	// 2D thread coordinates inside block (0..7)
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Global pixel coordinates
	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	// Flattened thread index inside block: 0..63
	int tid = ty * blockDim.x + tx;

	// Shared memory for brightness values and chosen ASCII index
	__shared__ float s_luma[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ int s_asciiIndex;



	float luma = 255.0f;
	// Check if within image bounds
	if(x < width && y < height) {
		int idx = (y * width + x) * channels;
		luma = calcPixelLuma(img_dev_in[idx + 0],
							img_dev_in[idx + 1],
							img_dev_in[idx + 2]);
	}

	// Store brightness in shared memory
	s_luma[tid] = luma;
	__syncthreads();

	// Tree reduction to compute sum of brightness in tile
	// Assumes blockDim.x == blockDim.y (64)
	for (int stride = BLOCK_SIZE * BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			s_luma[tid] += s_luma[tid + stride];
		}
		__syncthreads();
	}

	// Thread 0 computes mean brightness and selects ASCII index
	if (tid == 0) {
		float sum = s_luma[0];
		float mean = sum / 64.0f;   // 8 * 8 = 64 pixels

		// Map mean in [0, 255] to index in [0, 15]
		// ascii_scale is ordered bright -> dark, so we keep it monotonic
		int idx = static_cast<int>(mean * 16.0f / 256.0f);
		if (idx < 0)   idx = 0;
		if (idx > 15)  idx = 15;

		s_asciiIndex = idx;
	}

	__syncthreads();

	// All threads now know which ASCII glyph to use (reverse seuence).
#if INV_CHARS
	int asciiIdx = s_asciiIndex;
#else
	int asciiIdx = 15 - s_asciiIndex;
#endif

	// Draw chosen glyph into the output image
	if (x < width && y < height) {
		// Row of glyph corresponds to ty (0..7)
		unsigned char rowBits = static_cast<unsigned char>(ascii_scale[asciiIdx][ty]);

		// LSB..MSB left-to-right, so col = tx uses bit 'tx'
		int bit = (rowBits >> tx) & 0x01;

		// Base brightness: 1 = ink (dark), 0 = background (light)
#if INV_CHARS
		unsigned char base = bit ? 255 : 0;
#else
		unsigned char base = bit ? 0 : 255;
#endif

		int outIdx = (y * width + x) * channels;

		// Draw pixel!
		shadePixelMonochrome(img_dev_out, outIdx, channels, base);

		// If there is alpha channel
		if (channels == 4) {
			img_dev_out[outIdx + 3] = 255;
		}
	}

	return;	
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


