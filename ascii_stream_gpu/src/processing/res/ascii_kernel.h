#pragma once

#include <cuda_runtime.h>

// Host function that launches the ASCII kernel on given device buffers.
cudaError_t runAsciiKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels);


//cudaError_t runEdgesKernel(unsigned char* img_dev_in, unsigned char* img_dev_out, int width, int height, int channels);
