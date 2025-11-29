#define _CRT_SECURE_NO_WARNINGS

// stb_image: image loading
#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"

// stb_image_write: image saving
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "inc/ascii_kernel.h"   // runAsciiKernel declaration



struct HostImage {
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    size_t bytes = 0;
};

struct DeviceBuffers {
    unsigned char* in = nullptr;
    unsigned char* out = nullptr;
    size_t bytes = 0;
};


// Try to load "input/input.png", if it fails then "input/input.jpg".
bool loadFixedInputImage(HostImage& img) {
    const std::string pngPath = "input/input.png";
    const std::string jpgPath = "input/input.jpg";

    int w = 0, h = 0, ch = 0;

    // First try PNG
    unsigned char* data = stbi_load(pngPath.c_str(), &w, &h, &ch, 0);
    std::string usedPath = pngPath;

    if (!data) {
        std::cerr << "Could not load " << pngPath << ", trying " << jpgPath << "...\n";
        data = stbi_load(jpgPath.c_str(), &w, &h, &ch, 0);
        usedPath = jpgPath;
    }

    if (!data) {
        std::cerr << "Error: cannot load input image (input/input.png or input/input.jpg).\n";
        return false;
    }

    std::cout << "Loaded image: " << usedPath
        << " (" << w << " x " << h
        << ", channels = " << ch << ")\n";

    // RGB or ARGB
    if (ch != 3 && ch != 4) {
        std::cerr << "Error: expected 3 or 4 channels, got " << ch << "\n";
        stbi_image_free(data);
        return false;
    }


    img.data = data;
    img.width = w;
    img.height = h;
    img.channels = ch;
    img.bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(ch);

    return true;
}


// Allocate device buffers and copy input image to GPU.
bool prepareDeviceBuffers(const HostImage& img, DeviceBuffers& dev) {
    dev.bytes = img.bytes;

    checkCudaErrors(cudaSetDevice(0));

    dev.in = nullptr;
    dev.out = nullptr;

    checkCudaErrors(cudaMalloc((void**)&dev.in, dev.bytes));
    checkCudaErrors(cudaMalloc((void**)&dev.out, dev.bytes));

    checkCudaErrors(cudaMemcpy(dev.in, img.data, dev.bytes, cudaMemcpyHostToDevice));

    return true;
}


// Save output image from host buffer to fixed path "output/output.png".
bool saveFixedOutputImage(const HostImage& img,
    const std::vector<unsigned char>& out_host) {
    const std::string outPath = "output/output.png";
    const int stride_in_bytes = img.width * img.channels;

    if (!stbi_write_png(outPath.c_str(),
        img.width,
        img.height,
        img.channels,
        out_host.data(),
        stride_in_bytes)) {
        std::cerr << "Error: cannot write output: " << outPath << "\n";
        return false;
    }

    std::cout << "Saved output image: " << outPath << "\n";
    return true;
}


// Cleanup host + device resources.
void cleanupAll(HostImage& img, DeviceBuffers& dev) {
    if (img.data) {
        stbi_image_free(img.data);
        img.data = nullptr;
    }

    if (dev.in) {
        cudaFree(dev.in);
        dev.in = nullptr;
    }
    if (dev.out) {
        cudaFree(dev.out);
        dev.out = nullptr;
    }

    cudaDeviceReset();
}


// Main function: fixed input/output filenames (no argc/argv logic).
int main() {
    HostImage img;
    DeviceBuffers dev;

    // 1–2. Load input image (fixed filenames)
    if (!loadFixedInputImage(img)) {
        cleanupAll(img, dev);
        return EXIT_FAILURE;
    }

    // Host output buffer
    std::vector<unsigned char> out_host(img.bytes);

    // 3–4. Allocate GPU buffers and copy input
    if (!prepareDeviceBuffers(img, dev)) {
        cleanupAll(img, dev);
        return EXIT_FAILURE;
    }

    // 5. Run GPU ASCII filter
    runAsciiKernel(dev.in, dev.out, img.width, img.height, img.channels);

    // 6. Synchronize and copy result back to host
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(out_host.data(), dev.out, dev.bytes, cudaMemcpyDeviceToHost));

    // 7. Save output image (fixed filename)
    if (!saveFixedOutputImage(img, out_host)) {
        cleanupAll(img, dev);
        return EXIT_FAILURE;
    }

    // 8. Cleanup
    cleanupAll(img, dev);

    return EXIT_SUCCESS;
}
