#pragma once

/*
 * Edge markers (8x8 glyphs) extracted from font8x8_basic.h
 * Order: "/-|\\"
 */

#ifdef __CUDACC__   // when compiling with nvcc
__device__ __constant__ unsigned char AsciiEdges[16][8] = {
#else                // fallback for non-CUDA compilation (if ever needed)
static const char AsciiEdges[4][8] = {
#endif
    /* '/' (0x2F) */
    { 0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01 },
    /* '-' (0x2D) */
    { 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00 },
    /* '|' (0x7C) */
    { 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18 },
    /* '\' (0x5C) */
    { 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80 }
};

/* Parallel array with the characters in the same order. */
static const char AsciiEdges_chars[5] = "/-|\\";
