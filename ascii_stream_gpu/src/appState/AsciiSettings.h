#pragma once

#include <cstdint>

// Settings for the ASCII effect (UI state / configuration).
// Stored in BGR (0..255) to match BGRA8 pipeline logic.
// UI can present RGB and convert when writing to these fields.
struct AsciiSettings
{
    bool enabled = true;

    // Monochrome tint in BGR (0..255). Alpha exists in BGRA8 but is not used here.
    uint8_t tintB = 230; // ~0.9 * 255
    uint8_t tintG = 230; // ~0.9 * 255
    uint8_t tintR = 128; // ~0.5 * 255

    // edgeThreshold: threshold for mean Sobel magnitude (compared against 255 * edgeThreshold).
    float edgeThreshold = 0.2f;
    // coherenceThreshold: direction coherence threshold (0..1).
    float coherenceThreshold = 0.5f;
};
