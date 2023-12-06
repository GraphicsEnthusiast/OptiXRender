#include "../include/Renderer.h"

inline __device__ float Clampf(float f) { 
    return min(1.0f, max(0.0f, f)); 
}

inline __device__ float4 Clamp(float4 f) {
  return make_float4(Clampf(f.x),
                     Clampf(f.y),
                     Clampf(f.z),
                     Clampf(f.w));
}

inline __device__ float4 GammaCorrection(float4 c) {
    return make_float4(pow(c.x, 1.0f / 2.2f), pow(c.y, 1.0f / 2.2f), pow(c.z, 1.0f / 2.2f), pow(c.w, 1.0f / 2.2f));
}

inline __device__ float4 ToneMapping(float4 c, float limit) {
    float luminance = 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
    float factor = 1.0f / (1.0f + luminance / limit);

    return make_float4(c.x * factor, c.y * factor, c.z * factor, c.z * factor);
}

/*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
__global__ void ComputeFinalPixelColorsKernel(uint32_t* finalColorBuffer, float4* denoisedBuffer, vec2i size) {
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= size.x) {
        return;
    }
    if (pixelY >= size.y) {
        return;
    }

    int pixelID = pixelX + size.x * pixelY;

    float4 f4 = denoisedBuffer[pixelID];
    f4 = ToneMapping(f4, 1.5f);
    f4 = GammaCorrection(f4);
    f4 = Clamp(f4);
    uint32_t rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) << 0;
    rgba |= (uint32_t)(f4.y * 255.9f) << 8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255 << 24;
    finalColorBuffer[pixelID] = rgba;
}

void Renderer::ComputeFinalPixelColors() {
    vec2i fbSize = launchParams.frame.size;
    vec2i blockSize = 32;
    vec2i numBlocks = divRoundUp(fbSize, blockSize);
    ComputeFinalPixelColorsKernel
        <<< dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y) >>>
        ((uint32_t*)finalColorBuffer.d_pointer(),
            (float4*)denoisedBuffer.d_pointer(),
            fbSize);
}

