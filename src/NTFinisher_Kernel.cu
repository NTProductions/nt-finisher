#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"

#	if GF_DEVICE_TARGET_DEVICE
GF_KERNEL_FUNCTION(NTGlowKernel,
    ((const GF_PTR(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(threshold))
    ((float)(amount))
    ((float)(radius)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
        float luminance = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;

        float4 glowColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float totalWeight = 0.0f;

        if (luminance > threshold / 100.0f) {
            int radiusInt = static_cast<int>(radius);
            int step = max(1, radiusInt / 10); // Reduce the number of samples

            for (int dy = -radiusInt; dy <= radiusInt; dy += step) {
                for (int dx = -radiusInt; dx <= radiusInt; dx += step) {
                    float dist = sqrtf(dx * dx + dy * dy);
                    if (dist <= radius) {
                        uint2 samplePos = make_uint2(inXY.x + dx, inXY.y + dy);
                        if (samplePos.x < inWidth && samplePos.y < inHeight) {
                            float4 samplePixel = ReadFloat4(inSrc, samplePos.y * inSrcPitch + samplePos.x, !!in16f);
                            float weight = (radius - dist) / radius;
                            glowColor.x += samplePixel.x * weight;
                            glowColor.y += samplePixel.y * weight;
                            glowColor.z += samplePixel.z * weight;
                            glowColor.w += samplePixel.w * weight;
                            totalWeight += weight;
                        }
                    }
                }
            }

            if (totalWeight > 0.0f) {
                float invTotalWeight = 1.0f / totalWeight;
                glowColor.x *= invTotalWeight * amount / 100.0f;
                glowColor.y *= invTotalWeight * amount / 100.0f;
                glowColor.z *= invTotalWeight * amount / 100.0f;
                glowColor.w *= invTotalWeight * amount / 100.0f;
            }
        }

        float4 result = make_float4(
            pixel.x + glowColor.x,
            pixel.y + glowColor.y,
            pixel.z + glowColor.z,
            pixel.w + glowColor.w);

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}

#	endif

#	if __NVCC__

void Exposure_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float threshold,
	float amount,
	float radius)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	NTGlowKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, threshold, amount, radius);

	cudaDeviceSynchronize();
}

#	endif //GF_DEVICE_TARGET_HOST

#endif
