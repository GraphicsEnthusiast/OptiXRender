#include <optix_device.h>
#include <cuda_runtime.h>

#include "Utils.h"
#include "gdt/random/random.h"

#define NUM_LIGHT_SAMPLES 4

typedef gdt::LCG<16> Random;

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

/*! per-ray data now captures random number generator, so programs
    can access RNG state */
struct PRD {
    Random random;
    vec3f pixelColor;
    vec3f pixelNormal;
    vec3f pixelAlbedo;
};

/* 
unpackPainter 和 packPainter 就是对某个指针中保存的地址进行高32位和低32位的拆分和合并。
之所以需要拆分，是因为我们的计算机是64位的所以指针也是64位的，然而gpu的寄存器是32位的，
因此只能将指针拆分成两部分存进gpu的寄存器。
这里简单解释下 payload，payload 就类似一个负载寄存器，负责在不同 shader 之间传递信息。
后面我们会在 Raygen Shader 中申请一个颜色指针来存储最终的光追颜色，当开始 tracing 后，
会将这个颜色指针拆分（pack）写入0号和1号寄存器，当 Hit Shader 和 Miss Shader 想往里面写东西时，
就可以通过上面代码中的 GetPRD 函数获得0号和1号寄存器中的值，将其 unpack 便得到了那个颜色指针，
然后就可以往这个颜色指针里写内容了。
*/
static __forceinline__ __device__ void* UnPackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);

    return ptr;
}

static __forceinline__ __device__ void PackPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* GetPRD() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();

    return reinterpret_cast<T*>(UnPackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__shadow() {
    /* not going to be used ... */
}

extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData & sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
    PRD& prd = *GetPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f& A = sbtData.vertex[index.x];
    const vec3f& B = sbtData.vertex[index.y];
    const vec3f& C = sbtData.vertex[index.z];
    vec3f Ng = cross(B - A, C - A);
    vec3f Ns = (sbtData.normal)
        ? ((1.0f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
        : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.0f) {
        Ng = -Ng;
    }
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.0f) {
        // 如果着色法线Ns的方向与几何法线Ng的方向相反（即夹角小于0），
        // 则需要对Ns进行修正。这里使用了几何法线Ng来计算着色法线Ns的反射方向。
        // 这是根据反射定律来实现的，即入射角等于反射角。
        // 这个公式实际上是将着色法线Ns沿着几何法线Ng的方向进行反射，
        // 并将结果赋给Ns，以确保光线与表面相交时，着色法线的方向是正确的。
        Ns -= 2.0f * dot(Ng, Ns) * Ng;
    }
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
        const vec2f tc
            = (1.0f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= (vec3f)fromTexture;
    }

    // start with some ambient term
    vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const vec3f surfPos
        = (1.0f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z];

    prd.pixelNormal = Ns;
    prd.pixelAlbedo = diffuseColor;
    prd.pixelColor = (1.0f + Ns) * 0.5f;
}

extern "C" __global__ void __anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow() { /*! not going to be used */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    const vec3f rayDir = optixGetWorldRayDirection();
    vec3f unit_direction = normalize(rayDir);
    float t = 0.5f * (unit_direction.y + 1.0f);

    PRD& prd = *GetPRD<PRD>();
    // background color
    prd.pixelColor = (1.0f - t) * vec3f(1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
}

extern "C" __global__ void __miss__shadow() {
    // we didn't hit anything, so the light is visible
    vec3f& prd = *(vec3f*)GetPRD<vec3f>();
    prd = vec3f(1.0f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto& camera = optixLaunchParams.camera;

    PRD prd;
    prd.random.init(ix + optixLaunchParams.frame.size.x * iy,
        optixLaunchParams.frame.frameID);
    prd.pixelColor = vec3f(0.0f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    vec3f pixelColor = 0.0f;
    vec3f pixelNormal = 0.0f;
    vec3f pixelAlbedo = 0.0f;
    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
        // normalized screen plane position, in [0,1]^2

        // iw: note for denoising that's not actually correct - if we
        // assume that the camera should only(!) cover the denoised
        // screen then the actual screen plane we shuld be using during
        // rendreing is slightly larger than [0,1]^2
        vec2f screen(vec2f(ix + prd.random(), iy + prd.random())
            / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        Ray ray;
        ray.origin = camera.position;
        ray.direction = rayDir;

        optixTrace(optixLaunchParams.traversable,
            ray.origin,
            ray.direction,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
            u0, u1);
        pixelColor += prd.pixelColor;
        pixelNormal += prd.pixelNormal;
        pixelAlbedo += prd.pixelAlbedo;
    }

    vec4f rgba(pixelColor / numPixelSamples, 1.0f);
    vec4f albedo(pixelAlbedo / numPixelSamples, 1.0f);
    vec4f normal(pixelNormal / numPixelSamples, 1.0f);

    // and write/accumulate to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    if (optixLaunchParams.frame.frameID > 0) {
        rgba
            += float(optixLaunchParams.frame.frameID)
            * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
        rgba /= (optixLaunchParams.frame.frameID + 1.0f);
    }
    optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
    optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
}

