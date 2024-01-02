#include <optix_device.h>
#include <cuda_runtime.h>

#include "Utils.h"
#include "Filter.h"
#include "Light.h"
#include "Material.h"
#include "Medium.h"
#include "Environment.h"

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

/*! per-ray data now captures random number generator, so programs
    can access RNG state */
struct PRD {
    Random random;
    bool lightVisible;
    Interaction isect;
    vec3f pixelAlbedo;
    vec3f pixelNormal;
    bool firstBounce;
};

/*
UnpackPointer 和 PackPointer 就是对某个指针中保存的地址进行高32位和低32位的拆分和合并。
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
}

extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    PRD& prd = *GetPRD<PRD>();

    const int primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const vec3f rayDir = optixGetWorldRayDirection();

    vec3f Ns = ((1.0f - u - v) * sbtData.normal[index.x]
        + u * sbtData.normal[index.y]
        + v * sbtData.normal[index.z]);
    Ns = normalize(Ns);

    const vec2f tc = (1.0f - u - v) * sbtData.texcoord[index.x]
        + u * sbtData.texcoord[index.y]
        + v * sbtData.texcoord[index.z];

    Material material = sbtData.material;
    if (material.albedoTextureID != -1) {
        material.albedo = (vec3f)tex2D<float4>(material.albedo_texture, tc.x, tc.y);
    }
    if (material.roughnessTextureID != -1) {
        material.roughness = tex2D<float4>(material.roughness_texture, tc.x, tc.y).x;
    }
    if (material.anisotropyTextureID != -1) {
        material.anisotropy = tex2D<float4>(material.anisotropy_texture, tc.x, tc.y).x;
    }
    if (material.specularTextureID != -1) {
        material.specular = (vec3f)tex2D<float4>(material.specular_texture, tc.x, tc.y);
    }
    if (material.metallicTextureID != -1) {
        material.metallic = tex2D<float4>(material.metallic_texture, tc.x, tc.y).x;
    }
    if (material.coat_roughness_uTextureID != -1) {
        material.coat_roughness_u = tex2D<float4>(material.coat_roughness_u_texture, tc.x, tc.y).x;
    }
    if (material.coat_roughness_vTextureID != -1) {
        material.coat_roughness_v = tex2D<float4>(material.coat_roughness_v_texture, tc.x, tc.y).x;
    }
    if (material.normalTextureID != -1) {
        vec3f t_normal = (vec3f)tex2D<float4>(material.normal_texture, tc.x, tc.y);
        Ns = NormalFromTangentToWorld(Ns, t_normal);
    }

    const vec3f surfPos = (1.0f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z];

    prd.isect.SetFaceNormal(rayDir, Ns);
    prd.isect.distance = length(surfPos - prd.isect.position);
    prd.isect.position = surfPos;
    prd.isect.material = material;
    prd.isect.mi = MediumInterface(sbtData.in_medium, sbtData.out_medium);

    if (prd.firstBounce) {
        prd.pixelNormal = Ns;
        prd.pixelAlbedo = material.albedo;
        prd.firstBounce = false;
    }
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
    const auto& camera = optixLaunchParams.camera;
    PRD& prd = *GetPRD<PRD>();
    prd.isect.distance = FLT_MAX;
    prd.isect.mi = MediumInterface(camera.medium);
    prd.isect.frontFace = true;
    if (prd.firstBounce) {
        prd.pixelAlbedo = 0.0f;
        prd.pixelNormal = 0.0f;
        prd.firstBounce = false;
    }
}

extern "C" __global__ void __miss__shadow() {
    // we didn't hit anything, so the light is visible
    PRD& prd = *GetPRD<PRD>();
    prd.lightVisible = true;
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto& camera = optixLaunchParams.camera;
    const auto& lights = optixLaunchParams.lights;
    const auto& mediums = optixLaunchParams.mediums;

    PRD prd;
    prd.random.init(ix + optixLaunchParams.frame.size.x * iy,
        optixLaunchParams.frame.frameID);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    int numPixelSamples = optixLaunchParams.numPixelSamples;

    vec3f pixelColor = 0.0f;
    vec3f pixelNormal = 0.0f;
    vec3f pixelAlbedo = 0.0f;

    auto lightTrace = [&lights](const Ray& ray, Interaction& isect, vec3f& light_radiance, float& light_pdf, float& closest_distance) -> bool {
        bool hitLight = false;
        for (int i = 0; i < lights.lightSize; i++) {
            const Light& light = lights.lightsBuffer[i];
            float light_distance = 0.0f;
            vec3f Li = EvaluateLight(light, ray, closest_distance, light_pdf, light_distance);

            // 没有击中光源，pdf = 0.0f
            if (light_pdf == 0.0f) {
                continue;
            }

            hitLight = true;
            light_radiance = Li;
            closest_distance = light_distance;
            isect.mi = MediumInterface(light.medium);
        }

        return hitLight;
    };

    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
        // normalized screen plane position, in [0,1]^2

        // iw: note for denoising that's not actually correct - if we
        // assume that the camera should only(!) cover the denoised
        // screen then the actual screen plane we shuld be using during
        // rendreing is slightly larger than [0,1]^2
        vec2f jitter = FilterJitter(vec2f(prd.random(), prd.random()), optixLaunchParams.filterType);
        vec2f screen(vec2f(ix + 0.5f + jitter.x, iy + 0.5f + jitter.y)
            / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        Ray ray;
        ray.origin = camera.position;
        ray.direction = rayDir;

        vec3f radiance(0.0f);
        vec3f history(1.0f);
        vec3f V = -ray.direction;
        vec3f L = ray.direction;
        float light_pdf = 0.0f;
        float bsdf_pdf = 0.0f;
        float phase_pdf = 0.0f;
        vec3f bsdf = 0.0f;
        vec3f attenuation = 0.0f;
        vec3f light_radiance = 0.0f;
        float closest_distance = FLT_MAX;
        vec3f pre_position = camera.position;
        float pre_pdf = 0.0f;
        float mult_trans_pdf = 1.0f;
        float trans_pdf = 1.0f;
        vec3f transmittance = 0.0f;

        prd.isect.distance = FLT_MAX;
        prd.isect.position = ray.origin;
        prd.firstBounce = true;
        prd.isect.mi = MediumInterface(camera.medium);

        for (int bounce = 0; bounce < optixLaunchParams.maxBounce; bounce++) {
            //*************************场景中的物体以及灯光求交*************************
            // 分别遍历场景中的物体以及光源，记录最近的那个
            optixTrace(optixLaunchParams.traversable,
                ray.origin,
                ray.direction,
                EPS,     // tmin
                FLT_MAX, // tmax
                0.0f,    // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,// OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);
            closest_distance = prd.isect.distance;
            bool hitLight = lightTrace(ray, prd.isect, light_radiance, light_pdf, closest_distance);
            //*************************场景中的物体以及灯光求交*************************

            bool scattered = false;
            int medium_index = -1;
            if (mediums.mediumSize != 0) {
                medium_index = prd.isect.mi.GetMedium(hitLight ? true : prd.isect.frontFace);
                float distance = 0.0f;

                // 处理介质
                if (medium_index != -1) {
                    const auto& medium = mediums.mediumsBuffer[medium_index];
                    scattered = SampleMediumDistance(medium, closest_distance, distance, trans_pdf, transmittance, prd.random);
                    history *= (transmittance / trans_pdf);
                    mult_trans_pdf *= trans_pdf;

                    if (scattered) {
                        prd.isect.position = pre_position + distance * L;
                        prd.isect.distance = distance;

                        // 采样直接光照
                        if (lights.lightSize != 0) {
                            prd.lightVisible = false;
                            // uniform sample one light
                            int index = clamp(int(lights.lightSize * prd.random()), 0, lights.lightSize - 1);
                            const Light& light = lights.lightsBuffer[index];
                            float light_distance = 0.0f;

                            Ray shadowRay;
                            shadowRay.origin = prd.isect.position;
                            light_radiance = SampleLight(light, shadowRay.origin, vec2f(prd.random(), prd.random()), shadowRay.direction, light_distance, light_pdf);
                            light_pdf *= (1.0f / (float)lights.lightSize);
                            light_radiance *= lights.lightSize;
                            optixTrace(optixLaunchParams.traversable,
                                shadowRay.origin,
                                shadowRay.direction,
                                EPS,                   // tmin
                                light_distance - EPS,  // tmax
                                0.0f,                  // rayTime
                                OptixVisibilityMask(255),
                                // For shadow rays: skip any/closest hit shaders and terminate on first
                                // intersection with anything. The miss shader is used to mark if the
                                // light was visible.
                                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                SHADOW_RAY_TYPE,            // SBT offset
                                RAY_TYPE_COUNT,             // SBT stride
                                SHADOW_RAY_TYPE,            // missSBTIndex 
                                u0, u1);

                            vec3f tv; float t; Interaction ti;
                            closest_distance = light_distance - EPS;
                            if (!lightTrace(shadowRay, ti, tv, t, closest_distance) && prd.lightVisible) {
                                transmittance = EvaluateMediumDistance(medium, false, closest_distance, trans_pdf);
                                attenuation = EvaluatePhase(medium, prd.isect, V, shadowRay.direction, phase_pdf);

                                float misWeight = PowerHeuristic(light_pdf, phase_pdf * trans_pdf, 2);

                                radiance += misWeight * (transmittance / trans_pdf) *  light_radiance * attenuation * history / light_pdf;
                            }
                        }

                        attenuation = SamplePhase(medium, prd.isect, prd.random, V, L, phase_pdf);
                        history *= (attenuation / phase_pdf);
                        pre_pdf = phase_pdf;
                    }
                }
            }

            if (!scattered) {
                // 处理光源
                if (hitLight) {
                    float misWeight = 1.0f;
                    if (bounce != 0) {
                        pre_pdf *= mult_trans_pdf;
                        misWeight = PowerHeuristic(pre_pdf, light_pdf, 2);
                    }
                    radiance += misWeight * history * light_radiance;

                    break;
                }

                // 处理环境
                bool notHit = closest_distance == FLT_MAX;
                if (notHit && medium_index == -1) {
                    if (optixLaunchParams.environment.hasEnv) {
                        int width = optixLaunchParams.environment.width;
                        int height = optixLaunchParams.environment.height;
                        vec2f uv = EvaluateEnvironment(width, height, light_pdf, ray.direction);
                        light_radiance = (vec3f)tex2D<float4>(optixLaunchParams.environment.envMap, uv.x, uv.y);
                        light_pdf *= tex2D<float4>(optixLaunchParams.environment.envCache, uv.x, uv.y).z;

                        float misWeight = 1.0f;
                        if (bounce != 0) {
                            pre_pdf *= mult_trans_pdf;
                            misWeight = PowerHeuristic(pre_pdf, light_pdf, 2);
                        }

                        radiance += misWeight * history * light_radiance;
                    }

                    break;
                }

                // 采样直接光照
                if (optixLaunchParams.environment.hasEnv) {
                    prd.lightVisible = false;

                    int width = optixLaunchParams.environment.width;
                    int height = optixLaunchParams.environment.height;
                    vec4f uv_t = tex2D<float4>(optixLaunchParams.environment.envCache, prd.random(), prd.random());
                    vec2f uv(uv_t.x, uv_t.y);

                    Ray shadowRay;
                    shadowRay.origin = prd.isect.position;
                    SampleEnvironment(width, height, light_pdf, shadowRay.direction, uv);
                    light_pdf *= tex2D<float4>(optixLaunchParams.environment.envCache, uv.x, uv.y).z;
                    light_radiance = (vec3f)tex2D<float4>(optixLaunchParams.environment.envMap, uv.x, uv.y);
                    optixTrace(optixLaunchParams.traversable,
                        shadowRay.origin,
                        shadowRay.direction,
                        EPS,      // tmin
                        FLT_MAX,  // tmax
                        0.0f,     // rayTime
                        OptixVisibilityMask(255),
                        // For shadow rays: skip any/closest hit shaders and terminate on first
                        // intersection with anything. The miss shader is used to mark if the
                        // light was visible.
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT
                        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        SHADOW_RAY_TYPE,            // SBT offset
                        RAY_TYPE_COUNT,             // SBT stride
                        SHADOW_RAY_TYPE,            // missSBTIndex 
                        u0, u1);

                    vec3f tv; float t; Interaction ti;
                    closest_distance = FLT_MAX;
                    if (!lightTrace(shadowRay, ti, tv, t, closest_distance) && prd.lightVisible) {
                        bsdf = EvaluateMaterial(prd.isect, V, shadowRay.direction, bsdf_pdf);
                        float costheta = abs(dot(prd.isect.shadeNormal, shadowRay.direction));
                        float misWeight = PowerHeuristic(light_pdf, bsdf_pdf, 2);

                        vec3f t_history = 1.0f;
                        if (medium_index != -1) {
                            const auto& medium = mediums.mediumsBuffer[medium_index];
                            transmittance = EvaluateMediumDistance(medium, false, closest_distance, trans_pdf);
                            t_history = (transmittance / trans_pdf);
                            misWeight = PowerHeuristic(light_pdf, bsdf_pdf * trans_pdf, 2);
                        }

                        radiance += misWeight * t_history * light_radiance * bsdf * costheta * history / light_pdf;
                    }
                }

                if (lights.lightSize != 0) {
                    prd.lightVisible = false;
                    // uniform sample one light
                    int index = clamp(int(lights.lightSize * prd.random()), 0, lights.lightSize - 1);
                    const Light& light = lights.lightsBuffer[index];
                    float light_distance = 0.0f;

                    Ray shadowRay;
                    shadowRay.origin = prd.isect.position;
                    light_radiance = SampleLight(light, shadowRay.origin, vec2f(prd.random(), prd.random()), shadowRay.direction, light_distance, light_pdf);
                    light_pdf *= (1.0f / (float)lights.lightSize);
                    light_radiance *= lights.lightSize;
                    optixTrace(optixLaunchParams.traversable,
                        shadowRay.origin,
                        shadowRay.direction,
                        EPS,                   // tmin
                        light_distance - EPS,  // tmax
                        0.0f,                  // rayTime
                        OptixVisibilityMask(255),
                        // For shadow rays: skip any/closest hit shaders and terminate on first
                        // intersection with anything. The miss shader is used to mark if the
                        // light was visible.
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT
                        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        SHADOW_RAY_TYPE,            // SBT offset
                        RAY_TYPE_COUNT,             // SBT stride
                        SHADOW_RAY_TYPE,            // missSBTIndex 
                        u0, u1);

                    vec3f tv; float t; Interaction ti;
                    closest_distance = light_distance - EPS;
                    if (!lightTrace(shadowRay, ti, tv, t, closest_distance) && prd.lightVisible) {
                        bsdf = EvaluateMaterial(prd.isect, V, shadowRay.direction, bsdf_pdf);
                        float costheta = abs(dot(prd.isect.shadeNormal, shadowRay.direction));
                        float misWeight = PowerHeuristic(light_pdf, bsdf_pdf, 2);
                        vec3f t_history = 1.0f;
                        if (medium_index != -1) {
                            const auto& medium = mediums.mediumsBuffer[medium_index];
                            transmittance = EvaluateMediumDistance(medium, false, closest_distance, trans_pdf);
                            t_history = (transmittance / trans_pdf);
                            misWeight = PowerHeuristic(light_pdf, bsdf_pdf * trans_pdf, 2);
                        }

                        radiance += misWeight * t_history * light_radiance * bsdf * costheta * history / light_pdf;
                    }
                }

                // 采样物体表面材质
                bsdf = SampleMaterial(prd.isect, prd.random, V, L, bsdf_pdf);
                float costheta = abs(dot(prd.isect.shadeNormal, L));
                history *= (bsdf * costheta / bsdf_pdf);
                pre_pdf = bsdf_pdf;
            }

            // 俄罗斯轮盘赌
            float prr = min((history.x + history.y + history.z) / 3.0f, 0.95f);
            if (prd.random() > prr) {
                break;
            }
            history /= prr;

            if (IsNan(history)) {
                break;
            }

            // 更新光线信息
            V = -L;
            ray.origin = prd.isect.position;
            ray.direction = L;
            mult_trans_pdf = 1.0f;
            pre_position = prd.isect.position;
        }

        if (IsNan(radiance)) {
            break;
        }

        // 这一步可以极大的减少白噪点（特别是由点光源产生）, 有偏
//        int lightNum = lights.lightSize;
//        if (optixLaunchParams.environment.hasEnv) {
//            lightNum++;
//        }
//        float illum = dot(radiance, vec3f(0.2126f, 0.7152f, 0.072f));
//        if (illum > lightNum) {
//            radiance *= lightNum / illum;
//        }

        pixelColor += radiance;
        pixelAlbedo += prd.pixelAlbedo;
        pixelNormal += prd.pixelNormal;
    }

    vec4f rgba(pixelColor / numPixelSamples, 1.0f);
    vec4f albedo(pixelAlbedo / numPixelSamples, 1.0f);
    vec4f normal(pixelNormal / numPixelSamples, 1.0f);

    // and write/accumulate to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    if (optixLaunchParams.frame.frameID > 0) {
        rgba += float(optixLaunchParams.frame.frameID)
            * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
        rgba /= (optixLaunchParams.frame.frameID + 1.0f);
    }
    optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = (float4)albedo;
    optixLaunchParams.frame.normalBuffer[fbIndex] = (float4)normal;
}