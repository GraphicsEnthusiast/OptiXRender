#include "Renderer.h"
#include "LaunchParams.h"
#include "../cuda/Material.h"
#include "../cuda/Light.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// .cu文件编译出来的 ptx 机器码
extern "C" char embedded_ptx_code[];

// Record 是 SBT 里面维护的着色器记录项，用来和场景中的材质实例做映射绑定的。
// 由于 Record 对于字节对齐要求很高，所以用到了 __align__ 关键字；
// 每个 Record 都有一个 header 表示该着色器实例的句柄。
// 由于一个场景中只需要一个 RaygenRecord 和一个 MissRecord，
// 所以他们内部直接声明场景数据即可，即 void* data；而 HitRecord 可能会有若干个，
// 分别对应不同的物体或材质，所以他们内部需要声明一个 int objectID，
// 表示当前的 HitRecord 绑定的是几号物体/材质。
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
};

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
Renderer::Renderer(const Scene* scene) : scene(scene) {
    InitOptix();

    if (this->scene->lights.size() != 0) {
        CUDABuffer lightsBuffer;
		lightsBuffer.alloc_and_upload(this->scene->lights);
		launchParams.lights.lightsBuffer = (Light*)lightsBuffer.d_pointer();
		launchParams.lights.lightSize = this->scene->lights.size();
    }

    if (this->scene->mediums.size() != 0) {
        CUDABuffer mediumsBuffer;
		mediumsBuffer.alloc_and_upload(this->scene->mediums);
		launchParams.mediums.mediumsBuffer = (Medium*)mediumsBuffer.d_pointer();
		launchParams.mediums.mediumSize = this->scene->mediums.size();
    }

    std::cout << "creating optix context ..." << std::endl;
    CreateContext();

    std::cout << "setting up module ..." << std::endl;
    CreateModule();

    std::cout << "creating raygen programs ..." << std::endl;
    CreateRaygenPrograms();
    std::cout << "creating miss programs ..." << std::endl;
    CreateMissPrograms();
    std::cout << "creating hitgroup programs ..." << std::endl;
    CreateHitgroupPrograms();

    launchParams.traversable = BuildAccel();

    std::cout << "setting up optix pipeline ..." << std::endl;
    CreatePipeline();

    CreateTextures();

    std::cout << "building SBT ..." << std::endl;
    BuildSBT();

    if (scene->env) {
        launchParams.environment.hasEnv = true;
        launchParams.environment.height = scene->env->height;
        launchParams.environment.width = scene->env->width;
        launchParams.environment.envMap = scene->env->cuda_texture_hdr;
        launchParams.environment.envCache = scene->env->cuda_texture_cache;
    }

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "context, module, pipeline, etc, all set up ..." << std::endl;

    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "RTRT_Render fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
}

void Renderer::CreateTextures() {
    int numTextures = (int)scene->textures.size();
    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for (int textureID = 0; textureID < numTextures; textureID++) {
        Texture* texture = scene->textures[textureID];

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width = texture->resolution.x;
        int32_t height = texture->resolution.y;
        int32_t numComponents = texture->comp;
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t& pixelArray = textureArrays[textureID];
        CUDA_CHECK(MallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(Memcpy2DToArray(pixelArray,
            /* offset */0, 0,
            texture->pixel,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        textureObjects[textureID] = cuda_tex;
    }
}

OptixTraversableHandle Renderer::BuildAccel() {
    /*
    一个 TriangleMesh 就只能注册成一个物体 id，
    一个物体 id 就只能绑定一个 Shader Record。
    只有一个 Shader Record 意味着大家都使用相同的 shader 实例和传入参数，
    那么大家都只能用相同材质了。
    因此，我们必须将不同材质的物体分离为多个模型。
    */
    const int numMeshes = (int)scene->meshes.size();
    vertexBuffer.resize(numMeshes);
    normalBuffer.resize(numMeshes);
    texcoordBuffer.resize(numMeshes);
    indexBuffer.resize(numMeshes);

    OptixTraversableHandle asHandle{ 0 };

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(numMeshes);
    std::vector<CUdeviceptr> d_vertices(numMeshes);
    std::vector<CUdeviceptr> d_indices(numMeshes);
    std::vector<uint32_t> triangleInputFlags(numMeshes);

    for (int meshID = 0; meshID < numMeshes; meshID++) {
        // upload the model to the device: the builder
        TriangleMesh& mesh = *scene->meshes[meshID];
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        indexBuffer[meshID].alloc_and_upload(mesh.index);
        normalBuffer[meshID].alloc_and_upload(mesh.normal);
        texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
        ;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
    (optixContext,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,  // num_build_inputs
        &blasBufferSizes
    ));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
        /* stream */0,
        &accelOptions,
        triangleInput.data(),
        (int)numMeshes,
        tempBuffer.d_pointer(),
        tempBuffer.sizeInBytes,
        outputBuffer.d_pointer(),
        outputBuffer.sizeInBytes,
        &asHandle,
        &emitDesc,
        1
    ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
        /*stream:*/0,
        asHandle,
        asBuffer.d_pointer(),
        asBuffer.sizeInBytes,
        &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void Renderer::InitOptix() {
    std::cout << "initializing optix..." << std::endl;

    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        throw std::runtime_error("no CUDA capable devices found!");
    }
    std::cout << "found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
    std::cout << GDT_TERMINAL_GREEN
        << "successfully initialized optix... yay!"
        << GDT_TERMINAL_DEFAULT << std::endl;
}

static void ContextLog(unsigned int level, const char* tag, const char* message, void*) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
  example, only for the primary GPU device) */
void Renderer::CreateContext() {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "running on device: " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS) {
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
    }

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, ContextLog, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
  to use. in this simple example, we use a single module from a
  single .cu file, using a single embedded ptx string */
void Renderer::CreateModule() {
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    // .cu文件里定义的“optixLaunchParams”会被置为光追的最终输出。
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    // embedded_ptx_code（即前面我们编译出来的着色器机器码）被绑定在了Module身上。
    // 我们的着色器机器码以 Module 为媒介，成功和管线取得了绑定。
    const std::string ptxCode = embedded_ptx_code;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &sizeof_log,
        &module
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }
}

// 创建管线中的着色器实例。要声明各类着色器分别有几个，
// 对应的是哪个 module，各个着色器在 module 中着色器机器码的入口函数是什么，
// 最后靠 optixProgramGroupCreate 函数来得到一个定义完全的着色器实例。
void Renderer::CreateRaygenPrograms() {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }
}

void Renderer::CreateMissPrograms() {
    // we do a single ray gen program in this example:
    missPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;

    // ------------------------------------------------------------------
    // radiance rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[RADIANCE_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }

    // ------------------------------------------------------------------
    // shadow rays
    // ------------------------------------------------------------------
    pgDesc.miss.entryFunctionName = "__miss__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[SHADOW_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }
}

void Renderer::CreateHitgroupPrograms() {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.moduleAH = module;

    // -------------------------------------------------------
    // radiance rays
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[RADIANCE_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }

    // -------------------------------------------------------
    // shadow rays: technically we don't need this hit group,
    // since we just use the miss shader to check if we were not
    // in shadow
    // -------------------------------------------------------
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[SHADOW_RAY_TYPE]
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }
}

/*! assembles the full pipeline of all programs */
void Renderer::CreatePipeline() {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs) {
        programGroups.push_back(pg);
    }
    for (auto pg : hitgroupPGs) {
        programGroups.push_back(pg);
    }
    for (auto pg : missPGs) {
        programGroups.push_back(pg);
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    PING;
    PRINT(programGroups.size());
    OPTIX_CHECK(optixPipelineCreate(optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipeline
    ));
    if (sizeof_log > 1) {
        PRINT(log);
    }

    OPTIX_CHECK(optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipeline,
        /* [in] The direct stack size requirement for direct
           callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
           callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
           passed to trace. */
        1));
    if (sizeof_log > 1) {
        PRINT(log);
    }
}

/*! constructs the shader binding table */
void Renderer::BuildSBT() {
    // 创建 Kulla-Conty LUT
    std::vector<float> bsdf_avg(kLutResolution * kLutResolution),
        bsdf_albedo_avg(kLutResolution);
    ComputeKullaConty(bsdf_avg.data(), bsdf_albedo_avg.data());
    CUDABuffer bsdf_avg_buffer, bsdf_albedo_avg_buffer;
    bsdf_avg_buffer.alloc_and_upload(bsdf_avg);
    bsdf_albedo_avg_buffer.alloc_and_upload(bsdf_albedo_avg);

    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)scene->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++) {
        for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
            auto mesh = scene->meshes[meshID];

            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
            rec.data.in_medium = mesh->in_medium;
            rec.data.out_medium = mesh->out_medium;
            rec.data.material = mesh->material;
            if (rec.data.material.albedoTextureID != -1) {
                rec.data.material.albedo_texture = textureObjects[rec.data.material.albedoTextureID];
            }
            if (rec.data.material.roughnessTextureID != -1) {
                rec.data.material.roughness_texture = textureObjects[rec.data.material.roughnessTextureID];
            }
            if (rec.data.material.anisotropyTextureID != -1) {
                rec.data.material.anisotropy_texture = textureObjects[rec.data.material.anisotropyTextureID];
            }
            if (rec.data.material.specularTextureID != -1) {
                rec.data.material.specular_texture = textureObjects[rec.data.material.specularTextureID];
            }
            if (rec.data.material.metallicTextureID != -1) {
                rec.data.material.metallic_texture = textureObjects[rec.data.material.metallicTextureID];
            }
            if (rec.data.material.coat_roughness_uTextureID != -1) {
                rec.data.material.coat_roughness_u_texture = textureObjects[rec.data.material.coat_roughness_uTextureID];
            }
            if (rec.data.material.coat_roughness_vTextureID != -1) {
                rec.data.material.coat_roughness_v_texture = textureObjects[rec.data.material.coat_roughness_vTextureID];
            }
            if (rec.data.material.normalTextureID != -1) {
                rec.data.material.normal_texture = textureObjects[rec.data.material.normalTextureID];
            }
            rec.data.material.bsdf_avg_buffer = (float*)bsdf_avg_buffer.d_pointer();
            rec.data.material.albedo_avg_buffer = (float*)bsdf_albedo_avg_buffer.d_pointer();
            rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
            rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
            rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
            rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! render one frame */
void Renderer::Render() {
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) {
        return;
    }

    if (!progressive) {
        launchParams.frame.frameID = 0;
    }
    launchParamsBuffer.upload(&launchParams, 1);
    launchParams.frame.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline, stream,
        /*! parameters and SBT */
        launchParamsBuffer.d_pointer(),
        launchParamsBuffer.sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        launchParams.frame.size.x,
        launchParams.frame.size.y,
        1
    ));

    denoiserIntensity.resize(sizeof(float));

    OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
#if OPTIX_VERSION >= 70300
    if (denoiserIntensity.sizeInBytes != sizeof(float)) {
        denoiserIntensity.alloc(sizeof(float));
    }
#endif
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    if (progressive) {
        denoiserParams.blendFactor = 1.0f / (launchParams.frame.frameID);
    }
    else {
        denoiserParams.blendFactor = 0.0f;
    }

    // -------------------------------------------------------
    OptixImage2D inputLayer[3];
    inputLayer[0].data = fbColor.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[0].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[0].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[2].data = fbNormal.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[2].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[2].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[1].data = fbAlbedo.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[1].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[1].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    outputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (denoiserOn) {
        OPTIX_CHECK(optixDenoiserComputeIntensity
        (denoiser,
            /*stream*/0,
            &inputLayer[0],
            (CUdeviceptr)denoiserIntensity.d_pointer(),
            (CUdeviceptr)denoiserScratch.d_pointer(),
            denoiserScratch.size()));

#if OPTIX_VERSION >= 70300
        OptixDenoiserGuideLayer denoiserGuideLayer = {};
        denoiserGuideLayer.albedo = inputLayer[1];
        denoiserGuideLayer.normal = inputLayer[2];

        OptixDenoiserLayer denoiserLayer = {};
        denoiserLayer.input = inputLayer[0];
        denoiserLayer.output = outputLayer;

        OPTIX_CHECK(optixDenoiserInvoke(denoiser,
            /*stream*/0,
            &denoiserParams,
            denoiserState.d_pointer(),
            denoiserState.size(),
            &denoiserGuideLayer,
            &denoiserLayer, 1,
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            denoiserScratch.d_pointer(),
            denoiserScratch.size()));
#else
        OPTIX_CHECK(optixDenoiserInvoke(denoiser,
            /*stream*/0,
            &denoiserParams,
            denoiserState.d_pointer(),
            denoiserState.size(),
            &inputLayer[0], 2,
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            &outputLayer,
            denoiserScratch.d_pointer(),
            denoiserScratch.size()));
#endif
    }
    else {
        cudaMemcpy((void*)outputLayer.data, (void*)inputLayer[0].data,
            outputLayer.width * outputLayer.height * sizeof(float4),
            cudaMemcpyDeviceToDevice);
    }
    ComputeFinalPixelColors();

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void Renderer::SetCamera(const Camera& camera) {
    lastSetCamera = camera;
    // reset accumulation
    launchParams.frame.frameID = 0;
    launchParams.camera.medium = camera.medium;
    launchParams.camera.position = camera.from;
    launchParams.camera.direction = normalize(camera.at - camera.from);
    const float cosFovy = 0.66f;
    const float aspect
        = float(launchParams.frame.size.x)
        / float(launchParams.frame.size.y);
    launchParams.camera.horizontal
        = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
            camera.up));
    launchParams.camera.vertical
        = cosFovy * normalize(cross(launchParams.camera.horizontal,
            launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void Renderer::Resize(const vec2i& newSize) {
    if (denoiser) {
        OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    };

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiserOptions = {};
#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
#else
    denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiserReturnSizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y,
        &denoiserReturnSizes));

#if OPTIX_VERSION < 70100
    denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
        denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
    fbColor.resize(newSize.x * newSize.y * sizeof(float4));
    fbNormal.resize(newSize.x * newSize.y * sizeof(float4));
    fbAlbedo.resize(newSize.x * newSize.y * sizeof(float4));
    finalColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size = newSize;
    launchParams.frame.colorBuffer = (float4*)fbColor.d_pointer();
    launchParams.frame.normalBuffer = (float4*)fbNormal.d_pointer();
    launchParams.frame.albedoBuffer = (float4*)fbAlbedo.d_pointer();

    // and re-set the camera, since aspect may have changed
    SetCamera(lastSetCamera);

    // ------------------------------------------------------------------
    OPTIX_CHECK(optixDenoiserSetup(denoiser, 0,
        newSize.x, newSize.y,
        denoiserState.d_pointer(),
        denoiserState.size(),
        denoiserScratch.d_pointer(),
        denoiserScratch.size()));
}

/*! download the rendered color buffer */
void Renderer::DownloadPixels(uint32_t h_pixels[]) {
    finalColorBuffer.download(h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);
}