#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

struct Camera {
    /*! camera position - *from* where we are looking */
    vec3f from;
    /*! which point we are looking *at* */
    vec3f at;
    /*! general up-vector */
    vec3f up;
};

class Renderer {
public:
    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    Renderer(const Model * model, const QuadLight & light);

    /*! render one frame */
    void Render();

    /*! resize frame buffer to given resolution */
    void Resize(const vec2i& newSize);

    /*! download the rendered color buffer */
    void DownloadPixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void SetCamera(const Camera& camera);

    bool denoiserOn = true;
    bool accumulate = true;

protected:
    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    void ComputeFinalPixelColors();

    /*! helper function that initializes optix and checks for errors */
    void InitOptix();

    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void CreateContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void CreateModule();

    /*! does all setup for the raygen program(s) we are going to use */
    void CreateRaygenPrograms();

    /*! does all setup for the miss program(s) we are going to use */
    void CreateMissPrograms();

    /*! does all setup for the hitgroup program(s) we are going to use */
    void CreateHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void CreatePipeline();

    /*! constructs the shader binding table */
    void BuildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle BuildAccel();

    /*! upload textures, and create cuda texture objects for them */
    void CreateTextures();

protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
public:
    LaunchParams launchParams;

protected:
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
    CUDABuffer fbColor;
    CUDABuffer fbNormal;
    CUDABuffer fbAlbedo;

    /*! output of the denoiser pass, in float4 */
    CUDABuffer denoisedBuffer;

    /* the actual final color buffer used for display, in rgba8 */
    CUDABuffer finalColorBuffer;

    OptixDenoiser denoiser = nullptr;
    CUDABuffer    denoiserScratch;
    CUDABuffer    denoiserState;
    CUDABuffer    denoiserIntensity;

    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    /*! the model we are going to trace rays against */
    const Model *model;

    /*! @{ one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;
    std::vector<CUDABuffer> normalBuffer;
    std::vector<CUDABuffer> texcoordBuffer;
    std::vector<CUDABuffer> indexBuffer;
    /*! @} */

    /*! @{ one texture object and pixel array per used texture */
    std::vector<cudaArray_t>         textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
    /*! @} */

    /*! the camera we are to render with. */
    Camera lastSetCamera;
};
