#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Scene.h"

// 参考资料：https://puluo.top/optix_01/

class Renderer {
public:
    /*! constructor - performs all setup, including initializing
      optix, creates module, pipeline, programs, SBT, etc. */
    Renderer(const Scene* scene);

    /*! render one frame */
    void Render();

    /*! resize frame buffer to given resolution */
    void Resize(const vec2i& newSize);

    /*! download the rendered color buffer */
    void DownloadPixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void SetCamera(const Camera& camera);

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
    // optix 管线将运行在以下 cuda 设备上
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;

    // Context（上下文）：Context 是 OptiX 中的核心概念，代表了整个光线追踪应用程序的执行环境。
    // 在 Context 中，包含了所有的光线追踪所需的资源，例如场景描述、加速结构、着色器程序等。
    // Context 还负责管理 GPU 设备和线程的分配，以及光线追踪的整体控制和调度。
    OptixDeviceContext optixContext;

    // Pipeline（管线）：Pipeline 是光线追踪流程的抽象表示，它包含了一系列的光线追踪阶段，
    // 例如光线的发射、相交测试、着色等。开发人员可以通过配置 Pipeline 来定义光线追踪的流程和渲染效果，
    // 以及指定各个阶段使用的着色器程序。
    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    // Module（模块）：Module 是 OptiX 中用于组织着色器程序的概念，
    // 它包含了一个或多个程序化着色器。开发人员可以将着色器程序编译为 Module，
    // 并将其加载到 Context 中供 Pipeline 使用。Module 提供了灵活的方式来组织和管理着色器程序，
    // 使得开发人员可以方便地重用和配置着色器程序。
    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

    // OptixProgramGroup 包含了一个或多个着色器程序，
    // 这些程序可以包括最接近光线相交点的最近的光线/相交程序，
    // 以及用于计算光线的发射、光线的相交测试、材质着色等各种渲染阶段所需的计算逻辑。
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;

    // Closet Hit Shader 就是负责计算求交后着色的，所以它本质上就是硬件光追的 BSDF Shader。
    // 如果场景中有很多不同 BSDF 的材质实例，意味着管线中要绑很多个对应的 Closet Hit Shader。
    // 
    // 一个场景下有很多物体，有很多 Shader，不同的物体要用不同的 Shader，
    // 要给 Shader 传入不同的材质参数，而 SBT 就维护了他们之间的映射关系，
    // 并储存了要传入的材质参数，这样在跑光追时，管线直接可以通过求交得到的物体 id 以及 SBT，
    // 计算出要调用哪个 hit shader 去计算，用什么材质参数去计算 BSDF。
    // 
    // SBT 这个数据结构本身存了非常多的槽，每个槽里维护了一个叫 Shader Record 的东西，
    // 每一个 Shader Record 和场景中的一个物体绑定，用来记录这个物体用的是哪个 shader，
    // 以及要传入的参数 data。Record 有三种类型：RayGenerationRecord、HitGroupRecord（可以细分为 Intersection Shader / Any-Hit Shader / Closet-Hit Shader）、
    // MissRecord，分别映射对应的 RayGen、Hit、Miss 着色器的句柄（表示用的那个 shader 实例）。
    // optix 会根据物体编号直接计算出 Shader 在 SBT 中的下标（不需要手动绑定，映射是自动的，1号物体自动绑第一个 HitRecord，2号物体绑第二个，以此类推），
    // 从而找到对应要调用的 Shader。
    OptixShaderBindingTable sbt = {};

    /*
        1、RayGen Shader（负责从像素中发射光线，并将该像素的最终结果写回）
        2、Miss Shader（处理求交失败时的着色情况）
        3、Any Shader（用户定义某个交点是否需要抛弃，比如纯透明物体的求交就没有意义，就需要抛弃）
        4、Intersection Shader（定义非三角形形状的求交算法）。
        5、Closet Shader（处理最近求交点的着色情况）

                                           Intersection ---> Any Hit
                                                ^               |
                                                |               v                   --- Yes ---> Closet Hit
        RayGen Shader ---> TraceRay() ---> AccelerationStructureTraversal ---> Hit?
                                                                                    --- No ---> Miss

        RayGen Shader负责创建每个像素出射的光线，然后Optix底层就会开始进行最近点的求交，
        如果遇到非三角形形状物体就需要Intersection Shader来定义求交规则，
        每次求交成功了就把求交的点存进Any Hit。Any Hit中定义了用户是否认可这个最近点
        （有时候用户可能认为某个最近交点不合适，比如透明物体的求交就是无效求交，需要强行抛弃），
        如果认可该交点，就保存为当前的最近交点。当求交任务完成后检测一下是否有交点，
        如果没有则调用Miss Shader来渲染求交失败后的颜色；如果有交点则通过Closest Hit得到那个最近交点，
        并完成求交光线的颜色计算，给出下一次弹射的方向。注意此时的弹射本质上是发射了一根新光线，
        所以还是通过RayGen Shader来管理发射，然后Optix底层就会进行下一次求交，依次类推……
    */

public:
    bool denoiserOn = false;
    bool progressive = false;

    // 这个文件是定义了一个缓冲区结构体，管线会不断将光追结果写入这个结构体。
    LaunchParams launchParams;

protected:
    CUDABuffer launchParamsBuffer;
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
    CUDABuffer denoiserScratch;
    CUDABuffer denoiserState;
    CUDABuffer denoiserIntensity;

    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    /*! the model we are going to trace rays against */
    const Scene* scene;

    /*! @{ one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;
    std::vector<CUDABuffer> normalBuffer;
    std::vector<CUDABuffer> texcoordBuffer;
    std::vector<CUDABuffer> indexBuffer;
    /*! @} */

    /*! @{ one texture object and pixel array per used texture */
    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
    /*! @} */

    /*! the camera we are to render with. */
    Camera lastSetCamera;
};