#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Scene.h"

// �ο����ϣ�https://puluo.top/optix_01/

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
    // optix ���߽����������� cuda �豸��
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;

    // Context�������ģ���Context �� OptiX �еĺ��ĸ����������������׷��Ӧ�ó����ִ�л�����
    // �� Context �У����������еĹ���׷���������Դ�����糡�����������ٽṹ����ɫ������ȡ�
    // Context ��������� GPU �豸���̵߳ķ��䣬�Լ�����׷�ٵ�������ƺ͵��ȡ�
    OptixDeviceContext optixContext;

    // Pipeline�����ߣ���Pipeline �ǹ���׷�����̵ĳ����ʾ����������һϵ�еĹ���׷�ٽ׶Σ�
    // ������ߵķ��䡢�ཻ���ԡ���ɫ�ȡ�������Ա����ͨ������ Pipeline ���������׷�ٵ����̺���ȾЧ����
    // �Լ�ָ�������׶�ʹ�õ���ɫ������
    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    // Module��ģ�飩��Module �� OptiX ��������֯��ɫ������ĸ��
    // ��������һ������������ɫ����������Ա���Խ���ɫ���������Ϊ Module��
    // ��������ص� Context �й� Pipeline ʹ�á�Module �ṩ�����ķ�ʽ����֯�͹�����ɫ������
    // ʹ�ÿ�����Ա���Է�������ú�������ɫ������
    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

    // OptixProgramGroup ������һ��������ɫ������
    // ��Щ������԰�����ӽ������ཻ�������Ĺ���/�ཻ����
    // �Լ����ڼ�����ߵķ��䡢���ߵ��ཻ���ԡ�������ɫ�ȸ�����Ⱦ�׶�����ļ����߼���
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;

    // Closet Hit Shader ���Ǹ�������󽻺���ɫ�ģ������������Ͼ���Ӳ����׷�� BSDF Shader��
    // ����������кܶ಻ͬ BSDF �Ĳ���ʵ������ζ�Ź�����Ҫ��ܶ����Ӧ�� Closet Hit Shader��
    // 
    // һ���������кܶ����壬�кܶ� Shader����ͬ������Ҫ�ò�ͬ�� Shader��
    // Ҫ�� Shader ���벻ͬ�Ĳ��ʲ������� SBT ��ά��������֮���ӳ���ϵ��
    // ��������Ҫ����Ĳ��ʲ������������ܹ�׷ʱ������ֱ�ӿ���ͨ���󽻵õ������� id �Լ� SBT��
    // �����Ҫ�����ĸ� hit shader ȥ���㣬��ʲô���ʲ���ȥ���� BSDF��
    // 
    // SBT ������ݽṹ������˷ǳ���Ĳۣ�ÿ������ά����һ���� Shader Record �Ķ�����
    // ÿһ�� Shader Record �ͳ����е�һ������󶨣�������¼��������õ����ĸ� shader��
    // �Լ�Ҫ����Ĳ��� data��Record ���������ͣ�RayGenerationRecord��HitGroupRecord������ϸ��Ϊ Intersection Shader / Any-Hit Shader / Closet-Hit Shader����
    // MissRecord���ֱ�ӳ���Ӧ�� RayGen��Hit��Miss ��ɫ���ľ������ʾ�õ��Ǹ� shader ʵ������
    // optix �����������ֱ�Ӽ���� Shader �� SBT �е��±꣨����Ҫ�ֶ��󶨣�ӳ�����Զ��ģ�1�������Զ����һ�� HitRecord��2�������ڶ������Դ����ƣ���
    // �Ӷ��ҵ���ӦҪ���õ� Shader��
    OptixShaderBindingTable sbt = {};

    /*
        1��RayGen Shader������������з�����ߣ����������ص����ս��д�أ�
        2��Miss Shader��������ʧ��ʱ����ɫ�����
        3��Any Shader���û�����ĳ�������Ƿ���Ҫ���������紿͸��������󽻾�û�����壬����Ҫ������
        4��Intersection Shader���������������״�����㷨����
        5��Closet Shader����������󽻵����ɫ�����

                                           Intersection ---> Any Hit
                                                ^               |
                                                |               v                   --- Yes ---> Closet Hit
        RayGen Shader ---> TraceRay() ---> AccelerationStructureTraversal ---> Hit?
                                                                                    --- No ---> Miss

        RayGen Shader���𴴽�ÿ�����س���Ĺ��ߣ�Ȼ��Optix�ײ�ͻῪʼ�����������󽻣�
        �����������������״�������ҪIntersection Shader�������󽻹���
        ÿ���󽻳ɹ��˾Ͱ��󽻵ĵ���Any Hit��Any Hit�ж������û��Ƿ��Ͽ���������
        ����ʱ���û�������Ϊĳ��������㲻���ʣ�����͸��������󽻾�����Ч�󽻣���Ҫǿ����������
        ����Ͽɸý��㣬�ͱ���Ϊ��ǰ��������㡣����������ɺ���һ���Ƿ��н��㣬
        ���û�������Miss Shader����Ⱦ��ʧ�ܺ����ɫ������н�����ͨ��Closest Hit�õ��Ǹ�������㣬
        ������󽻹��ߵ���ɫ���㣬������һ�ε���ķ���ע���ʱ�ĵ��䱾�����Ƿ�����һ���¹��ߣ�
        ���Ի���ͨ��RayGen Shader�������䣬Ȼ��Optix�ײ�ͻ������һ���󽻣��������ơ���
    */

public:
    bool denoiserOn = false;
    bool progressive = false;

    // ����ļ��Ƕ�����һ���������ṹ�壬���߻᲻�Ͻ���׷���д������ṹ�塣
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