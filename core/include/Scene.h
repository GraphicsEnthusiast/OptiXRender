#pragma once

#include "gdt/math/AffineSpace.h"
#include "CUDABuffer.h"
#include <vector>
#include <cuda_runtime.h>

using namespace gdt;

//*************************************camera*************************************
struct Camera {
    /*! camera position - *from* where we are looking */
    vec3f from;
    /*! which point we are looking *at* */
    vec3f at;
    /*! general up-vector */
    vec3f up;
};
//*************************************camera*************************************

//*************************************material*************************************
enum MaterialType {
    Diffuse,
    Conductor,
    Dielectric,
    Plastic
};

struct Material {
    MaterialType type = MaterialType::Plastic;

    vec3f albedo = 0.8f;
    int albedoTextureID = -1;
    cudaTextureObject_t albedo_texture;

    float roughness = 0.1f;
    int roughnessTextureID = -1;
    cudaTextureObject_t roughness_texture;

    float anisotropy = 0.0f;
    int anisotropyTextureID = -1;
    cudaTextureObject_t anisotropy_texture;

    vec3f eta{ 0.14282f, 0.37414f, 1.43944f };
    vec3f k{ 3.97472f, 2.38066f, 1.59981f };

    float int_ior = 1.5f;
    float ext_ior = 1.0f;

    bool nonlinear = true;

    vec3f specular = 1.0f;
    int specularTextureID = -1;
    cudaTextureObject_t specular_texture;

    int normalTextureID = -1;
    cudaTextureObject_t normal_texture;

    float* bsdf_avg_buffer = NULL;
    float* albedo_avg_buffer = NULL;
};
//*************************************material*************************************

//*************************************triangle mesh*************************************
struct TriangleMesh {
    TriangleMesh() = default;

    void LoadMesh(const std::string& objFile, const Material& material);

    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    Material material;
};
//*************************************triangle mesh*************************************

//*************************************light*************************************
enum LightType {
    Quad,
    Sphere
};

struct Light {
    LightType type = LightType::Sphere;
    bool doubleSide = false;
    vec3f radiance = 3.0f;
    vec3f position{ 6.0f, 2.0f, 0.0f };

    vec3f u{ 0.0f, 0.0f, 6.0f };
    vec3f v{ 6.0f, 0.0f, 0.0f };

    float radius = 3.0f;
};
//*************************************light*************************************

//*************************************texture*************************************
struct TextureFile {
	std::string albedoFile = "";
	std::string roughnessFile = "";
	std::string anisotropyFile = "";
	std::string specularFile = "";
    std::string normalFile = "";
};

struct Texture {
    Texture(const std::string& fileName);
    ~Texture();

    uint8_t* pixel{ nullptr };
    vec2i resolution{ -1 };
    int comp{ 4 };
};

struct HdrTexture {
    HdrTexture(const std::string& fileName);
    ~HdrTexture();

    void CalculateHdrCache();
    void CreateCudaTexture();

    float* cache{ nullptr };
    float* hdr{ nullptr };
    int width, height, comp;

    cudaTextureObject_t cuda_texture_hdr, cuda_texture_cache;
};
//*************************************texture*************************************

//*************************************scene*************************************
class Scene {
public:
    Scene() = default;
    ~Scene();

    void AddMesh(const std::string& objFile, Material& material, const TextureFile& textureFile);
    void AddTexture(const std::string& fileName);
    void AddLight(const Light& l);
    void AddEnv(const std::string& fileName);

public:
    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*> textures;
    std::vector<Light> lights;
    HdrTexture* env = NULL;
    //! bounding box of all vertices in the scene
    box3f bounds;
};
//*************************************scene*************************************