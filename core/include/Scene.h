#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include <cuda_runtime.h>

using namespace gdt;

struct Camera {
	/*! camera position - *from* where we are looking */
	vec3f from;
	/*! which point we are looking *at* */
	vec3f at;
	/*! general up-vector */
	vec3f up;
};

enum MaterialType {
    Diffuse,
    Conductor,
    Dielectric,
    Plastic
};

struct Material {
    MaterialType type = MaterialType::Conductor;

    vec3f albedo = 1.0f;
    int albedoTextureID = -1;
    cudaTextureObject_t albedo_texture;

    float roughness = 0.9f;
    int roughnessTextureID = - 1;
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

    float* bsdf_avg_buffer = NULL;
    float* albedo_avg_buffer = NULL;

    vec3f emitter = 0.0f;
};
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
struct TriangleMesh {
    TriangleMesh(const std::string& objFile);

    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    Material material;
};

struct Light {
    vec3f origin, du, dv, power;
};
  
struct Texture {
    Texture();
    ~Texture();
    
    uint32_t* pixel { nullptr };
    vec2i resolution { -1 };
};
  
class Scene {
public:
    Scene() = default;
    ~Scene();

    void LoadMesh(const std::string& objFile);
    void AddLight(const Light& l);
    
public:
    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*> textures;
    std::vector<Light> lights;
    //! bounding box of all vertices in the scene
    box3f bounds;
};