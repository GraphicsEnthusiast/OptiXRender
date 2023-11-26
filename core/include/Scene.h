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
    DIFFUSE
};

struct Material {
    MaterialType type = MaterialType::DIFFUSE;

    vec3f albedo = 0.5f;
    int albedoTextureID = -1;
    cudaTextureObject_t albedo_texture;

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