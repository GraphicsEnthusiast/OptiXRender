#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>

using namespace gdt;

struct Camera {
	/*! camera position - *from* where we are looking */
	vec3f from;
	/*! which point we are looking *at* */
	vec3f at;
	/*! general up-vector */
	vec3f up;
};
  
  /*! a simple indexed triangle mesh that our sample renderer will
      render */
struct TriangleMesh {
    TriangleMesh(const std::string& objFile);

    std::vector<vec3f> vertex;
    std::vector<vec3f> normal;
    std::vector<vec2f> texcoord;
    std::vector<vec3i> index;

    // material data:
    vec3f diffuse;
    int diffuseTextureID { -1 };
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
    inline void AddLight(const Light& l) {
        lights.emplace_back(l);
    }
    
public:
    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*> textures;
    std::vector<Light> lights;
    //! bounding box of all vertices in the scene
    box3f bounds;
};