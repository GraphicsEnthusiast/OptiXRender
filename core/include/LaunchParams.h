#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

using namespace gdt;

  // for this simple example, we have a single ray type
enum { 
    RADIANCE_RAY_TYPE = 0, 
    SHADOW_RAY_TYPE, RAY_TYPE_COUNT 
};

struct TriangleMeshSBTData {
    vec3f  color;
    vec3f* vertex;
    vec3f* normal;
    vec2f* texcoord;
    vec3i* index;
    bool hasTexture;
    cudaTextureObject_t texture;
  };
  
struct LaunchParams {
    int numPixelSamples = 1;
    struct {
      int frameID = 0;
      float4* colorBuffer;
      float4* normalBuffer;
      float4* albedoBuffer;
      
      /*! the size of the frame buffer to render */
      vec2i size;
    } frame;
    
    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    struct {
      vec3f origin, du, dv, power;
    } light;
    
    OptixTraversableHandle traversable;
};

