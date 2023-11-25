#pragma once

#include "gdt/math/vec.h"
#include "../include/LaunchParams.h"
#include "math.h"

#define M_PIf 3.14159265358979323846f
#define M_PI_2f 1.57079632679489661923f
#define M_1_PIf	0.318309886183790671538f
#define M_2PIf 6.28318530717958647692f

struct Ray {
    vec3f origin;
    vec3f direction;
    float tmax = FLT_MAX;
};

__forceinline__ __device__ constexpr float Origin() { 
    return 1.0f / 32.0f; 
}

__forceinline__ __device__ constexpr float FloatScale() { 
    return 1.0f / 65536.0f; 
}

__forceinline__ __device__ constexpr float IntScale() { 
    return 256.0f; 
}

__forceinline__ __device__ vec3f OffsetRay(const vec3f p, const vec3f n) {
    int3 of_i = make_int3(IntScale() * n.x, IntScale() * n.y, IntScale() * n.z);

    vec3f p_i(
        float(int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        float(int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        float(int(p.z) + ((p.z < 0) ? - of_i.z : of_i.z)));

    return vec3f(
        fabsf(p.x) < Origin() ? p.x + FloatScale() * n.x : p_i.x,
        fabsf(p.y) < Origin() ? p.y + FloatScale() * n.y : p_i.y,
        fabsf(p.z) < Origin() ? p.z + FloatScale() * n.z : p_i.z);
}

struct Interaction {
    float bias = 0.001f;
    float distance;
    vec3f position;
    vec3f geomNormal;
    vec3f mat_color;
    __forceinline__ __device__ Ray SpawnRay(const vec3f& L) const {
        vec3f N = geomNormal;
        if (dot(L, geomNormal) < 0.0f) {
            N = -geomNormal;
        }

        Ray ray;
        ray.origin = OffsetRay(position, N);
        ray.direction = L;
        ray.tmax = FLT_MAX;

        return ray;
    }
};

// 将单位向量从世界坐标系转换到局部坐标系
// dir 待转换的单位向量
// up 局部坐标系的竖直向上方向在世界坐标系下的方向
__forceinline__ __device__ vec3f ToLocal(const vec3f& dir, const vec3f& up) {
	auto B = vec3f(0.0f), C = vec3f(0.0f);
	if (abs(up.x) > abs(up.y)) {
		float len_inv = 1.0f / sqrt(up.x * up.x + up.z * up.z);
		C = vec3f(up.z * len_inv, 0.0f, - up.x * len_inv);
	}
	else {
		float len_inv = 1.0f / sqrt(up.y * up.y + up.z * up.z);
		C = vec3f(0.0f, up.z * len_inv, -up.y * len_inv);
	}
	B = cross(C, up);

	return vec3f(dot(dir, B), dot(dir, C), dot(dir, up));
}

// 将单位向量从局部坐标系转换到世界坐标系
// dir 待转换的单位向量
// up 局部坐标系的竖直向上方向在世界坐标系下的方向
__forceinline__ __device__ vec3f ToWorld(const vec3f& dir, const vec3f& up) {
	auto B = vec3f(0.0f), C = vec3f(0.0f);
	if (abs(up.x) > abs(up.y)) {
		float len_inv = 1.0f / sqrt(up.x * up.x + up.z * up.z);
		C = vec3f(up.z * len_inv, 0.0f, - up.x * len_inv);
	}
	else {
		float len_inv = 1.0f / sqrt(up.y * up.y + up.z * up.z);
		C = vec3f(0.0f, up.z * len_inv, -up.y * len_inv);
	}
	B = cross(C, up);

	return normalize(dir.x * B + dir.y * C + dir.z * up);
}