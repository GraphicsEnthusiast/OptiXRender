#pragma once

#include "gdt/math/vec.h"
#include "../include/LaunchParams.h"
#include "math.h"

#define EPS 1e-4f
#define M_PIf 3.14159265358979323846f
#define M_PI_2f 1.57079632679489661923f
#define M_1_PIf	0.318309886183790671538f
#define M_2PIf 6.28318530717958647692f

__forceinline__ __device__ float sqr(float x) {
	return x * x;
}

__forceinline__ __device__ float mix(float x, float y, float t) {
	return x * (1.0f - t) + y * t;
}

__forceinline__ __device__ vec3f reflect(const vec3f& v, const vec3f& n) {
	return normalize(v - 2.0f * dot(v, n) * n);
}

__forceinline__ __device__ vec3f refract(const vec3f& v, const vec3f& n, float etai_over_etat) {
	float cos_theta = min(dot(-v, n), 1.0f);
	vec3f r_out_perp = etai_over_etat * (v + cos_theta * n);
	vec3f r_out_parallel = -sqrt(abs(1.0f - dot(r_out_perp, r_out_perp))) * n;

	return normalize(r_out_perp + r_out_parallel);
}

struct Ray {
	vec3f origin;
	vec3f direction;
};

struct Interaction {
	float distance;
	vec3f position;
	vec3f shadeNormal;
	bool frontFace;
	Material material;

	__forceinline__ __device__ void SetFaceNormal(const vec3f& dir, const vec3f& outward_normal) {
		frontFace = dot(dir, outward_normal) < 0.0f;
		shadeNormal = frontFace ? outward_normal : -outward_normal;
	}
};

__forceinline__ __device__ __host__ float Luminance(const vec3f& c) {
	return dot(vec3f(0.299f, 0.587f, 0.114f), c);
}


__forceinline__ __device__ bool IsNan(const vec3f& v) {
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

__forceinline__ __device__ bool IsValid(float value) {
	if (isnan(value) || value < 0.0f) {
		return false;
	}

	return true;
}

// 将单位向量从世界坐标系转换到局部坐标系
// dir 待转换的单位向量
// up 局部坐标系的竖直向上方向在世界坐标系下的方向
__forceinline__ __device__ vec3f ToLocal(const vec3f& dir, const vec3f& up) {
	auto B = vec3f(0.0f), C = vec3f(0.0f);
	if (abs(up.x) > abs(up.y)) {
		float len_inv = 1.0f / sqrt(up.x * up.x + up.z * up.z);
		C = vec3f(up.z * len_inv, 0.0f, -up.x * len_inv);
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
		C = vec3f(up.z * len_inv, 0.0f, -up.x * len_inv);
	}
	else {
		float len_inv = 1.0f / sqrt(up.y * up.y + up.z * up.z);
		C = vec3f(0.0f, up.z * len_inv, -up.y * len_inv);
	}
	B = cross(C, up);

	return normalize(dir.x * B + dir.y * C + dir.z * up);
}