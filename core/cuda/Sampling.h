#pragma once

#include "Utils.h"

__forceinline__ __device__ float PowerHeuristic(float pdf1, float pdf2, int beta) {
	float p1 = pow(pdf1, beta);
	float p2 = pow(pdf2, beta);

	return p1 / (p1 + p2);
}

__forceinline__ __device__ vec3f PlaneToSphere(const vec2f& uv) {
	vec2f xy = uv;

	//获取角度
	float phi = M_2PIf * (xy.x - 0.5f);    // [-pi ~ pi]
	float theta = M_PIf * (xy.y - 0.5f);   // [-pi/2 ~ pi/2]   

	//球坐标计算方向
	vec3f L = normalize(vec3f(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi)));

	return L;
}

__forceinline__ __device__ vec2f SphereToPlane(const vec3f& v) {
	vec2f uv(atan2(v.z, v.x), asin(v.y));
	uv.x /= M_2PIf;
	uv.y /= M_PIf;
	uv += 0.5f;

	return uv;
}

//*************************************cone*************************************
__forceinline__ __device__ vec3f UniformSampleCone(const vec2f& sample, float cos_angle) {
	vec3f p = 0.0f;

	float phi = M_2PIf * sample.x;
	float cos_theta = mix(cos_angle, 1.0f, sample.y);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cos(phi);
	p.y = sin_theta * sin(phi);
	p.z = cos_theta;

	return normalize(p);
}

__forceinline__ __device__ float UniformPdfCone(float cos_angle) {
	return 1.0f / (M_2PIf * (1.0f - cos_angle));
}
//*************************************cone*************************************

//*************************************cosine*************************************
__forceinline__ __device__ vec3f CosineSampleHemisphere(const vec2f& sample) {
	vec3f p = 0.0f;

	// uniformly sample disk
	float r = sqrtf(sample.x);
	float phi = M_2PIf * sample.y;
	p.x = r * cos(phi);
	p.y = r * sin(phi);

	// project up to hemisphere
	p.z = sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y)); //cosTheta NoL

	return normalize(p);
}

__forceinline__ __device__ float CosinePdfHemisphere(float NdotL) {
	return NdotL * M_1_PIf;
}
//*************************************cosine*************************************

//*************************************sphere*************************************
__forceinline__ __device__ vec3f UniformSampleSphere(const vec2f& sample) {
	vec3f p = 0.0f;

	float phi = M_2PIf * sample.x;
	float cos_theta = 1.0f - 2.0f * sample.y;
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cos(phi);
	p.y = sin_theta * sin(phi);
	p.z = cos_theta;

	return normalize(p);
}

__forceinline__ __device__ float UniformPdfSphere() {
	return 1.0f / (4.0f * M_PIf);
}
//*************************************sphere*************************************

//*************************************hemisphere*************************************
__forceinline__ __device__ vec3f UniformSampleHemisphere(const vec2f& sample) {
	float r = sqrt(max(0.0f, 1.0f - sample.x * sample.x));
	float phi = M_2PIf * sample.y;
	vec3f p(r * cos(phi), r * sin(phi), sample.x);

	return normalize(p);
}

__forceinline__ __device__ float UniformPdfHemisphere() {
	return 1.0f / M_2PIf;
}
//*************************************hemisphere*************************************