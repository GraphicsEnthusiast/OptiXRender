#pragma once

#include "Utils.h"

__forceinline__ __device__ float PowerHeuristic(float pdf1, float pdf2, int beta) {
	float p1 = pow(pdf1, beta);
	float p2 = pow(pdf2, beta);

	return p1 / (p1 + p2);
}

__forceinline__ __device__ vec3f UniformSampleCone(const vec2f& sample, float cos_angle) {
	vec3f p = 0.0f;

	float phi = M_2PIf * sample.x;
	float cos_theta = mix(cos_angle, 1.0f, sample.y);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cosf(phi);
	p.y = sin_theta * sinf(phi);
	p.z = cos_theta;

	return normalize(p);
}

__forceinline__ __device__ float UniformPdfCone(float cos_angle) {
	return 1.0f / (M_2PIf * (1.0f - cos_angle));
}

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