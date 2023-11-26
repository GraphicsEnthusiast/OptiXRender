#pragma once

#include "Utils.h"

__forceinline__ __device__ vec3f CosineSampleHemisphere(const vec2f& r2) {
	vec3f p = 0.0f;

	// uniformly sample disk
	float r = sqrtf(r2.x);
	float phi = M_2PIf * r2.y;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// project up to hemisphere
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y)); //cosTheta NoL

	return normalize(p);
}

__forceinline__ __device__ float CosinePdfHemisphere(float NdotL) {
	return NdotL * M_1_PIf;
}