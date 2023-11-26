#pragma once

#include "Utils.h"

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