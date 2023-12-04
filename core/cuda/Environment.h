#pragma once

#include "Sampling.h"

__forceinline__ __device__ vec2f EvaluateEnvironment(int width, int height, float& pdf, const vec3f& world_L) {
	vec2f uv = SphereToPlane(world_L);

	pdf = float(width * height) / (2.0f * sqr(M_PIf));

	return uv;
}

__forceinline__ __device__ void SampleEnvironment(int width, int height, float& pdf, vec3f& world_L, const vec2f& uv) {
	world_L = PlaneToSphere(uv);

	pdf = float(width * height) / (2.0f * sqr(M_PIf));
}