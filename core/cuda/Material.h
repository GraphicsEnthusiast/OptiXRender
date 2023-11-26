#pragma once

#include "Sampling.h"
#include "Utils.h"

extern "C" __device__ vec3f EvaluateDiffuse(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f local_L = ToLocal(isect.shadeNormal, world_L);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);

	return isect.material.albedo * M_1_PIf;
}

extern "C" __device__ vec3f SampleDiffuse(const Interaction& isect, const vec2f& r2,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	vec3f local_L = CosineSampleHemisphere(r2);
	world_L = ToWorld(isect.shadeNormal, local_L);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);

	return isect.material.albedo * M_1_PIf;
}