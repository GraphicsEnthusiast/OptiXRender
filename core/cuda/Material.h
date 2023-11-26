#pragma once

#include "Sampling.h"
#include "Microfacet.h"

//*************************************diffuse*************************************
__forceinline__ __device__ vec3f EvaluateDiffuse(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f local_L = ToLocal(isect.shadeNormal, world_L);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);

	return isect.material.albedo * M_1_PIf;
}

__forceinline__ __device__ vec3f SampleDiffuse(const Interaction& isect, const vec2f& sample,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	vec3f local_L = CosineSampleHemisphere(sample);
	world_L = ToWorld(isect.shadeNormal, local_L);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);

	vec3f brdf = isect.material.albedo * M_1_PIf;

	return brdf;
}
//*************************************diffuse*************************************

//*************************************conductor*************************************
__forceinline__ __device__ vec3f EvaluateConductor(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
    float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	vec3f eta = isect.material.eta;
    vec3f k = isect.material.k;
	vec3f albedo = isect.material.albedo;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H = normalize(V + L);
	if(dot(N, H) < 0.0f) {
		H = -H;
	}

	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));

	float NdotV = dot(N, V);
	float NdotL = dot(N, L);

	if(NdotV <= 0.0f || NdotL <= 0.0f) {
		return 0.0f;
	}

	vec3f F = FresnelConductor(V, H, eta, k);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);

	vec3f brdf = albedo * F * D * G / (4.0f * NdotV * NdotL);

	return brdf;
}

__forceinline__ __device__ vec3f SampleConductor(const Interaction& isect, const vec2f& sample,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	vec3f eta = isect.material.eta;
    vec3f k = isect.material.k;
	vec3f albedo = isect.material.albedo;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f H = SampleVisibleGGX(N, V, alpha_u, alpha_v, sample);
	H = ToWorld(H, N);
	world_L = reflect(-V, H);
	vec3f L = world_L;

	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));

	float NdotV = dot(N, V);
	float NdotL = dot(N, L);

	if(NdotV <= 0.0f || NdotL <= 0.0f) {
		return 0.0f;
	}

	vec3f F = FresnelConductor(V, H, eta, k);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);

	vec3f brdf = albedo * F * D * G / (4.0f * NdotV * NdotL);

	return brdf;
}
//*************************************conductor*************************************