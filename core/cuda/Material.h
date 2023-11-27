#pragma once

#include "Sampling.h"
#include "Microfacet.h"
#include "gdt/random/random.h"

typedef gdt::LCG<16> Random;

//*************************************diffuse*************************************
__forceinline__ __device__ vec3f EvaluateDiffuse(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f local_L = ToLocal(isect.shadeNormal, world_L);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);

	return isect.material.albedo * M_1_PIf;
}

__forceinline__ __device__ vec3f SampleDiffuse(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	vec3f local_L = CosineSampleHemisphere(vec2f(random(), random()));
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

__forceinline__ __device__ vec3f SampleConductor(const Interaction& isect, Random& random,
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
	vec3f H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
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

//*************************************dielectric*************************************
__forceinline__ __device__ vec3f EvaluateDielectric(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
    float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	float etai_over_etat = isect.frontFace ? (1.0f / eta) : (eta);
	vec3f albedo = isect.material.albedo;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H;

	bool isReflect = dot(N, L) * dot(N, V) > 0.0f;
	if(isReflect){
        H = normalize(V + L);
		if(dot(N, H) < 0.0f) {
		    H = -H;
	    }
	}
	else{
		H = -normalize(etai_over_etat * V + L);
		if (dot(N, H) < 0.0f) {
			H = -H;
		}
	}

	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));

	float NdotV = abs(dot(N, V));
	float NdotL = abs(dot(N, L));

	float F = FresnelDielectric(V, H, etai_over_etat);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
	vec3f bsdf = 0.0f;
    if (isReflect) {
		float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
		pdf = F * Dv * dwh_dwi;

		bsdf = albedo * F * D * G / (4.0f * NdotV * NdotL);
	}
	else {
        float HdotV = dot(H, V);
		float HdotL = dot(H, L);
		float sqrtDenom = etai_over_etat * HdotV + HdotL;
		float factor = abs(HdotL * HdotV / (NdotL * NdotV));

		float dwh_dwi = abs(HdotL) / sqr(sqrtDenom);
		pdf = (1.0f - F) * Dv * dwh_dwi;

		bsdf = albedo * (1.0f - F) * D * G * factor / sqr(sqrtDenom);
	}

	return bsdf;
}

__forceinline__ __device__ vec3f SampleDielectric(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	float etai_over_etat = isect.frontFace ? (1.0f / eta) : (eta);
	vec3f albedo = isect.material.albedo;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
	H = ToWorld(H, N);
	if (dot(N, H) < 0.0f) {
		H = -H;
	}

    vec3f bsdf = 0.0f;
	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
    float F = FresnelDielectric(V, H, etai_over_etat);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
    if (random() < F) {
		world_L = reflect(-V, H);
	    vec3f L = world_L;

		if (dot(N, L) <= 0.0f) {
			return 0.0f;
		}

		float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
		pdf = F * Dv * dwh_dwi;

		float NdotV = abs(dot(N, V));
	    float NdotL = abs(dot(N, L));

	    float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);

		bsdf = albedo * F * D * G / (4.0f * NdotV * NdotL);
	}
	else {
		world_L = refract(-V, H, etai_over_etat);
		vec3f L = world_L;

		//折射不可能在同侧，舍去
		if (dot(N, L) * dot(N, V) >= 0.0f) {
			return 0.0f;
		}

		float NdotV = abs(dot(N, V));
	    float NdotL = abs(dot(N, L));

	    float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);

        float HdotV = dot(H, V);
		float HdotL = dot(H, L);
		float sqrtDenom = etai_over_etat * HdotV + HdotL;
		float factor = abs(HdotL * HdotV / (NdotL * NdotV));

		float dwh_dwi = abs(HdotL) / sqr(sqrtDenom);
		pdf = (1.0f - F) * Dv * dwh_dwi;

		bsdf = albedo * (1.0f - F) * D * G * factor / sqr(sqrtDenom);
	}

	return bsdf;
}
//*************************************dielectric*************************************