#pragma once

#include "Sampling.h"
#include "Microfacet.h"

//*************************************diffuse*************************************
__forceinline__ __device__ vec3f EvaluateDiffuse(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f local_L = ToLocal(world_L, isect.shadeNormal);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);
	if(NdotL <= 0.0f) {
		return 0.0f;
	}

	vec3f brdf = isect.material.albedo * M_1_PIf;

	return brdf;
}

__forceinline__ __device__ vec3f SampleDiffuse(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	vec3f local_L = CosineSampleHemisphere(vec2f(random(), random()));
	world_L = ToWorld(local_L, isect.shadeNormal);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);
	if(NdotL <= 0.0f) {
		return 0.0f;
	}

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
	vec3f F_avg = AverageFresnelConductor(eta, k);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H = normalize(V + L);

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

	vec3f brdf = albedo * (F * D * G / (4.0f * NdotV * NdotL) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg));

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
	vec3f F_avg = AverageFresnelConductor(eta, k);

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

	vec3f brdf = albedo * (F * D * G / (4.0f * NdotV * NdotL) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg));

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
	float F_avg_ = AverageFresnelDielectric(eta);
	float F_avg_inv_ = AverageFresnelDielectric(1.0f / eta);
	float F_avg = isect.frontFace ? (F_avg_inv_) : (F_avg_);
	float F_avg_inv = !isect.frontFace ? (F_avg_inv_) : (F_avg_);
	float ratio_trans = ((1.0f - F_avg) * (1.0f - F_avg_inv) * sqr(etai_over_etat) / 
	    ((1.0f - F_avg) + (1.0f - F_avg_inv) * sqr(etai_over_etat)));

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H;

	bool isReflect = dot(N, L) * dot(N, V) > 0.0f;
	if(isReflect){
        H = normalize(V + L);
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
	vec3f bsdf_mult = EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg);
    if (isReflect) {
		float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
		pdf = F * Dv * dwh_dwi;

		bsdf = albedo * (F * D * G / (4.0f * NdotV * NdotL) + (1.0f - ratio_trans) * bsdf_mult);
	}
	else {
        float HdotV = dot(H, V);
		float HdotL = dot(H, L);
		float sqrtDenom = etai_over_etat * HdotV + HdotL;
		float factor = abs(HdotL * HdotV / (NdotL * NdotV));

		float dwh_dwi = abs(HdotL) / sqr(sqrtDenom);
		pdf = (1.0f - F) * Dv * dwh_dwi;

		bsdf = albedo * ((1.0f - F) * D * G * factor / sqr(sqrtDenom) + ratio_trans * bsdf_mult);
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
	float F_avg_ = AverageFresnelDielectric(eta);
	float F_avg_inv_ = AverageFresnelDielectric(1.0f / eta);
	float F_avg = isect.frontFace ? (F_avg_inv_) : (F_avg_);
	float F_avg_inv = !isect.frontFace ? (F_avg_inv_) : (F_avg_);
	float ratio_trans = ((1.0f - F_avg) * (1.0f - F_avg_inv) * sqr(etai_over_etat) / 
	    ((1.0f - F_avg) + (1.0f - F_avg_inv) * sqr(etai_over_etat)));

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
	H = ToWorld(H, N);

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
		vec3f bsdf_mult = EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg);

		bsdf = albedo * (F * D * G / (4.0f * NdotV * NdotL) + (1.0f - ratio_trans) * bsdf_mult);
	}
	else {
		world_L = refract(-V, H, etai_over_etat);
		vec3f L = world_L;

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
		vec3f bsdf_mult = EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg);

		bsdf = albedo * ((1.0f - F) * D * G * factor / sqr(sqrtDenom) + ratio_trans * bsdf_mult);
	}

	return bsdf;
}
//*************************************dielectric*************************************

//*************************************plastic*************************************
__forceinline__ __device__ vec3f EvaluatePlastic(const Interaction& isect, 
    const vec3f& world_V, const vec3f& world_L, float& pdf) {
    const vec3f kd = isect.material.albedo,
		ks = isect.material.specular;
	const float d_sum = kd.x + kd.y + kd.z,
		s_sum = ks.x + ks.y + ks.z;

	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	bool nonlinear = isect.material.nonlinear;
	float F_avg = AverageFresnelDielectric(eta);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H = normalize(V + L);

	float NdotV = dot(N, V);
	float NdotL = dot(N, L);
	if(NdotV <= 0.0f || NdotL <= 0.0f) {
		return 0.0f;
	}
    
	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	float Fo = FresnelDielectric(V, N, 1.0f / eta),
		Fi = FresnelDielectric(L, N, 1.0f / eta),
		specular_sampling_weight = s_sum / (s_sum + d_sum),
		pdf_specular = Fi * specular_sampling_weight,
		pdf_diffuse = (1.0f - Fi) * (1.0f - specular_sampling_weight);
	pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
	
	vec3f F = FresnelDielectric(L, H, 1.0f / eta);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);

    vec3f brdf = 0.0f;
	vec3f diffuse = kd, specular = ks;
	if (nonlinear) {
		brdf = diffuse / (1.0f - diffuse * F_avg);
	}
	else {
		brdf = diffuse / (1.0f - F_avg);
	}
	brdf *= (1.0f - Fi) * (1.0f - Fo) * M_1_PIf;
	brdf += specular * (F * D * G / (4.0f * NdotL * NdotV) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, vec3f(F_avg)));
	
	pdf = pdf_specular * Dv * abs(1.0f / (4.0f * dot(V, H))) + (1.0f - pdf_specular) * CosinePdfHemisphere(NdotL);

	return brdf;
}

__forceinline__ __device__ vec3f SamplePlastic(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
    const vec3f kd = isect.material.albedo,
		ks = isect.material.specular;
	const float d_sum = kd.x + kd.y + kd.z,
		s_sum = ks.x + ks.y + ks.z;

	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	bool nonlinear = isect.material.nonlinear;
	float F_avg = AverageFresnelDielectric(eta);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;

	float NdotV = dot(N, V);
	if(NdotV <= 0.0f) {
		return 0.0f;
	}
    
	float Fo = FresnelDielectric(V, N, 1.0f / eta),
		Fi = Fo,
		specular_sampling_weight = s_sum / (s_sum + d_sum),
		pdf_specular = Fi * specular_sampling_weight,
		pdf_diffuse = (1.0f - Fi) * (1.0f - specular_sampling_weight);
	pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

	vec3f brdf = 0.0f;
	vec3f L = 0.0f;
	vec3f H = 0.0f;
	float NdotL = 0.0f;
	if (random() < pdf_specular) {
        H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
	    H = ToWorld(H, N);

		world_L = reflect(-V, H);
		L = world_L;

		NdotL = dot(N, L);
		if (NdotL <= 0.0f) {
			return 0.0f;
		}
	}
	else {
		vec3f local_L = CosineSampleHemisphere(vec2f(random(), random()));
	    world_L = ToWorld(local_L, N);
		L = world_L;
		H = normalize(V + L);
		Fi = FresnelDielectric(L, N, 1.0f / eta);

		NdotL = dot(N, L);
		if (NdotL <= 0.0f) {
			return 0.0f;
		}
	}
    float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
	vec3f F = FresnelDielectric(L, H, 1.0f / eta);

	vec3f diffuse = kd, specular = ks;
	if (nonlinear) {
		brdf = diffuse / (1.0f - diffuse * F_avg);
	}
	else {
		brdf = diffuse / (1.0f - F_avg);
	}
	brdf *= (1.0f - Fi) * (1.0f - Fo) * M_1_PIf;
	brdf += specular * (F * D * G / (4.0f * NdotL * NdotV) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, vec3f(F_avg)));

	pdf = pdf_specular * Dv * abs(1.0f / (4.0f * dot(V, H))) + (1.0f - pdf_specular) * CosinePdfHemisphere(NdotL);

	return brdf;
}
//*************************************plastic*************************************