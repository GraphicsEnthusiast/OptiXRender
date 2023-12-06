#pragma once

#include "Sampling.h"
#include "Microfacet.h"

//*************************************diffuse*************************************
__forceinline__ __device__ vec3f EvaluateDiffuse(const Interaction& isect,
	const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f local_L = ToLocal(world_L, isect.shadeNormal);
	float NdotL = local_L.z;
	pdf = CosinePdfHemisphere(NdotL);
	if (NdotL <= 0.0f) {
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
	if (NdotL <= 0.0f) {
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

	if (NdotV <= 0.0f || NdotL <= 0.0f) {
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

	if (NdotV <= 0.0f || NdotL <= 0.0f) {
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

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H;

	bool isReflect = dot(N, L) * dot(N, V) > 0.0f;
	if (isReflect) {
		H = normalize(V + L);
	}
	else {
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
		bsdf *= sqr(etai_over_etat);
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
		bsdf *= sqr(etai_over_etat);
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
	if (NdotV <= 0.0f || NdotL <= 0.0f) {
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
	brdf += specular * (F * D * G / (4.0f * NdotL * NdotV) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg));

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
	if (NdotV <= 0.0f) {
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
	brdf += specular * (F * D * G / (4.0f * NdotL * NdotV) + EvaluateMultipleScatter(isect, NdotL, NdotV, roughness, F_avg));

	pdf = pdf_specular * Dv * abs(1.0f / (4.0f * dot(V, H))) + (1.0f - pdf_specular) * CosinePdfHemisphere(NdotL);

	return brdf;
}
//*************************************plastic*************************************

//*************************************metal workflow*************************************
__forceinline__ __device__ vec3f EvaluateMetalWorkflow(const Interaction& isect,
	const vec3f& world_V, const vec3f& world_L, float& pdf) {
	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	float metallic = isect.material.metallic;
	vec3f albedo = isect.material.albedo;
	bool nonlinear = isect.material.nonlinear;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H = normalize(L + V);

	float NdotV = dot(N, V);
	float NdotL = dot(N, L);
	if (NdotV <= 0.0f || NdotL <= 0.0f) {
		return 0.0f;
	}

	float metallic_brdf = metallic;
	float dieletric_brdf = (1.0f - metallic);
	float diffuse = dieletric_brdf;
	float specular = metallic_brdf + dieletric_brdf;
	float deom = diffuse + specular;
	float p_diffuse = diffuse / deom;

	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
	vec3f F0 = mix(vec3f(0.04f), albedo, metallic);
	vec3f F = FresnelSchlick(F0, dot(V, H));

	vec3f specular_brdf = D * F * G / (4.0f * NdotL * NdotV);
	vec3f diffuse_brdf = albedo * M_1_PIf;
	vec3f brdf = p_diffuse * diffuse_brdf + (1.0f - p_diffuse) * specular_brdf;

	pdf = (1.0f - p_diffuse) * Dv * abs(1.0f / (4.0f * dot(V, H))) + p_diffuse * CosinePdfHemisphere(NdotL);

	return brdf;
}

__forceinline__ __device__ vec3f SampleMetalWorkflow(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	float roughness = isect.material.roughness;
	float aniso = isect.material.anisotropy;
	float alpha_u = sqr(roughness) * (1.0f + aniso);
	float alpha_v = sqr(roughness) * (1.0f - aniso);
	float eta = isect.material.int_ior / isect.material.ext_ior;
	float metallic = isect.material.metallic;
	vec3f albedo = isect.material.albedo;
	bool nonlinear = isect.material.nonlinear;

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;

	float NdotV = dot(N, V);
	if (NdotV <= 0.0f) {
		return 0.0f;
	}

	float metallic_brdf = metallic;
	float dieletric_brdf = (1.0f - metallic);
	float diffuse = dieletric_brdf;
	float specular = metallic_brdf + dieletric_brdf;
	float deom = diffuse + specular;
	float p_diffuse = diffuse / deom;

	vec3f L = 0.0f;
	vec3f H = 0.0f;
	float NdotL = 0.0f;
	if (random() < p_diffuse) {
		vec3f local_L = CosineSampleHemisphere(vec2f(random(), random()));
		world_L = ToWorld(local_L, N);
		L = world_L;
		H = normalize(V + L);

		NdotL = dot(N, L);
		if (NdotL <= 0.0f) {
			return 0.0f;
		}
	}
	else {
		H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
		H = ToWorld(H, N);

		world_L = reflect(-V, H);
		L = world_L;

		NdotL = dot(N, L);
		if (NdotL <= 0.0f) {
			return 0.0f;
		}
	}
	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);
	vec3f F0 = mix(vec3f(0.04f), albedo, metallic);
	vec3f F = FresnelSchlick(F0, dot(V, H));

	vec3f specular_brdf = D * F * G / (4.0f * NdotL * NdotV);
	vec3f diffuse_brdf = albedo * M_1_PIf;
	vec3f brdf = p_diffuse * diffuse_brdf + (1.0f - p_diffuse) * specular_brdf;

	pdf = (1.0f - p_diffuse) * Dv * abs(1.0f / (4.0f * dot(V, H))) + p_diffuse * CosinePdfHemisphere(NdotL);

	return brdf;
}
//*************************************metal workflow*************************************

//*************************************thin dielectric*************************************
__forceinline__ __device__ vec3f EvaluateThinDielectric(const Interaction& isect,
	const vec3f& world_V, const vec3f& world_L, float& pdf) {
	float eta = isect.material.int_ior / isect.material.ext_ior;
	vec3f albedo = isect.material.albedo;
	float roughness = isect.material.roughness;
	float alpha = sqr(roughness);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H;

	bool isReflect = dot(N, L) * dot(N, V) > 0.0f;
	if (isReflect) {
		H = normalize(V + L);
	}
	else {
		vec3f Vr = ToWorld(ToLocal(V, -N), N);
		H = normalize(Vr + L);
		if (dot(N, H) < 0.0f) {
			H = -H;
		}
	}

	vec3f bsdf = 0.0f;
	float Dv = DistributionVisibleGGX(V, H, N, alpha, alpha);
	float F = FresnelDielectric(V, H, 1.0f / eta);
	if (F < 1.0f) {
        F *= 2.0f / (1.0f + F);
    }
	float D = DistributionGGX(H, N, alpha, alpha);
	float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
	float NdotV = abs(dot(N, V));
	float NdotL = abs(dot(N, L));
	float G = GeometrySmith_1(V, H, N, alpha, alpha) * GeometrySmith_1(L, H, N, alpha, alpha);
	if (isReflect) {
		if (dot(N, L) <= 0.0f) {
			return 0.0f;
		}

		pdf = F * Dv * dwh_dwi;

		bsdf = albedo * F * D * G / (4.0f * NdotV * NdotL);
	}
	else {
		if (dot(N, L) * dot(N, V) >= 0.0f) {
			return 0.0f;
		}

		pdf = (1.0f - F) * Dv * dwh_dwi;

		bsdf = albedo * (1.0f - F) * D * G / (4.0f * NdotV * NdotL);
	}

	return bsdf;
}

__forceinline__ __device__ vec3f SampleThinDielectric(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	float eta = isect.material.int_ior / isect.material.ext_ior;
	vec3f albedo = isect.material.albedo;
	float roughness = isect.material.roughness;
	float alpha = sqr(roughness);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f H = SampleVisibleGGX(N, V, alpha, alpha, vec2f(random(), random()));
	H = ToWorld(H, N);

	vec3f bsdf = 0.0f;
	float Dv = DistributionVisibleGGX(V, H, N, alpha, alpha);
	float F = FresnelDielectric(V, H, 1.0f / eta);
	if (F < 1.0f) {
        F *= 2.0f / (1.0f + F);
    }
	float D = DistributionGGX(H, N, alpha, alpha);
	if (random() < F) {
		world_L = reflect(-V, H);
		vec3f L = world_L;

		if (dot(N, L) <= 0.0f) {
			return 0.0f;
		}

		float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
		float NdotV = abs(dot(N, V));
		float NdotL = abs(dot(N, L));
		float G = GeometrySmith_1(V, H, N, alpha, alpha) * GeometrySmith_1(L, H, N, alpha, alpha);
		pdf = F * Dv * dwh_dwi;

		bsdf = albedo * F * D * G / (4.0f * NdotV * NdotL);
	}
	else {
		world_L = -V;
		vec3f L = world_L;

		if (dot(N, L) * dot(N, V) >= 0.0f) {
			return 0.0f;
		}

		float NdotV = abs(dot(N, V));
		float NdotL = abs(dot(N, L));
		float G = GeometrySmith_1(V, H, N, alpha, alpha) * GeometrySmith_1(L, H, N, alpha, alpha);
		float dwh_dwi = abs(1.0f / (4.0f * dot(V, H)));
		pdf = (1.0f - F) * Dv * dwh_dwi;

		bsdf = albedo * (1.0f - F) * D * G / (4.0f * NdotV * NdotL);
	}

	return bsdf;
}
//*************************************thin dielectric*************************************

//*************************************clearcoated conductor*************************************
__forceinline__ __device__ vec3f EvaluateClearCoatedConductor(const Interaction& isect,
	const vec3f& world_V, const vec3f& world_L, float& pdf) {
	float alpha_u = sqr(isect.material.coat_roughness_u);
	float alpha_v = sqr(isect.material.coat_roughness_v);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f L = world_L;
	vec3f H = normalize(V + L);

	float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);

	float NdotV = dot(N, V);
	float NdotL = dot(N, L);

	if (NdotV <= 0.0f || NdotL <= 0.0f) {
		return 0.0f;
	}

	float F = FresnelDielectric(V, H, 1.0f / 1.5f);
	float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	float D = DistributionGGX(H, N, alpha_u, alpha_v);

    float cond_pdf = 0.0f;
	float coat_pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));
	vec3f cond_brdf = EvaluateConductor(isect, world_V, world_L, cond_pdf);
	vec3f coat_brdf = F * D * G / (4.0f * NdotV * NdotL);

	vec3f brdf = F * coat_brdf + (1.0f - F) * cond_brdf;
	pdf = F * coat_pdf + (1.0f - F) * cond_pdf;

	return brdf;
}

__forceinline__ __device__ vec3f SampleClearCoatedConductor(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	float alpha_u = sqr(isect.material.coat_roughness_u);
	float alpha_v = sqr(isect.material.coat_roughness_v);

	vec3f N = isect.shadeNormal;
	vec3f V = world_V;
	vec3f H;

    float F = FresnelDielectric(V, N, 1.0f / 1.5f);
	if(random() < F * 10.0f) {
		H = SampleVisibleGGX(N, V, alpha_u, alpha_v, vec2f(random(), random()));
	    H = ToWorld(H, N);
	    world_L = reflect(-V, H);
	    vec3f L = world_L;

		float NdotV = dot(N, V);
	    float NdotL = dot(N, L);

	    if (NdotV <= 0.0f || NdotL <= 0.0f) {
		    return 0.0f;
	    }

		float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
		F = FresnelDielectric(V, H, 1.0f / 1.5f);
		float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	    float D = DistributionGGX(H, N, alpha_u, alpha_v);
	    float coat_pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));
        float cond_pdf = 0.0f;
	    vec3f cond_brdf = EvaluateConductor(isect, world_V, world_L, cond_pdf);
		vec3f coat_brdf = F * D * G / (4.0f * NdotV * NdotL);

	    vec3f brdf = F * coat_brdf + (1.0f - F) * cond_brdf;
	    pdf = F * coat_pdf + (1.0f - F) * cond_pdf;

		return brdf;
	}
	else {
		float cond_pdf = 0.0f;
		vec3f cond_brdf = SampleConductor(isect, random, world_V, world_L, cond_pdf);
		vec3f L = world_L;
		H = normalize(V + L);

		float NdotV = dot(N, V);
	    float NdotL = dot(N, L);

	    if (NdotV <= 0.0f || NdotL <= 0.0f) {
		    return 0.0f;
	    }

		float Dv = DistributionVisibleGGX(V, H, N, alpha_u, alpha_v);
		F = FresnelDielectric(V, H, 1.0f / 1.5f);
		float G = GeometrySmith_1(V, H, N, alpha_u, alpha_v) * GeometrySmith_1(L, H, N, alpha_u, alpha_v);
	    float D = DistributionGGX(H, N, alpha_u, alpha_v);
	    float coat_pdf = Dv * abs(1.0f / (4.0f * dot(V, H)));
		vec3f coat_brdf = F * D * G / (4.0f * NdotV * NdotL);

		vec3f brdf = F * coat_brdf + (1.0f - F) * cond_brdf;
	    pdf = F * coat_pdf + (1.0f - F) * cond_pdf;

		return brdf;
	}
}
//*************************************clearcoated conductor*************************************

//*************************************material*************************************
__forceinline__ __device__ vec3f EvaluateMaterial(const Interaction& isect,
	const vec3f& world_V, const vec3f& world_L, float& pdf) {
	vec3f bsdf = 0.0f;
	pdf = 0.0f;
	if (isect.material.type == MaterialType::Diffuse) {
		bsdf = EvaluateDiffuse(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Conductor) {
		bsdf = EvaluateConductor(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Dielectric) {
		bsdf = EvaluateDielectric(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Plastic) {
		bsdf = EvaluatePlastic(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::MetalWorkflow) {
		bsdf = EvaluateMetalWorkflow(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::ThinDielectric) {
		bsdf = EvaluateThinDielectric(isect, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::ClearCoatedConductor) {
		bsdf = EvaluateClearCoatedConductor(isect, world_V, world_L, pdf);
	}

	return bsdf;
}

__forceinline__ __device__ vec3f SampleMaterial(const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
	vec3f bsdf = 0.0f;
	pdf = 0.0f;
	if (isect.material.type == MaterialType::Diffuse) {
		bsdf = SampleDiffuse(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Conductor) {
		bsdf = SampleConductor(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Dielectric) {
		bsdf = SampleDielectric(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::Plastic) {
		bsdf = SamplePlastic(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::MetalWorkflow) {
		bsdf = SampleMetalWorkflow(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::ThinDielectric) {
		bsdf = SampleThinDielectric(isect, random, world_V, world_L, pdf);
	}
	else if (isect.material.type == MaterialType::ClearCoatedConductor) {
		bsdf = SampleClearCoatedConductor(isect, random, world_V, world_L, pdf);
	}

	return bsdf;
}
//*************************************material*************************************