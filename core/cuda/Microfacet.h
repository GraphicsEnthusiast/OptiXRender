#pragma once

#include "Utils.h"

//*************************************fresnel*************************************
__forceinline__ __device__ vec3f FresnelConductor(const vec3f& V, const vec3f& H, const vec3f& eta_r, const vec3f& eta_i) {
	vec3f N = H;
	float cos_v_n = dot(V, N),
		cos_v_n_2 = cos_v_n * cos_v_n,
		sin_v_n_2 = 1.0f - cos_v_n_2,
		sin_v_n_4 = sin_v_n_2 * sin_v_n_2;

	vec3f temp_1 = eta_r * eta_r - eta_i * eta_i - sin_v_n_2,
		a_2_pb_2 = temp_1 * temp_1 + 4.0f * eta_i * eta_i * eta_r * eta_r;
	for (int i = 0; i < 3; i++) {
		a_2_pb_2[i] = sqrt(max(0.0f, a_2_pb_2[i]));
	}
	vec3f a = 0.5f * (a_2_pb_2 + temp_1);
	for (int i = 0; i < 3; i++) {
		a[i] = sqrt(max(0.0f, a[i]));
	}
	vec3f term_1 = a_2_pb_2 + sin_v_n_2,
		term_2 = 2.0f * cos_v_n * a,
		term_3 = a_2_pb_2 * cos_v_n_2 + sin_v_n_4,
		term_4 = term_2 * sin_v_n_2,
		r_s = (term_1 - term_2) / (term_1 + term_2),
		r_p = r_s * (term_3 - term_4) / (term_3 + term_4);

	return 0.5f * (r_s + r_p);
}

__forceinline__ __device__ vec3f AverageFresnelConductor(vec3f eta, vec3f k) {
	auto reflectivity = vec3f(0.0f),
		edgetint = vec3f(0.0f);
	float temp1 = 0.0f, temp2 = 0.0f, temp3 = 0.0f;
	for (int i = 0; i < 3; i++) {
		reflectivity[i] = (sqr(eta[i] - 1.0f) + sqr(k[i])) / (sqr(eta[i] + 1.0f) + sqr(k[i]));
		temp1 = 1.0f + sqrt(reflectivity[i]);
		temp2 = 1.0f - sqrt(reflectivity[i]);
		temp3 = (1.0f - reflectivity[i]) / (1.0f + reflectivity[i]);
		edgetint[i] = (temp1 - eta[i] * temp2) / (temp1 - temp3 * temp2);
	}

	return vec3f(0.087237f) +
		0.0230685f * edgetint -
		0.0864902f * edgetint * edgetint +
		0.0774594f * edgetint * edgetint * edgetint +
		0.782654f * reflectivity -
		0.136432f * reflectivity * reflectivity +
		0.278708f * reflectivity * reflectivity * reflectivity +
		0.19744f * edgetint * reflectivity +
		0.0360605f * edgetint * edgetint * reflectivity -
		0.2586f * edgetint * reflectivity * reflectivity;
}

__forceinline__ __device__ float FresnelDielectric(const vec3f& V, const vec3f& H, float eta_inv) {
	float cos_theta_i = abs(dot(V, H));
	float cos_theta_t_2 = 1.0f - sqr(eta_inv) * (1.0f - sqr(cos_theta_i));
	if (cos_theta_t_2 <= 0.0f) {
		return 1.0f;
	}
	else {
		float cos_theta_t = sqrt(cos_theta_t_2),
			Rs_sqrt = (eta_inv * cos_theta_i - cos_theta_t) / (eta_inv * cos_theta_i + cos_theta_t),
			Rp_sqrt = (cos_theta_i - eta_inv * cos_theta_t) / (cos_theta_i + eta_inv * cos_theta_t);

		return (Rs_sqrt * Rs_sqrt + Rp_sqrt * Rp_sqrt) / 2.0f;
	}
}

__forceinline__ __device__ float AverageFresnelDielectric(float eta) {
	if (eta < 1.0f) {
		/* Fit by Egan and Hilgeman (1973). Vrks reasonably well for
			"normal" IOR values (<2).
			Max rel. error in 1.0 - 1.5 : 0.1%
			Max rel. error in 1.5 - 2   : 0.6%
			Max rel. error in 2.0 - 5   : 9.5%
		*/
		return -1.4399f * (eta * eta) + 0.7099f * eta + 0.6681f + 0.0636f / eta;
	}
	else {
		/* Fit by d'Eon and Irving (2011)

			Maintains a good accuracy even for unistic IOR values.

			Max rel. error in 1.0 - 2.0   : 0.1%
			Max rel. error in 2.0 - 10.0  : 0.2%
		*/
		float inv_eta = 1.0f / eta,
			inv_eta_2 = inv_eta * inv_eta,
			inv_eta_3 = inv_eta_2 * inv_eta,
			inv_eta_4 = inv_eta_3 * inv_eta,
			inv_eta_5 = inv_eta_4 * inv_eta;

		return 0.919317f - 3.4793f * inv_eta + 6.75335f * inv_eta_2 - 7.80989f * inv_eta_3 + 4.98554f * inv_eta_4 - 1.36881f * inv_eta_5;
	}
}
//*************************************fresnel*************************************

//*************************************ggx*************************************
__forceinline__ __device__ float GeometrySmith_1(const vec3f& V, const vec3f& H, const vec3f& N, float alpha_u, float alpha_v) {
	float cos_v_n = dot(V, N);

	if (cos_v_n * dot(V, H) <= 0.0f) {
		return 0.0f;
	}

	if (abs(cos_v_n - 1.0f) < EPS) {
		return 1.0f;
	}

	if (alpha_u == alpha_v) {
		float cos_v_n_2 = sqr(cos_v_n),
			tan_v_n_2 = (1.0f - cos_v_n_2) / cos_v_n_2,
			alpha_2 = alpha_u * alpha_u;

		return 2.0f / (1.0f + sqrt(1.0f + alpha_2 * tan_v_n_2));
	}
	else {
		vec3f dir = ToLocal(V, N);
		float xy_alpha_2 = sqr(alpha_u * dir.x) + sqr(alpha_v * dir.y),
			tan_v_n_alpha_2 = xy_alpha_2 / sqr(dir.z);

		return 2.0f / (1.0f + sqrt(1.0f + tan_v_n_alpha_2));
	}
}

__forceinline__ __device__ float DistributionGGX(const vec3f& H, const vec3f& N, float alpha_u, float alpha_v) {
	float cos_theta = dot(H, N);
	if (cos_theta <= 0.0f) {
		return 0.0f;
	}
	float cos_theta_2 = sqr(cos_theta),
		tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2,
		alpha_2 = alpha_u * alpha_v;
	if (alpha_u == alpha_v) {
		return alpha_2 / (M_PIf * pow(cos_theta, 3.0f) * sqr(alpha_2 + tan_theta_2));
	}
	else {
		vec3f dir = ToLocal(H, N);

		return cos_theta / (M_PIf * alpha_2 * sqr(sqr(dir.x / alpha_u) + sqr(dir.y / alpha_v) + sqr(dir.z)));
	}
}

__forceinline__ __device__ float DistributionVisibleGGX(const vec3f& V, const vec3f& H, const vec3f& N, float alpha_u, float alpha_v) {
	return GeometrySmith_1(V, H, N, alpha_u, alpha_v) * dot(V, H) * DistributionGGX(H, N, alpha_u, alpha_v) / dot(N, V);
}

__forceinline__ __device__ vec3f SampleGGX(const vec3f& N, float alpha_u, float alpha_v, const vec2f& sample) {
	float sin_phi = 0.0f, cos_phi = 0.0f, alpha_2 = 0.0f;
	if (alpha_u == alpha_v) {
		float phi = M_2PIf * sample.y;
		cos_phi = cos(phi);
		sin_phi = sin(phi);
		alpha_2 = alpha_u * alpha_u;
	}
	else {
		float phi = atan(alpha_v / alpha_u * tan(M_PIf + M_2PIf * sample.y)) + M_PIf * floor(2.0f * sample.y + 0.5f);
		cos_phi = cos(phi);
		sin_phi = sin(phi);
		alpha_2 = 1.0f / (sqr(cos_phi / alpha_u) + sqr(sin_phi / alpha_v));
	}
	float tan_theta_2 = alpha_2 * sample.x / (1.0f - sample.x),
		cos_theta = 1.0f / sqrt(1.0f + tan_theta_2),
		sin_theta = sqrt(1.0f - cos_theta * cos_theta);
	vec3f H = normalize(vec3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta));

	return H;
}

__forceinline__ __device__ vec3f SampleVisibleGGX(const vec3f& N, const vec3f& Ve, float alpha_u, float alpha_v, const vec2f& sample) {
	vec3f V = ToLocal(Ve, N);
	vec3f Vh = normalize(vec3f(alpha_u * V.x, alpha_v * V.y, V.z));

	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float len2 = sqr(Vh.x) + sqr(Vh.y);
	vec3f T1 = len2 > 0.0f ? vec3f(-Vh.y, Vh.x, 0.0f) * sqrt(1.0f / len2) : vec3f(1.0f, 0.0f, 0.0f);
	vec3f T2 = cross(Vh, T1);

	// Section 4.2: parameterization of the projected area
	float u = sample.x;
	float v = sample.y;
	float r = sqrt(u);
	float phi = v * M_2PIf;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * sqrt(1.0f - sqr(t1)) + s * t2;

	// Section 4.3: reprojection onto hemisphere
	vec3f Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - sqr(t1) - sqr(t2))) * Vh;

	// Section 3.4: transforming the normal back to the ellipsoid configuration
	vec3f H = normalize(vec3f(alpha_u * Nh.x, alpha_v * Nh.y, max(0.0f, Nh.z)));

	return H;
}
//*************************************ggx*************************************