#pragma once

#include "Sampling.h"

//*************************************quad*************************************
__forceinline__ __device__ float IntersectQuad(const Ray& ray, const vec3f& corner, const vec3f& _u, const vec3f& _v) {
    vec3f n = normalize(cross(_u, _v));
    vec3f u = _u * (1.0f / dot(_u, _u));
    vec3f v = _v * (1.0f / dot(_v, _v));

	float dt = dot(ray.direction, n);
	float t = (dot(n, corner) - dot(n, ray.origin)) / dt;
	if (t > EPS) {
        vec3f p = ray.origin + ray.direction * t;
        vec3f vi = p - corner;
		float a1 = dot(u, vi);
		if (a1 >= 0.0f && a1 <= 1.0f) {
			float a2 = dot(v, vi);
			if (a2 >= 0.0f && a2 <= 1.0f) {
                return t;
            }
		}
	}

    return FLT_MAX;
}

__forceinline__ __device__ vec3f EvaluateQuad(const Light& light, const Ray& ray, float distance, float& pdf, float& light_distance) {
	vec3f normal = normalize(cross(light.u, light.v));
    float cos_theta = dot(ray.direction, normal);
    if (!light.doubleSide && cos_theta < 0.0f) {
		pdf = 0.0f;

		return 0.0f;
	}

	light_distance = IntersectQuad(ray, light.position, light.u, light.v);
    float area = length(light.u) * length(light.v);
	if (light_distance < distance) {
        //surface pdf
		pdf = 1.0f / area; 

		//solid angel pdf
		pdf *= sqr(light_distance) / abs(cos_theta);

		return light.radiance;
	}

	pdf = 0.0f;

	return 0.0f;
}

__forceinline__ __device__ vec3f SampleQuad(const Light& light, const vec3f& hitpos, const vec2f& sample, vec3f& world_L, float& distance, float& pdf) {
	vec3f normal = normalize(cross(light.u, light.v));
	vec3f pos = light.position + light.u * sample.x + light.v * sample.y;
	world_L = pos - hitpos;
	float dist_sq = dot(world_L, world_L);
	distance = sqrt(dist_sq);
	world_L /= distance;

	float cos_theta = dot(world_L, normal);
	if (!light.doubleSide && cos_theta < 0.0f){ 
		pdf = 0.0f;

		return 0.0f;
	}

    float area = length(light.u) * length(light.v);
	pdf = 1.0f / area;
	pdf *= dist_sq / abs(cos_theta);

	return light.radiance;
}
//*************************************quad*************************************

//*************************************light*************************************
__forceinline__ __device__ vec3f EvaluateLight(const Light& light, const Ray& ray, float distance, float& pdf, float& light_distance) {
	if (light.type == LightType::Quad) {
		return EvaluateQuad(light, ray, distance, pdf, light_distance);
	}
	
	return 0.0f;
}

__forceinline__ __device__ vec3f SampleLight(const Light& light, const vec3f& hitpos, const vec2f& sample, vec3f& world_L, float& distance, float& pdf) {
	if (light.type == LightType::Quad) {
		return SampleQuad(light, hitpos, sample, world_L, distance, pdf);
	}
	
	pdf = 0.0f;

	return 0.0f;
}
//*************************************light*************************************