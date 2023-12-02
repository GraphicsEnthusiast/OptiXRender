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
	if (!light.doubleSide && cos_theta < 0.0f) {
		pdf = 0.0f;

		return 0.0f;
	}

	float area = length(light.u) * length(light.v);
	pdf = 1.0f / area;
	pdf *= dist_sq / abs(cos_theta);

	return light.radiance;
}
//*************************************quad*************************************

//*************************************sphere*************************************
__forceinline__ __device__ float IntersectSphere(const Ray& ray, const vec3f& center, float radius) {
	vec3f co = center - ray.origin;
	float b = dot(co, ray.direction);
	float det = b * b - dot(co, co) + radius * radius;
	if (det < 0.0f) {
		return FLT_MAX;
	}

	det = sqrt(det);
	float t1 = b - det;
	if (t1 > EPS) {
		return t1;
	}

	float t2 = b + det;
	if (t2 > EPS) {
		return t2;
	}

	return FLT_MAX;
}

__forceinline__ __device__ vec3f EvaluateSphere(const Light& light, const Ray& ray, float distance, float& pdf, float& light_distance) {
	const vec3f& light_pos = light.position;
	light_distance = IntersectSphere(ray, light_pos, light.radius);
	if (light_distance < distance) {
		vec3f dir = light_pos - ray.origin;
		float dist_sq = dot(dir, dir);
		float sin_theta_sq = light.radius * light.radius / dist_sq;
		float cos_theta = sqrt(1.0f - sin_theta_sq);
		pdf = UniformPdfCone(cos_theta);

		return light.radiance;
	}

	pdf = 0.0f;

	return 0.0f;
}

__forceinline__ __device__ vec3f SampleSphere(const Light& light, const vec3f& hitpos, const vec2f& sample, vec3f& world_L, float& distance, float& pdf) {
	// 相比均匀采样球体，球中心和相交点连线构成的圆锥更快
	vec3f dir = light.position - hitpos;
	float dist_sq = dot(dir, dir);
	float inv_dist = 1.0f / sqrt(dist_sq);
	dir *= inv_dist;
	distance = dist_sq * inv_dist;

	float sin_theta = light.radius * inv_dist;
	if (sin_theta < 1.0f) {
		float cos_theta = sqrt(1.0f - sin_theta * sin_theta);
		vec3f local_L = UniformSampleCone(sample, cos_theta);
		float cos_i = local_L.z;
		world_L = ToWorld(local_L, dir);
		pdf = UniformPdfCone(cos_theta);
		distance = cos_i * (distance) - sqrt(max(0.0f, light.radius * light.radius - (1.0f - cos_i * cos_i) * dist_sq));

		return light.radiance;
	}

	pdf = 0.0f;

	return 0.0f;
}
//*************************************sphere*************************************

//*************************************infinite area*************************************
__forceinline__ __host__ AliasTable2D ComputeInfiniteAliasTable(uint32_t* data, int nx, int ny, int nn) {
    float* pdf = new float[nx * ny];
	float sum = 0.0f;
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			vec3f l(data[nn * (j * nx + i)], data[nn * (j * nx + i) + 1], data[nn * (j * nx + i) + 2]);
			pdf[j * nx + i] = Luminance(l) * sin((float)(j + 0.5f) / ny * M_PIf);
			sum += pdf[j * nx + i];
		}
	}
	AliasTable2D table(pdf, nx, ny);
	delete[] pdf;

	return table;
}
//*************************************infinite area*************************************

//*************************************light*************************************
__forceinline__ __device__ vec3f EvaluateLight(const Light& light, const Ray& ray, float distance, float& pdf, float& light_distance) {
	if (light.type == LightType::Quad) {
		return EvaluateQuad(light, ray, distance, pdf, light_distance);
	}
	else if (light.type == LightType::Sphere) {
		return EvaluateSphere(light, ray, distance, pdf, light_distance);
	}

	return 0.0f;
}

__forceinline__ __device__ vec3f SampleLight(const Light& light, const vec3f& hitpos, const vec2f& sample, vec3f& world_L, float& distance, float& pdf) {
	if (light.type == LightType::Quad) {
		return SampleQuad(light, hitpos, sample, world_L, distance, pdf);
	}
	else if (light.type == LightType::Sphere) {
		return SampleSphere(light, hitpos, sample, world_L, distance, pdf);
	}

	pdf = 0.0f;

	return 0.0f;
}
//*************************************light*************************************