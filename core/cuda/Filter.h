#pragma once

#include "Utils.h"

__forceinline__ __device__ vec2f FilterBox(const vec2f& sample) {
	return sample - vec2f(0.5f, 0.5f);
}

__forceinline__ __device__ vec2f FilterTent(const vec2f& sample) {
	vec2f j = sample;
	j = j * 2.0f;
	j.x = j.x < 1.0f ? sqrt(j.x) - 1.0f : 1.0f - sqrt(2.0f - j.x);
	j.y = j.y < 1.0f ? sqrt(j.y) - 1.0f : 1.0f - sqrt(2.0f - j.y);

	return vec2f(0.5f, 0.5f) + j;
}

__forceinline__ __device__ vec2f FilterTriangle(const vec2f& sample) {
	float u1 = sample.x;
	float u2 = sample.y;
	if (u2 > u1) {
		u1 *= 0.5f;
		u2 -= u1;
	}
	else {
		u2 *= 0.5f;
		u1 -= u2;
	}

	return vec2f(0.5f, 0.5f) + vec2f(u1, u2);
}

__forceinline__ __device__ vec2f FilterGaussian(const vec2f& sample) {
	float r1 = fmaxf(FLT_MIN, sample.x);
	float r = sqrtf(-2.0f * logf(r1));
	float theta = M_2PIf * sample.y;
	vec2f uv = r * vec2f(cosf(theta), sinf(theta));

	return vec2f(0.5f, 0.5f) + 0.375f * uv;
}

__forceinline__ __device__ vec2f FilterJitter(const vec2f& sample, FilterType type) {
    if (type == FilterType::Box) {
        return FilterBox(sample);
    }
    else if (type == FilterType::Tent) {
        return FilterTent(sample);
    }
    else if (type == FilterType::Triangle) {
        return FilterTriangle(sample);
    }
    else if (type == FilterType::Gaussian) {
        return FilterGaussian(sample);
    }

    return sample;
}