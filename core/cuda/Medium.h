#pragma once

#include "Utils.h"
#include "Sampling.h"

//*************************************isotropic phase*************************************
__forceinline__ __device__ vec3f EvaluateIsotropic(const Medium& medium, const Interaction& isect, const vec3f& world_V, const vec3f& world_L, float& pdf) {
    vec3f attenuation(0.25f * M_1_PIf);
    pdf = 0.25f * M_1_PIf;
    
    return attenuation;
}

__forceinline__ __device__ vec3f SampleIsotropic(const Medium& medium, const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
    world_L = UniformSampleSphere(vec2f(random(), random()));

    vec3f attenuation(0.25f * M_1_PIf);
    pdf = 0.25f * M_1_PIf;
    
    return attenuation;
}
//*************************************isotropic phase*************************************

//*************************************henyey greenstein phase*************************************
__forceinline__ __device__ vec3f EvaluateHenyeyGreenstein(const Medium& medium, const Interaction& isect, const vec3f& world_V, const vec3f& world_L, float& pdf) {
    float cos_theta = dot(world_L, world_V);
    vec3f attenuation = 0.0f;
    pdf = 0.0f;
    for (int dim = 0; dim < 3; ++dim) {
        float temp = 1.0f + medium.phase.g[dim] * medium.phase.g[dim] + 2.0f * medium.phase.g[dim] * cos_theta;
        attenuation[dim] = (0.25f * M_1_PIf) * (1.0f - medium.phase.g[dim] * medium.phase.g[dim]) / (temp * sqrt(temp));
        pdf += attenuation[dim];
    }

    pdf *= (1.0f / 3.0f);

    if (pdf <= 0.0f) {
        return 0.0f;
    }

    return attenuation;
}

__forceinline__ __device__ vec3f SampleHenyeyGreenstein(const Medium& medium, const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
    int channel = min(static_cast<int>(random() * 3), 2);
    float g = medium.phase.g[channel];

    float cos_theta = 0.0f;
    if (abs(g) < EPS) {
        cos_theta = 1.0f - 2.0f * random();
    }
    else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * random());
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }

    pdf = 0.0f;
    vec3f attenuation = 0.0f;
    for (int dim = 0; dim < 3; ++dim) {
        float temp = 1.0f + medium.phase.g[dim] * medium.phase.g[dim] + 2.0f * medium.phase.g[dim] * cos_theta;
        attenuation[dim] = (0.25f * M_1_PIf) * (1.0f - medium.phase.g[dim] * medium.phase.g[dim]) / (temp * sqrt(temp));
        pdf += attenuation[dim];
    }
    pdf *= (1.0f / 3.0f);

    if (pdf <= 0.0f) {
        return 0.0f;
    }

    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = M_2PIf * random();

    vec3f local_L = normalize(vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta));
    world_L = ToWorld(local_L, world_V);

    return attenuation;
}
//*************************************henyey greenstein phase*************************************

//*************************************phase*************************************
__forceinline__ __device__ vec3f EvaluatePhase(const Medium& medium, const Interaction& isect, const vec3f& world_V, const vec3f& world_L, float& pdf) {
    vec3f attenuation = 0.0f;
    pdf = 0.0f;
    if (medium.phase.type == PhaseType::Isotropic) {
        attenuation = EvaluateIsotropic(medium, isect, world_V, world_L, pdf);
    }
    else if (medium.phase.type == PhaseType::HenyeyGreenstein) {
        attenuation = EvaluateHenyeyGreenstein(medium, isect, world_V, world_L, pdf);
    }
    
    return attenuation;
}

__forceinline__ __device__ vec3f SamplePhase(const Medium& medium, const Interaction& isect, Random& random,
	const vec3f& world_V, vec3f& world_L, float& pdf) {
    vec3f attenuation = 0.0f;
    pdf = 0.0f;
    if (medium.phase.type == PhaseType::Isotropic) {
        attenuation = SampleIsotropic(medium, isect, random, world_V, world_L, pdf);
    }
    else if (medium.phase.type == PhaseType::HenyeyGreenstein) {
        attenuation = SampleHenyeyGreenstein(medium, isect, random, world_V, world_L, pdf);
    }
    
    return attenuation;
}
//*************************************phase*************************************

//*************************************homogeneous medium*************************************
__forceinline__ __device__ vec3f EvaluateHomogeneousDistance(const Medium& medium, bool scattered, float distance, float& trans_pdf) {
    vec3f sigma_s = medium.sigma_s * medium.scale;
    vec3f sigma_a = medium.sigma_a * medium.scale;
    vec3f sigma_t = sigma_a + sigma_s;
    vec3f albedo = sigma_s / sigma_t;
    float medium_sampling_weight = 0.0f;
    for (int dim = 0; dim < 3; ++dim) {
		if (albedo[dim] > medium_sampling_weight && sigma_t[dim] != 0.0f) {
			medium_sampling_weight = albedo[dim];
		}
	}
	if (medium_sampling_weight > 0.0f) {
		medium_sampling_weight = max(medium_sampling_weight, 0.5f);
	}
    
    vec3f attenuation(0.0f);
	float pdf = 0.0f;
	bool valid = false;
	for (int dim = 0; dim < 3; ++dim) {
		attenuation[dim] = exp(-sigma_t[dim] * distance);
		if (attenuation[dim] > 0.0f) {
			valid = true;
		}
	}

	if (scattered) {
		for (int dim = 0; dim < 3; ++dim) {
			pdf += sigma_t[dim] * attenuation[dim];
		}
		pdf *= medium_sampling_weight * (1.0f / 3.0f);
		attenuation *= sigma_s;
	}
	else {
		for (int dim = 0; dim < 3; ++dim) {
			pdf += attenuation[dim];
		}
		pdf = medium_sampling_weight * (1.0f / 3.0f) * pdf + (1.0f - medium_sampling_weight);
	}
    trans_pdf = pdf;

	if (!valid) {
		attenuation = vec3f(0.0f);
	}
    vec3f transmittance = attenuation;

    return transmittance;
}

__forceinline__ __device__ bool SampleHomogeneousDistance(const Medium& medium, float max_distance, float& distance, 
    float& trans_pdf, vec3f& transmittance, Random& random) {
    vec3f sigma_s = medium.sigma_s * medium.scale;
    vec3f sigma_a = medium.sigma_a * medium.scale;
    vec3f sigma_t = sigma_a + sigma_s;
    vec3f albedo = sigma_s / sigma_t;
    float medium_sampling_weight = 0.0f;
    for (int dim = 0; dim < 3; ++dim) {
		if (albedo[dim] > medium_sampling_weight && sigma_t[dim] != 0.0f) {
			medium_sampling_weight = albedo[dim];
		}
	}
	if (medium_sampling_weight > 0.0f) {
		medium_sampling_weight = max(medium_sampling_weight, 0.5f);
	}

	bool scattered = false;
	float xi_1 = random();
	if (xi_1 < medium_sampling_weight) { // 抽样光线在介质内部是否发生散射
		xi_1 /= medium_sampling_weight;
		const int channel = min(static_cast<int>(random() * 3), 2);
		distance = -log(1.0f - xi_1) / sigma_t[channel];
		if (distance < max_distance) { // 光线在介质内部发生了散射
			trans_pdf = 0.0f;
			for (int dim = 0; dim < 3; ++dim) {
				trans_pdf += sigma_t[dim] * exp(-sigma_t[dim] * distance);
			}
			trans_pdf *= medium_sampling_weight * (1.0f / 3.0f);
			scattered = true;
		}
	}

	if (!scattered) { // 光线在介质内部没有发生散射
		distance = max_distance;
		trans_pdf = 0.0f;
		for (int dim = 0; dim < 3; ++dim) {
			trans_pdf += exp(-sigma_t[dim] * distance);
		}
		trans_pdf = medium_sampling_weight * (1.0f / 3.0f) * trans_pdf + (1.0f - medium_sampling_weight);
	}

	bool valid = false;
	for (int dim = 0; dim < 3; ++dim) {
		(transmittance)[dim] = exp(-sigma_t[dim] * distance);
		if ((transmittance)[dim] > 0.0f) {
			valid = true;
		}
	}

	if (scattered) {
		transmittance *= sigma_s;
	}

	if (!valid) {
		transmittance = vec3f(0.0f);
	}

	return scattered;
}
//*************************************homogeneous medium*************************************

//*************************************medium*************************************
__forceinline__ __device__ vec3f EvaluateMediumDistance(const Medium& medium, bool scattered, float distance, float& trans_pdf) {
    vec3f transmittance = 0.0f;
    trans_pdf = 0.0f;

    if (medium.type == MediumType::Homogeneous) {
        transmittance = EvaluateHomogeneousDistance(medium, scattered, distance, trans_pdf);
    }

    return transmittance;
}

__forceinline__ __device__ bool SampleMediumDistance(const Medium& medium, float max_distance, float& distance, 
    float& trans_pdf, vec3f& transmittance, Random& random) {
    bool scattered = false;
    trans_pdf = 0.0f;
    transmittance = 0.0f;

    if (medium.type == MediumType::Homogeneous) {
        scattered = SampleHomogeneousDistance(medium, max_distance, distance, trans_pdf, transmittance, random);
    }

    return scattered;
}
//*************************************medium*************************************