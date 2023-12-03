#pragma once

#include "Utils.h"
#include "CUDABuffer.h"
#include <vector>

__forceinline__ __device__ float PowerHeuristic(float pdf1, float pdf2, int beta) {
	float p1 = pow(pdf1, beta);
	float p2 = pow(pdf2, beta);

	return p1 / (p1 + p2);
}

__forceinline__ __device__ vec3f PlaneToSphere(const vec2f& uv) {
	vec2f xy = uv;
	xy.y = 1.0f - xy.y; //flip y

	//获取角度
	float phi = M_2PIf * (xy.x - 0.5f);    //[-pi ~ pi]
	float theta = M_PIf * (xy.y - 0.5f);        //[-pi/2 ~ pi/2]   

	//球坐标计算方向
	vec3f L = normalize(vec3f(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi)));

	return L;
}

__forceinline__ __device__ vec2f SphereToPlane(const vec3f& v) {
	vec2f uv(atan2(v.z, v.x), asin(v.y));
	uv.x /= M_2PIf;
	uv.y /= M_PIf;
	uv += 0.5f;

	return uv;
}

//*************************************cone*************************************
__forceinline__ __device__ vec3f UniformSampleCone(const vec2f& sample, float cos_angle) {
	vec3f p = 0.0f;

	float phi = M_2PIf * sample.x;
	float cos_theta = mix(cos_angle, 1.0f, sample.y);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cos(phi);
	p.y = sin_theta * sin(phi);
	p.z = cos_theta;

	return normalize(p);
}

__forceinline__ __device__ float UniformPdfCone(float cos_angle) {
	return 1.0f / (M_2PIf * (1.0f - cos_angle));
}
//*************************************cone*************************************

//*************************************cosine*************************************
__forceinline__ __device__ vec3f CosineSampleHemisphere(const vec2f& sample) {
	vec3f p = 0.0f;

	// uniformly sample disk
	float r = sqrtf(sample.x);
	float phi = M_2PIf * sample.y;
	p.x = r * cos(phi);
	p.y = r * sin(phi);

	// project up to hemisphere
	p.z = sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y)); //cosTheta NoL

	return normalize(p);
}

__forceinline__ __device__ float CosinePdfHemisphere(float NdotL) {
	return NdotL * M_1_PIf;
}
//*************************************cosine*************************************

//*************************************alias table*************************************
/**
* Transform a discrete distribution to a set of binomial distributions
*   so that an O(1) sampling approach can be applied
*/
struct DiscreteSampler1D {
	using Distrib = BinomialDistrib;

	DiscreteSampler1D() = default;
	DiscreteSampler1D(std::vector<float> values) {
		for (const auto& val : values) {
			sumAll += val;
		}
		float sumInv = static_cast<float>(values.size()) / sumAll;

		for (auto& val : values) {
			val *= sumInv;
		}

		binomDistribs.resize(values.size());
		std::vector<Distrib> stackGtOne(values.size() * 2);
		std::vector<Distrib> stackLsOne(values.size() * 2);
		int topGtOne = 0;
		int topLsOne = 0;

		for (int i = 0; i < values.size(); i++) {
			auto& val = values[i];
			(val > static_cast<float>(1) ? stackGtOne[topGtOne++] : stackLsOne[topLsOne++]) = Distrib{ val, i };
		}

		while (topGtOne && topLsOne) {
			Distrib gt = stackGtOne[--topGtOne];
			Distrib ls = stackLsOne[--topLsOne];

			binomDistribs[ls.failId] = Distrib{ ls.prob, gt.failId };
			// Place ls in the table, and "fill" the rest of probability with gt.prob
			gt.prob -= (static_cast<float>(1) - ls.prob);
			// See if gt.prob is still greater than 1 that it needs more iterations to
			//   be splitted to different binomial distributions
			(gt.prob > static_cast<float>(1) ? stackGtOne[topGtOne++] : stackLsOne[topLsOne++]) = gt;
		}

		for (int i = topGtOne - 1; i >= 0; i--) {
			Distrib gt = stackGtOne[i];
			binomDistribs[gt.failId] = gt;
		}

		for (int i = topLsOne - 1; i >= 0; i--) {
			Distrib ls = stackLsOne[i];
			binomDistribs[ls.failId] = ls;
		}
	}

	void Clear() {
		binomDistribs.clear();
		sumAll = 0.0f;
	}

	int Sample(float r1, float r2) {
		int passId = int(float(binomDistribs.size()) * r1);
		Distrib distrib = binomDistribs[passId];

		return (r2 < distrib.prob) ? passId : distrib.failId;
	}

	std::vector<Distrib> binomDistribs;
	float sumAll = 0.0f;
};

struct DevDiscreteSampler1D {
	using Distrib = BinomialDistrib;

	void Create(const DiscreteSampler1D& hstSampler) {
		CUDABuffer hstBuffer;
		hstBuffer.alloc_and_upload(hstSampler.binomDistribs);
		devBinomDistribs = (Distrib*)hstBuffer.d_pointer();
		length = hstSampler.binomDistribs.size();
		sumPower = hstSampler.sumAll;
	}
	void Destory() {
		if (devBinomDistribs != nullptr) {
			cudaFree(devBinomDistribs);
			devBinomDistribs = nullptr;
		}
		length = 0;
	}

	Distrib* devBinomDistribs = nullptr;
	int length = 0;
	float sumPower;
};

__forceinline__ __device__ int SampleAliasTable(const vec2f& sample, int length, BinomialDistrib* devBinomDistribs) {
	float r1 = sample.x;
	float r2 = sample.y;
	int passId = min(int(float(length) * r1), length - 1);
	auto distrib = devBinomDistribs[passId];

	return (r2 < distrib.prob) ? passId : distrib.failId;
}

typedef DevDiscreteSampler1D AliasTable;
//*************************************alias table*************************************