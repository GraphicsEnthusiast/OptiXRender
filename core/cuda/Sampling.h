#pragma once

#include "Utils.h"
#include <vector>
#include <queue>

__forceinline__ __device__ float PowerHeuristic(float pdf1, float pdf2, int beta) {
	float p1 = pow(pdf1, beta);
	float p2 = pow(pdf2, beta);

	return p1 / (p1 + p2);
}

//*************************************cone*************************************
__forceinline__ __device__ vec3f UniformSampleCone(const vec2f& sample, float cos_angle) {
	vec3f p = 0.0f;

	float phi = M_2PIf * sample.x;
	float cos_theta = mix(cos_angle, 1.0f, sample.y);
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	p.x = sin_theta * cosf(phi);
	p.y = sin_theta * sinf(phi);
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
__host__ struct AliasTable1D {
public:
    AliasTable1D() = default;
    AliasTable1D(const std::vector<float>& distrib) {
		std::queue<Element> greater, lesser;

	    sumDistrib = 0.0f;
	    for (auto i : distrib) {
		    sumDistrib += i;
	    }

	    for (int i = 0; i < distrib.size(); i++) {
		    float scaledPdf = distrib[i] * distrib.size();
		    (scaledPdf >= sumDistrib ? greater : lesser).push(Element(i, scaledPdf));
	    }

	    table.resize(distrib.size(), Element(-1, 0.0f));

	    while (!greater.empty() && !lesser.empty()) {
		    int l = lesser.front().first;
			float pl = lesser.front().second;
		    lesser.pop();

			int g = greater.front().first;
			float pg = greater.front().second;
		    greater.pop();

		    table[l] = Element(g, pl);

		    pg += pl - sumDistrib;
		    (pg < sumDistrib ? lesser : greater).push(Element(g, pg));
	    }

	    while (!greater.empty()) {
		    int g = greater.front().first;
			float pg = greater.front().second;
		    greater.pop();
		    table[g] = Element(g, pg);
	    }

	    while (!lesser.empty()) {
		    int l = lesser.front().first;
			float pl = lesser.front().second;
		    lesser.pop();
		    table[l] = Element(l, pl);
	    }
	}

public:
    typedef std::pair<int, float> Element;
	std::vector<Element> table;
	float sumDistrib;
};

__host__ struct AliasTable2D {
public:
    AliasTable2D(float* pdf, int width, int height) {
		std::vector<float> colDistrib(height);
	    for (int i = 0; i < height; i++) {
		    std::vector<float> table(pdf + i * width, pdf + (i + 1) * width);
		    AliasTable1D rowDistrib(table);
		    rowTables.emplace_back(rowDistrib);
		    colDistrib[i] = rowDistrib.sumDistrib;
	    }
	    colTable = AliasTable1D(colDistrib);

        colAlia = (int*)malloc(height);
		colProb = (float*)malloc(height);
		rowAlia = (int*)malloc(width * height);
		rowProb = (float*)malloc(width * height);

		for (int i = 0; i < height; i++) {
			colAlia[i] = colTable.table[i].first;
			colProb[i] = colTable.table[i].second;
			for (int j = 0; j < width; j++) {
				int index = i * width + j;
				rowAlia[index] = rowTables[i].table[j].first;
				rowProb[index] = rowTables[i].table[j].second;
			}
		}
	}

	~AliasTable2D() {
		if (rowAlia) {
			free(rowAlia);
		}
		if (rowProb) {
			free(rowProb);
		}
		if (colAlia) {
			free(colAlia);
		}
		if (colProb) {
			free(colProb);
		}
	}

public:
    std::vector<AliasTable1D> rowTables;// 行
	AliasTable1D colTable;// 列
	int* rowAlia = NULL;
	float* rowProb = NULL;
	int* colAlia = NULL;
	float* colProb = NULL;
};
//*************************************alias table*************************************