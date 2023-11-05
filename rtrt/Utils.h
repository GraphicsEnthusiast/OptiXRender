#pragma once

namespace rtrt {
	enum RayType {
		RAY_TYPE_RADIANCE = 0,
		RAY_TYPE_OCCLUSION = 1,
		RAY_TYPE_COUNT
	};

	struct ParallelogramLight {
		float3 corner;
		float3 v1, v2;
		float3 normal;
		float3 emission;
	};

	struct Params {
		unsigned int subframe_index;
		float4* accum_buffer;
		uchar4* frame_buffer;
		unsigned int width;
		unsigned int height;
		unsigned int samples_per_launch;

		float3 eye;
		float3 U;
		float3 V;
		float3 W;

		ParallelogramLight light; // TODO: make light list
		OptixTraversableHandle handle;
	};

	struct RayGenData {

	};

	struct MissData {
		float4 bg_color;
	};

	struct HitGroupData {
		float3  emission_color;
		float3  diffuse_color;
		float4* vertices;
	};
}