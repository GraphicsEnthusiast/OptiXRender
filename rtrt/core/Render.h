#pragma once

#include <glad/glad.h>  // Needs to be included before gl_interop
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sampleConfig.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>
#include <GLFW/glfw3.h>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../Utils.h"

namespace rtrt {
	struct Vertex
	{
		float x, y, z, pad;
	};


	struct IndexedTriangle
	{
		uint32_t v1, v2, v3, pad;
	};


	struct Instance
	{
		float transform[12];
	};

	const int32_t TRIANGLE_COUNT = 32;
	const int32_t MAT_COUNT = 4;
	const static std::array<Vertex, TRIANGLE_COUNT * 3> g_vertices =
	{ {
			// Floor  -- white lambert
			{    0.0f,    0.0f,    0.0f, 0.0f },
			{    0.0f,    0.0f,  559.2f, 0.0f },
			{  556.0f,    0.0f,  559.2f, 0.0f },
			{    0.0f,    0.0f,    0.0f, 0.0f },
			{  556.0f,    0.0f,  559.2f, 0.0f },
			{  556.0f,    0.0f,    0.0f, 0.0f },

			// Ceiling -- white lambert
			{    0.0f,  548.8f,    0.0f, 0.0f },
			{  556.0f,  548.8f,    0.0f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },

			{    0.0f,  548.8f,    0.0f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },
			{    0.0f,  548.8f,  559.2f, 0.0f },

			// Back wall -- white lambert
			{    0.0f,    0.0f,  559.2f, 0.0f },
			{    0.0f,  548.8f,  559.2f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },

			{    0.0f,    0.0f,  559.2f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },
			{  556.0f,    0.0f,  559.2f, 0.0f },

			// Right wall -- green lambert
			{    0.0f,    0.0f,    0.0f, 0.0f },
			{    0.0f,  548.8f,    0.0f, 0.0f },
			{    0.0f,  548.8f,  559.2f, 0.0f },

			{    0.0f,    0.0f,    0.0f, 0.0f },
			{    0.0f,  548.8f,  559.2f, 0.0f },
			{    0.0f,    0.0f,  559.2f, 0.0f },

			// Left wall -- red lambert
			{  556.0f,    0.0f,    0.0f, 0.0f },
			{  556.0f,    0.0f,  559.2f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },

			{  556.0f,    0.0f,    0.0f, 0.0f },
			{  556.0f,  548.8f,  559.2f, 0.0f },
			{  556.0f,  548.8f,    0.0f, 0.0f },

			// Short block -- white lambert
			{  130.0f,  165.0f,   65.0f, 0.0f },
			{   82.0f,  165.0f,  225.0f, 0.0f },
			{  242.0f,  165.0f,  274.0f, 0.0f },

			{  130.0f,  165.0f,   65.0f, 0.0f },
			{  242.0f,  165.0f,  274.0f, 0.0f },
			{  290.0f,  165.0f,  114.0f, 0.0f },

			{  290.0f,    0.0f,  114.0f, 0.0f },
			{  290.0f,  165.0f,  114.0f, 0.0f },
			{  240.0f,  165.0f,  272.0f, 0.0f },

			{  290.0f,    0.0f,  114.0f, 0.0f },
			{  240.0f,  165.0f,  272.0f, 0.0f },
			{  240.0f,    0.0f,  272.0f, 0.0f },

			{  130.0f,    0.0f,   65.0f, 0.0f },
			{  130.0f,  165.0f,   65.0f, 0.0f },
			{  290.0f,  165.0f,  114.0f, 0.0f },

			{  130.0f,    0.0f,   65.0f, 0.0f },
			{  290.0f,  165.0f,  114.0f, 0.0f },
			{  290.0f,    0.0f,  114.0f, 0.0f },

			{   82.0f,    0.0f,  225.0f, 0.0f },
			{   82.0f,  165.0f,  225.0f, 0.0f },
			{  130.0f,  165.0f,   65.0f, 0.0f },

			{   82.0f,    0.0f,  225.0f, 0.0f },
			{  130.0f,  165.0f,   65.0f, 0.0f },
			{  130.0f,    0.0f,   65.0f, 0.0f },

			{  240.0f,    0.0f,  272.0f, 0.0f },
			{  240.0f,  165.0f,  272.0f, 0.0f },
			{   82.0f,  165.0f,  225.0f, 0.0f },

			{  240.0f,    0.0f,  272.0f, 0.0f },
			{   82.0f,  165.0f,  225.0f, 0.0f },
			{   82.0f,    0.0f,  225.0f, 0.0f },

			// Tall block -- white lambert
			{  423.0f,  330.0f,  247.0f, 0.0f },
			{  265.0f,  330.0f,  296.0f, 0.0f },
			{  314.0f,  330.0f,  455.0f, 0.0f },

			{  423.0f,  330.0f,  247.0f, 0.0f },
			{  314.0f,  330.0f,  455.0f, 0.0f },
			{  472.0f,  330.0f,  406.0f, 0.0f },

			{  423.0f,    0.0f,  247.0f, 0.0f },
			{  423.0f,  330.0f,  247.0f, 0.0f },
			{  472.0f,  330.0f,  406.0f, 0.0f },

			{  423.0f,    0.0f,  247.0f, 0.0f },
			{  472.0f,  330.0f,  406.0f, 0.0f },
			{  472.0f,    0.0f,  406.0f, 0.0f },

			{  472.0f,    0.0f,  406.0f, 0.0f },
			{  472.0f,  330.0f,  406.0f, 0.0f },
			{  314.0f,  330.0f,  456.0f, 0.0f },

			{  472.0f,    0.0f,  406.0f, 0.0f },
			{  314.0f,  330.0f,  456.0f, 0.0f },
			{  314.0f,    0.0f,  456.0f, 0.0f },

			{  314.0f,    0.0f,  456.0f, 0.0f },
			{  314.0f,  330.0f,  456.0f, 0.0f },
			{  265.0f,  330.0f,  296.0f, 0.0f },

			{  314.0f,    0.0f,  456.0f, 0.0f },
			{  265.0f,  330.0f,  296.0f, 0.0f },
			{  265.0f,    0.0f,  296.0f, 0.0f },

			{  265.0f,    0.0f,  296.0f, 0.0f },
			{  265.0f,  330.0f,  296.0f, 0.0f },
			{  423.0f,  330.0f,  247.0f, 0.0f },

			{  265.0f,    0.0f,  296.0f, 0.0f },
			{  423.0f,  330.0f,  247.0f, 0.0f },
			{  423.0f,    0.0f,  247.0f, 0.0f },

			// Ceiling light -- emmissive
			{  343.0f,  548.6f,  227.0f, 0.0f },
			{  213.0f,  548.6f,  227.0f, 0.0f },
			{  213.0f,  548.6f,  332.0f, 0.0f },

			{  343.0f,  548.6f,  227.0f, 0.0f },
			{  213.0f,  548.6f,  332.0f, 0.0f },
			{  343.0f,  548.6f,  332.0f, 0.0f }
		} };

	static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = { {
		0, 0,                          // Floor         -- white lambert
		0, 0,                          // Ceiling       -- white lambert
		0, 0,                          // Back wall     -- white lambert
		1, 1,                          // Right wall    -- green lambert
		2, 2,                          // Left wall     -- red lambert
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Short block   -- white lambert
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
		3, 3                           // Ceiling light -- emmissive
	} };



	const std::array<float3, MAT_COUNT> g_emission_colors =
	{ {
		{  0.0f,  0.0f,  0.0f },
		{  0.0f,  0.0f,  0.0f },
		{  0.0f,  0.0f,  0.0f },
		{ 15.0f, 15.0f,  5.0f }

	} };


	const std::array<float3, MAT_COUNT> g_diffuse_colors =
	{ {
		{ 0.80f, 0.80f, 0.80f },
		{ 0.05f, 0.80f, 0.05f },
		{ 0.80f, 0.05f, 0.05f },
		{ 0.50f, 0.00f, 0.00f }
	} };

	template <typename T>
	struct Record
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	typedef Record<RayGenData>   RayGenRecord;
	typedef Record<MissData>     MissRecord;
	typedef Record<HitGroupData> HitGroupRecord;

	namespace CameraController {
		static bool resize_dirty = false;
		static bool minimized = false;

		// Camera state
		static bool camera_changed = true;
		static sutil::Camera camera;
		static sutil::Trackball trackball;

		// Mouse state
		static int32_t mouse_button = -1;

		static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
		static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
		static void WindowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y);
		static void WindowIconifyCallback(GLFWwindow* window, int32_t iconified);
		static void KeyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/);
		static void ScrollCallback(GLFWwindow* window, double xscroll, double yscroll);
	}

	class Render {
	public:
		Render() = default;
		Render(sutil::CUDAOutputBufferType type, int32_t spl, int w, int h) : output_buffer_type(type), samples_per_launch(spl), width(w), height(h) {}

	protected:
		static void ContextLog(unsigned int level, const char* tag, const char* message, void* /*cbdata */);

		void CreateContext();
		void BuildMeshAccel();
		void CreateModule();
		void CreateProgramGroups();
		void CreatePipeline();
		void CreateSBT();
		void InitLaunchParams();
		void Cleanup();
		void InitCamera();
		void InitRender();

		void HandleCameraUpdate();
		void HandleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer);
		void Update(sutil::CUDAOutputBuffer<uchar4>& output_buffer);
		void LaunchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer);
		void DisplaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display);

	public:
		void RenderLoop(const std::string& outfile);

	protected:
		OptixDeviceContext context = 0;

		OptixTraversableHandle gas_handle = 0;  // Traversable handle for triangle AS
		CUdeviceptr d_gas_output_buffer = 0;  // Triangle AS memory
		CUdeviceptr d_vertices = 0;

		OptixModule ptx_module = 0;
		OptixPipelineCompileOptions pipeline_compile_options = {};
		OptixPipeline pipeline = 0;

		OptixProgramGroup raygen_prog_group = 0;
		OptixProgramGroup radiance_miss_group = 0;
		OptixProgramGroup occlusion_miss_group = 0;
		OptixProgramGroup radiance_hit_group = 0;
		OptixProgramGroup occlusion_hit_group = 0;

		CUstream stream = 0;
		Params params;
		Params* d_params;

		OptixShaderBindingTable sbt = {};

		GLFWwindow* window = NULL;
		sutil::CUDAOutputBufferType output_buffer_type;
		int32_t samples_per_launch;
		int width, height;
	};
}