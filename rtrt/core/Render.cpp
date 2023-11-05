#include "Render.h"

#include <optix_function_table_definition.h>

using namespace rtrt;

void CameraController::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS) {
		mouse_button = button;
		trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else {
		mouse_button = -1;
	}
}

void CameraController::CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
		trackball.setViewMode(sutil::Trackball::LookAtFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT) {
		trackball.setViewMode(sutil::Trackball::EyeFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
}

void CameraController::WindowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y) {
	// Keep rendering at the current resolution when the window is minimized.
	if (minimized) {
		return;
	}

	// Output dimensions must be at least 1 in both x and y.
	sutil::ensureMinimumSize(res_x, res_y);

	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
	params->width = res_x;
	params->height = res_y;
	camera_changed = true;
	resize_dirty = true;
}

void CameraController::WindowIconifyCallback(GLFWwindow* window, int32_t iconified) {
	minimized = (iconified > 0);
}

void CameraController::KeyCallback(GLFWwindow* window, int32_t key, int32_t, int32_t action, int32_t) {
	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
			glfwSetWindowShouldClose(window, true);
		}
	}
	else if (key == GLFW_KEY_G) {
		// toggle UI draw
	}
}

void CameraController::ScrollCallback(GLFWwindow* window, double xscroll, double yscroll) {
	if (trackball.wheelEvent((int)yscroll)) {
		camera_changed = true;
	}
}

void Render::ContextLog(unsigned int level, const char* tag, const char* message, void*) {
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void Render::CreateContext() {
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	CUcontext cu_ctx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &ContextLog;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
}

void Render::BuildMeshAccel() {
	//
	// copy mesh data to device
	//
	const size_t vertices_size_in_bytes = g_vertices.size() * sizeof(Vertex);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_vertices),
		g_vertices.data(), vertices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr d_mat_indices = 0;
	const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof(uint32_t);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_mat_indices),
		g_mat_indices.data(),
		mat_indices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	//
	// Build triangle GAS
	//
	// One per SBT record for this build input
	uint32_t triangle_input_flags[MAT_COUNT] = {
		OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
		OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
		OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
		OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
	};

	OptixBuildInput triangle_input = {};
	triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
	triangle_input.triangleArray.numVertices = static_cast<uint32_t>(g_vertices.size());
	triangle_input.triangleArray.vertexBuffers = &d_vertices;
	triangle_input.triangleArray.flags = triangle_input_flags;
	triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
	triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
	triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context,
		&accel_options,
		&triangle_input,
		1,  // num_build_inputs
		&gas_buffer_sizes
	));

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

	// non-compacted output
	CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
	size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
		compactedSizeOffset + 8
	));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(
		context,
		0,                                  // CUDA stream
		&accel_options,
		&triangle_input,
		1,                                  // num build inputs
		d_temp_buffer,
		gas_buffer_sizes.tempSizeInBytes,
		d_buffer_temp_output_gas_and_compacted_size,
		gas_buffer_sizes.outputSizeInBytes,
		&gas_handle,
		&emitProperty,                      // emitted property list
		1                                   // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

		CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
	}
	else {
		d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
	}
}

void Render::CreateModule() {
	OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
	pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	size_t inputSize = 0;
	const char* input = sutil::getInputData("rtrt", "rtrt/cuda/", "pt.cu", inputSize);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
		context,
		&module_compile_options,
		&pipeline_compile_options,
		input,
		inputSize,
		log,
		&sizeof_log,
		&ptx_module
	));
}

void Render::CreateProgramGroups() {
	OptixProgramGroupOptions  program_group_options = {};

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OptixProgramGroupDesc raygen_prog_group_desc = {};
	raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygen_prog_group_desc.raygen.module = ptx_module;
	raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context, &raygen_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&raygen_prog_group
	));
	
	OptixProgramGroupDesc miss_prog_group_desc = {};
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = ptx_module;
	miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context, &miss_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log, &sizeof_log,
		&radiance_miss_group
	));

	memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
	miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_prog_group_desc.miss.module = nullptr;  // NULL miss program for occlusion rays
	miss_prog_group_desc.miss.entryFunctionName = nullptr;
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context, &miss_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&occlusion_miss_group
	));
	
	OptixProgramGroupDesc hit_prog_group_desc = {};
	hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
	hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		context,
		&hit_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&radiance_hit_group
	));

	memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
	hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
	hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
	sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(
		context,
		&hit_prog_group_desc,
		1,  // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		&occlusion_hit_group
	));
}

void Render::CreatePipeline() {
	OptixProgramGroup program_groups[] = {
		raygen_prog_group,
		radiance_miss_group,
		occlusion_miss_group,
		radiance_hit_group,
		occlusion_hit_group
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = 2;
	pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		context,
		&pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		log,
		&sizeof_log,
		&pipeline
	));

	// We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
	// parameters to optixPipelineSetStackSize.
	OptixStackSizes stack_sizes = {};
	OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_prog_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(radiance_miss_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(occlusion_miss_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(radiance_hit_group, &stack_sizes));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(occlusion_hit_group, &stack_sizes));

	uint32_t max_trace_depth = 2;
	uint32_t max_cc_depth = 0;
	uint32_t max_dc_depth = 0;
	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stack_sizes,
		max_trace_depth,
		max_cc_depth,
		max_dc_depth,
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size
	));

	const uint32_t max_traversal_depth = 1;
	OPTIX_CHECK(optixPipelineSetStackSize(
		pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		max_traversal_depth
	));
}

void Render::CreateSBT() {
	CUdeviceptr  d_raygen_record;
	const size_t raygen_record_size = sizeof(RayGenRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

	RayGenRecord rg_sbt = {};
	OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_raygen_record),
		&rg_sbt,
		raygen_record_size,
		cudaMemcpyHostToDevice
	));


	CUdeviceptr d_miss_records;
	const size_t miss_record_size = sizeof(MissRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

	MissRecord ms_sbt[2];
	OPTIX_CHECK(optixSbtRecordPackHeader(radiance_miss_group, &ms_sbt[0]));
	ms_sbt[0].data.bg_color = make_float4(0.0f);
	OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_miss_group, &ms_sbt[1]));
	ms_sbt[1].data.bg_color = make_float4(0.0f);

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_miss_records),
		ms_sbt,
		miss_record_size * RAY_TYPE_COUNT,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr  d_hitgroup_records;
	const size_t hitgroup_record_size = sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_hitgroup_records),
		hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
	));

	HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
	for (int i = 0; i < MAT_COUNT; ++i) {
		const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

		OPTIX_CHECK(optixSbtRecordPackHeader(radiance_hit_group, &hitgroup_records[sbt_idx]));
		hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
		hitgroup_records[sbt_idx].data.diffuse_color = g_diffuse_colors[i];
		hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(d_vertices);
		
		const int sbt_idx2 = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
		memset(&hitgroup_records[sbt_idx2], 0, hitgroup_record_size);

		OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_hit_group, &hitgroup_records[sbt_idx2]));
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroup_records),
		hitgroup_records,
		hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT,
		cudaMemcpyHostToDevice
	));

	sbt.raygenRecord = d_raygen_record;
	sbt.missRecordBase = d_miss_records;
	sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
	sbt.missRecordCount = RAY_TYPE_COUNT;
	sbt.hitgroupRecordBase = d_hitgroup_records;
	sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
	sbt.hitgroupRecordCount = RAY_TYPE_COUNT * MAT_COUNT;
}

void Render::InitLaunchParams() {
	params.width = width;
	params.height = height;

	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.accum_buffer),
		params.width * params.height * sizeof(float4)
	));
	params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

	params.samples_per_launch = samples_per_launch;
	params.subframe_index = 0u;

	params.light.emission = make_float3(15.0f, 15.0f, 5.0f);
	params.light.corner = make_float3(343.0f, 548.5f, 227.0f);
	params.light.v1 = make_float3(0.0f, 0.0f, 105.0f);
	params.light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
	params.light.normal = normalize(cross(params.light.v1, params.light.v2));
	params.handle = gas_handle;

	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

void Render::Cleanup() {
	OPTIX_CHECK(optixPipelineDestroy(pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(radiance_miss_group));
	OPTIX_CHECK(optixProgramGroupDestroy(radiance_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(occlusion_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(occlusion_miss_group));
	OPTIX_CHECK(optixModuleDestroy(ptx_module));
	OPTIX_CHECK(optixDeviceContextDestroy(context));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

void Render::InitCamera() {
	CameraController::camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
	CameraController::camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
	CameraController::camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	CameraController::camera.setFovY(35.0f);
	CameraController::camera_changed = true;

	CameraController::trackball.setCamera(&CameraController::camera);
	CameraController::trackball.setMoveSpeed(10.0f);
	CameraController::trackball.setReferenceFrame(
		make_float3(1.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 1.0f),
		make_float3(0.0f, 1.0f, 0.0f)
	);
	CameraController::trackball.setGimbalLock(true);
}

void Render::InitRender() {
	InitCamera();
	CreateContext();
 	BuildMeshAccel();
 	CreateModule();
	CreateProgramGroups();
	CreatePipeline();
	CreateSBT();
	InitLaunchParams();
}

void Render::HandleCameraUpdate() {
	if (!CameraController::camera_changed) {
		return;
	}
	CameraController::camera_changed = false;

	CameraController::camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
	params.eye = CameraController::camera.eye();
	CameraController::camera.UVWFrame(params.U, params.V, params.W);
}

void Render::HandleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer) {
	if (!CameraController::resize_dirty) {
		return;
	}
	CameraController::resize_dirty = false;

	output_buffer.resize(params.width, params.height);

	// Realloc accumulation buffer
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.accum_buffer),
		params.width * params.height * sizeof(float4)
	));
}

void Render::Update(sutil::CUDAOutputBuffer<uchar4>& output_buffer) {
	// Update params on device
	if (CameraController::camera_changed || CameraController::resize_dirty) {
		params.subframe_index = 0;
	}

	HandleCameraUpdate();
	HandleResize(output_buffer);
}

void Render::LaunchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer) {
	// Launch
	uchar4* result_buffer_data = output_buffer.map();
	params.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(
		reinterpret_cast<void*>(d_params),
		&params, sizeof(Params),
		cudaMemcpyHostToDevice, stream
	));

	OPTIX_CHECK(optixLaunch(
		pipeline,
		stream,
		reinterpret_cast<CUdeviceptr>(d_params),
		sizeof(Params),
		&sbt,
		params.width,   // launch width
		params.height,  // launch height
		1               // launch depth
	));
	output_buffer.unmap();
	CUDA_SYNC_CHECK();
}

void Render::DisplaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display) {
	// Display
	int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;  //
	glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		output_buffer.width(),
		output_buffer.height(),
		framebuf_res_x,
		framebuf_res_y,
		output_buffer.getPBO()
	);
}

void Render::RenderLoop(const std::string& outfile) {
	try {
        InitRender();

		if (outfile.empty()) {
			window = sutil::initUI("rtrt", params.width, params.height);
			glfwSetMouseButtonCallback(window, CameraController::MouseButtonCallback);
			glfwSetCursorPosCallback(window, CameraController::CursorPosCallback);
			glfwSetWindowSizeCallback(window, CameraController::WindowSizeCallback);
			glfwSetWindowIconifyCallback(window, CameraController::WindowIconifyCallback);
			glfwSetKeyCallback(window, CameraController::KeyCallback);
			glfwSetScrollCallback(window, CameraController::ScrollCallback);
			glfwSetWindowUserPointer(window, &params);

			//
			// Render loop
			//
			sutil::CUDAOutputBuffer<uchar4> output_buffer(
				output_buffer_type,
				params.width,
				params.height
			);

			output_buffer.setStream(stream);
			sutil::GLDisplay gl_display;

			std::chrono::duration<double> state_update_time(0.0);
			std::chrono::duration<double> render_time(0.0);
			std::chrono::duration<double> display_time(0.0);

			do {
				auto t0 = std::chrono::steady_clock::now();
				glfwPollEvents();

				Update(output_buffer);
				auto t1 = std::chrono::steady_clock::now();
				state_update_time += t1 - t0;
				t0 = t1;

				LaunchSubframe(output_buffer);
				t1 = std::chrono::steady_clock::now();
				render_time += t1 - t0;
				t0 = t1;

				DisplaySubframe(output_buffer, gl_display);
				t1 = std::chrono::steady_clock::now();
				display_time += t1 - t0;

				sutil::displayStats(state_update_time, render_time, display_time);

				glfwSwapBuffers(window);

				++params.subframe_index;
			} while (!glfwWindowShouldClose(window));
			CUDA_SYNC_CHECK();
		}
		else {
			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP) {
				sutil::initGLFW();  // For GL context
				sutil::initGL();
			}

			sutil::CUDAOutputBuffer<uchar4> output_buffer(
				output_buffer_type,
				params.width,
				params.height
			);

			HandleCameraUpdate();
			HandleResize(output_buffer);
			LaunchSubframe(output_buffer);

			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = output_buffer.width();
			buffer.height = output_buffer.height();
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

			sutil::saveImage(outfile.c_str(), buffer, false);

			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP) {
				glfwTerminate();
			}
		}

		sutil::cleanupUI(window);
		Cleanup();
	}
	catch (std::exception& e){
		std::cerr << "Caught exception: " << e.what() << "\n";
	}
}
