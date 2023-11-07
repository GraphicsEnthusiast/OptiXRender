#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

enum struct LightType {
	None = 0,
	Envmap = 1,
	Rect = 2,
	Sphere = 3,
};

struct Light {
	//光源类型
	LightType type = LightType::None;

	//面光源radiance
	vec3f radiance{ 1.0f };

	//球心/矩形的起始顶点
	vec3f position{ 0.0f };

	//矩形光宽高向量
	vec3f du{ 1.0f };
	vec3f dv{ 1.0f };

	//球半径
	float radius = 1.0f;

	//HDR
	Texture* texture = nullptr;

	//面光源是否双面
	bool doubleSided = false;
};

class Scene {
public:
	Scene(const Model* model, const Light& light) :
		model(model), light(light) {}

	~Scene() {
		if (model != NULL) {
			delete model;
			model = NULL;
		}
	}

public:
	/*! the model we are going to trace rays against */
	const Model* model;
	const Light light;
};