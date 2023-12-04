#include "Scene.h"
#include "../cuda/Utils.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

Scene::~Scene() {
	for (auto mesh : meshes) {
		delete mesh;
	}
	for (auto texture : textures) {
		delete texture;
	}
	if (env) {
		delete env;
	}
}

void Scene::AddMesh(const std::string& objFile, Material& material, const TextureName& textureName) {
	TriangleMesh* mesh = new TriangleMesh();

	if (textureName.albedoFile != "") {
		material.albedoTextureID = this->textures.size();
		AddTexture(textureName.albedoFile);
	}
	if (textureName.anisotropyFile != "") {
		material.anisotropyTextureID = this->textures.size();
		AddTexture(textureName.anisotropyFile);
	}
	if (textureName.roughnessFile != "") {
		material.roughnessTextureID = this->textures.size();
		AddTexture(textureName.roughnessFile);
	}
	if (textureName.specularFile != "") {
		material.specularTextureID = this->textures.size();
		AddTexture(textureName.specularFile);
	}

	mesh->LoadMesh(objFile, material);
	this->meshes.emplace_back(mesh);

	for (auto vtx : mesh->vertex) {
		this->bounds.extend(vtx);
	}
}

void Scene::AddTexture(const std::string& fileName) {
	Texture* texture = new Texture(fileName);
	textures.emplace_back(texture);
}

void Scene::AddLight(const Light& l) {
	lights.emplace_back(l);
}

void Scene::AddEnv(const std::string& fileName) {
	env = new HdrTexture(fileName);
}

void TriangleMesh::LoadMesh(const std::string& objFile, const Material& material) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFile.c_str()) || shapes.size() == 0) {
		throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
	}

	this->material = material;

	// loop over shapes
	for (size_t s = 0; s < shapes.size(); ++s) {
		size_t index_offset = 0;
		// loop over faces
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
			const size_t fv =
				static_cast<size_t>(shapes[s].mesh.num_face_vertices[f]);

			std::vector<vec3f> vertices;
			std::vector<vec3f> normals;
			std::vector<vec2f> texcoords;

			// loop over vertices
			// get vertices, normals, texcoords of a triangle
			for (size_t v = 0; v < fv; ++v) {
				const tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				const tinyobj::real_t vx =
					attrib.vertices[3 * static_cast<size_t>(idx.vertex_index) + 0];
				const tinyobj::real_t vy =
					attrib.vertices[3 * static_cast<size_t>(idx.vertex_index) + 1];
				const tinyobj::real_t vz =
					attrib.vertices[3 * static_cast<size_t>(idx.vertex_index) + 2];

				// 				vec4 v_object(vx, vy, vz, 1.0f);
				// 				vec3 v_world = transform * v_object;
				vertices.emplace_back(vec3f(vx, vy, vz));

				if (idx.normal_index >= 0) {
					const tinyobj::real_t nx =
						attrib.normals[3 * static_cast<size_t>(idx.normal_index) + 0];
					const tinyobj::real_t ny =
						attrib.normals[3 * static_cast<size_t>(idx.normal_index) + 1];
					const tinyobj::real_t nz =
						attrib.normals[3 * static_cast<size_t>(idx.normal_index) + 2];

					// 					vec4 n_object(nx, ny, nz, 0.0f);
					// 					vec3 n_world = transform * n_object;
					normals.emplace_back(normalize(vec3f(nx, ny, nz)));
				}

				if (idx.texcoord_index >= 0) {
					const tinyobj::real_t tx =
						attrib
						.texcoords[2 * static_cast<size_t>(idx.texcoord_index) + 0];
					const tinyobj::real_t ty =
						attrib
						.texcoords[2 * static_cast<size_t>(idx.texcoord_index) + 1];
					texcoords.emplace_back(vec2f(tx, ty));
				}
			}

			// if normals is empty, add geometric normal
			if (normals.size() == 0) {
				const vec3f v1 = normalize(vertices[1] - vertices[0]);
				const vec3f v2 = normalize(vertices[2] - vertices[0]);
				const vec3f n = normalize(cross(v1, v2));
				normals.emplace_back(n);
				normals.emplace_back(n);
				normals.emplace_back(n);
			}

			// if texcoords is empty, add barycentric coords
			if (texcoords.size() == 0) {
				texcoords.emplace_back(vec2f(0.0f));
				texcoords.emplace_back(vec2f(1.0f, 0.0f));
				texcoords.emplace_back(vec2f(0.0f, 1.0f));
			}

			for (int i = 0; i < 3; ++i) {
				this->vertex.emplace_back(vertices[i]);
				this->normal.emplace_back(normals[i]);
				this->texcoord.emplace_back(texcoords[i]);
			}

			int index = this->index.size();
			this->index.emplace_back(vec3i(3 * index, 3 * index + 1, 3 * index + 2));

			index_offset += fv;
		}
	}
}

Texture::Texture(const std::string& fileName) {
	vec2i res;
	int t_comp;
	stbi_set_flip_vertically_on_load(true);
	uint8_t* image = stbi_load(fileName.c_str(), &res.x, &res.y, &t_comp, STBI_rgb_alpha);
	this->resolution = res;
	this->comp = STBI_rgb_alpha;
	this->pixel = image;
}

Texture::~Texture() {
	if (pixel) {
		delete[] pixel;
	}
}

HdrTexture::HdrTexture(const std::string& fileName) {
	stbi_set_flip_vertically_on_load(true);
	float* data = stbi_loadf(fileName.c_str(), &width, &height, &comp, STBI_rgb_alpha);
	if (data) {
		hdr = data;
		comp = STBI_rgb_alpha;
		CalculateHdrCache();
		CreateCudaTexture();
	}
}

HdrTexture::~HdrTexture() {
	if (hdr) {
		delete[] hdr;
	}
	if (cache) {
		delete[] cache;
	}
}

void HdrTexture::CalculateHdrCache() {
	float lumSum = 0.0f;

	//初始化h行w列的概率密度pdf并统计总亮度
	std::vector<std::vector<float>> pdf(height);
	for (auto& line : pdf) line.resize(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float R = hdr[comp * (i * width + j)];
			float G = hdr[comp * (i * width + j) + 1];
			float B = hdr[comp * (i * width + j) + 2];
			float lum = Luminance(vec3f(R, G, B));
			pdf[i][j] = lum;
			lumSum += lum;
		}
	}

	//概率密度归一化
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			pdf[i][j] /= lumSum;
		}
	}

	//累加每一列得到x的边缘概率密度
	std::vector<float> pdf_x_margin;
	pdf_x_margin.resize(width);
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			pdf_x_margin[j] += pdf[i][j];
		}
	}

	//计算x的边缘分布函数
	std::vector<float> cdf_x_margin = pdf_x_margin;
	for (int i = 1; i < width; i++) {
		cdf_x_margin[i] += cdf_x_margin[i - 1];
	}

	//计算y在X=x下的条件概率密度函数
	std::vector<std::vector<float>> pdf_y_condiciton = pdf;
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			pdf_y_condiciton[i][j] /= pdf_x_margin[j];
		}
	}

	//计算y在X=x下的条件概率分布函数
	std::vector<std::vector<float>> cdf_y_condiciton = pdf_y_condiciton;
	for (int j = 0; j < width; j++) {
		for (int i = 1; i < height; i++) {
			cdf_y_condiciton[i][j] += cdf_y_condiciton[i - 1][j];
		}
	}

	//cdf_y_condiciton转置为按列存储
	//cdf_y_condiciton[i]表示y在X=i下的条件概率分布函数
	std::vector<std::vector<float>> temp = cdf_y_condiciton;
	cdf_y_condiciton = std::vector<std::vector<float>>(width);
	for (auto& line : cdf_y_condiciton) {
		line.resize(height);
	}
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			cdf_y_condiciton[j][i] = temp[i][j];
		}
	}

	//穷举 xi_1, xi_2 预计算样本 xy
	//sample_x[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 x
	//sample_y[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 y
	//sample_p[i][j] 表示取 (i, j) 点时的概率密度
	std::vector<std::vector<float>> sample_x(height);
	for (auto& line : sample_x) {
		line.resize(width);
	}
	std::vector<std::vector<float>> sample_y(height);
	for (auto& line : sample_y) {
		line.resize(width);
	}
	std::vector<std::vector<float>> sample_p(height);
	for (auto& line : sample_p) {
		line.resize(width);
	}
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < height; i++) {
			float xi_1 = float(i) / height;
			float xi_2 = float(j) / width;

			//用xi_1在cdf_x_margin中lower bound得到样本x
			int x = std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) - cdf_x_margin.begin();
			//用xi_2在X=x的情况下得到样本y
			int y = std::lower_bound(cdf_y_condiciton[x].begin(), cdf_y_condiciton[x].end(), xi_2) - cdf_y_condiciton[x].begin();

			//存储纹理坐标xy和xy位置对应的概率密度
			sample_x[i][j] = float(x) / width;
			sample_y[i][j] = float(y) / height;
			sample_p[i][j] = pdf[i][j];
		}
	}

	//整合结果到纹理
	//R,G 通道存储样本(x,y)而B通道存储pdf(i, j)
	cache = new float[width * height * comp];
	//for (int i = 0; i < width * height * 3; i++) cache[i] = 0.0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cache[comp * (i * width + j)] = sample_x[i][j];        //R
			cache[comp * (i * width + j) + 1] = sample_y[i][j];    //G
			cache[comp * (i * width + j) + 2] = sample_p[i][j];    //B
		}
	}
}

void HdrTexture::CreateCudaTexture() {
	{
		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = this->width;
		int32_t height = this->height;
		int32_t numComponents = this->comp;
		int32_t pitch = width * numComponents * sizeof(float);
		channel_desc = cudaCreateChannelDesc<float4>();

		cudaArray_t pixelArray;
		CUDA_CHECK(MallocArray(&pixelArray,
			&channel_desc,
			width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
			/* offset */0, 0,
			hdr,
			pitch, pitch, height,
			cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		CUDA_CHECK(CreateTextureObject(&cuda_texture_hdr, &res_desc, &tex_desc, nullptr));
	}
	{
		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = this->width;
		int32_t height = this->height;
		int32_t numComponents = this->comp;
		int32_t pitch = width * numComponents * sizeof(float);
		channel_desc = cudaCreateChannelDesc<float4>();

		cudaArray_t pixelArray;
		CUDA_CHECK(MallocArray(&pixelArray,
			&channel_desc,
			width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
			/* offset */0, 0,
			cache,
			pitch, pitch, height,
			cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		CUDA_CHECK(CreateTextureObject(&cuda_texture_cache, &res_desc, &tex_desc, nullptr));
	}
}