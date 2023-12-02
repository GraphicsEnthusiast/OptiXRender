#include "Scene.h"

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
	if (envMap) {
		delete envMap;
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

void Scene::AddEnvMap(const std::string& fileName) {
	envMap = new Texture(fileName);
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
	unsigned char* image = stbi_load(fileName.c_str(), &res.x, &res.y, &t_comp, STBI_rgb_alpha);
	this->resolution = res;
	this->comp = STBI_rgb_alpha;
	this->pixel = (uint32_t*)image;

	if (image) {
		/* iw - actually, it seems that stbi loads the pictures
		   mirrored along the y axis - mirror them here */
		for (int y = 0; y < res.y / 2; y++) {
			uint32_t* line_y = this->pixel + y * res.x;
			uint32_t* mirrored_y = this->pixel + (res.y - 1 - y) * res.x;
			int mirror_y = res.y - 1 - y;
			for (int x = 0; x < res.x; x++) {
				std::swap(line_y[x], mirrored_y[x]);
			}
		}
	}
}

Texture::~Texture() {
	if (pixel) {
		delete[] pixel;
	}
}
