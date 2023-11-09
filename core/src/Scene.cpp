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
}

void Scene::LoadMesh(const std::string& objFile) {
	TriangleMesh* mesh = new TriangleMesh(objFile);
	this->meshes.emplace_back(mesh);
	// of course, you should be using tbb::parallel_for for stuff
	// like this:
	for (auto vtx : mesh->vertex) {
		this->bounds.extend(vtx);
	}
}

TriangleMesh::TriangleMesh(const std::string& objFile) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFile.c_str()) || shapes.size() == 0) {
		throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
	}

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
				this->vertex.emplace_back(vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
				this->normal.emplace_back(vec3f(normals[i][0], normals[i][1], normals[i][2]));
				this->texcoord.emplace_back(vec2f(texcoords[i][0], texcoords[i][1]));
			}

			int index = this->index.size();
			this->index.emplace_back(vec3i(3 * index, 3 * index + 1, 3 * index + 2));

			this->diffuse = vec3f(0.5f);
			this->diffuseTextureID = -1;

			index_offset += fv;
		}
	}
}

Texture::Texture() {

}

Texture::~Texture() {
	if (pixel) {
		delete[] pixel;
	}
}
