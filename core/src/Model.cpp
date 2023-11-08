#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

Model* LoadOBJ(const std::string& objFile) {
    Model* model = new Model;

    model->meshes.emplace_back(LoadFromObj(objFile));
    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes) {
        for (auto vtx : mesh->vertex) {
            model->bounds.extend(vtx);
        }
    }

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;

    return model;
}

TriangleMesh* LoadFromObj(const std::string& filepath) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str()) || shapes.size() == 0) {
        throw std::runtime_error("Could not read OBJ model from " + filepath + " : " + err);
	}

    TriangleMesh* mesh = new TriangleMesh;
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
                mesh->vertex.emplace_back(vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
                mesh->normal.emplace_back(vec3f(normals[i][0], normals[i][1], normals[i][2]));
                mesh->texcoord.emplace_back(vec2f(texcoords[i][0], texcoords[i][1]));
			}

            int index = mesh->index.size();
            mesh->index.emplace_back(vec3i(3 * index, 3 * index + 1, 3 * index + 2));

            mesh->diffuse = vec3f(0.5f);
            mesh->diffuseTextureID = -1;

			index_offset += fv;
		}
	}

    return mesh;
}
