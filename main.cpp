#include "Renderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

struct MyWindow : public GLFCameraWindow {
	MyWindow(const std::string& title, const Scene* scene, const Camera& camera, const float worldScale)
		: GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
		renderer(scene) {
		renderer.SetCamera(camera);
		camera_medium = camera.medium;
	}

	virtual void render() override {
		if (cameraFrame.modified) {
			renderer.SetCamera(Camera{ cameraFrame.get_from(),
									 cameraFrame.get_at(),
									 cameraFrame.get_up(),
									 camera_medium });
			cameraFrame.modified = false;
		}
		renderer.Render();
	}
	virtual void draw() override {
		renderer.DownloadPixels(pixels.data());
		if (fbTexture == 0) {
			glGenTextures(1, &fbTexture);
		}

		glBindTexture(GL_TEXTURE_2D, fbTexture);
		GLenum texFormat = GL_RGBA;
		GLenum texelType = GL_UNSIGNED_BYTE;
		glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
			texelType, pixels.data());

		glDisable(GL_LIGHTING);
		glColor3f(1, 1, 1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glDisable(GL_DEPTH_TEST);

		glViewport(0, 0, fbSize.x, fbSize.y);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, (float)fbSize.y, 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f((float)fbSize.x, 0.f, 0.f);
		}
		glEnd();
	}
	virtual void resize(const vec2i& newSize) {
		fbSize = newSize;
		renderer.Resize(newSize);
		pixels.resize(newSize.x * newSize.y);
	}
	virtual void key(int key, int mods) {
		if (key == 'D' || key == ' ' || key == 'd') {
			renderer.denoiserOn = !renderer.denoiserOn;
			std::cout << "denoising now " << (renderer.denoiserOn ? "ON" : "OFF") << std::endl;
		}
		if (key == 'A' || key == 'a') {
			renderer.accumulate = !renderer.accumulate;
			std::cout << "accumulation/progressive refinement now " << (renderer.accumulate ? "ON" : "OFF") << std::endl;
		}
		if (key == ',') {
			renderer.launchParams.numPixelSamples
				= std::max(1, renderer.launchParams.numPixelSamples - 1);
			std::cout << "num samples/pixel now "
				<< renderer.launchParams.numPixelSamples << std::endl;
		}
		if (key == '.') {
			renderer.launchParams.numPixelSamples
				= std::max(1, renderer.launchParams.numPixelSamples + 1);
			std::cout << "num samples/pixel now "
				<< renderer.launchParams.numPixelSamples << std::endl;
		}
	}

	vec2i fbSize;
	GLuint fbTexture{ 0 };
	Renderer renderer;
	std::vector<uint32_t> pixels;
	int camera_medium;
};

extern "C" int main(int ac, char** av) {
	TextureFile textureFile;
	Material material;
	Medium m;
	textureFile.albedoFile = "../../models/01_Head_Base_Color.png";
	textureFile.roughnessFile = "../../models/01_Head_Roughness.png";
	textureFile.metallicFile = "../../models/01_Head_Metallic.png";
	textureFile.normalFile = "../../models/01_Head_Normal_DirectX.png";
	try {
		Scene scene;
		scene.AddMedium(m);
		scene.AddMesh(
			"../../models/head.obj",
			material,
			textureFile,
			-1, -1
		);
		textureFile.albedoFile = "../../models/02_Body_Base_Color.png";
		textureFile.roughnessFile = "../../models/02_Body_Roughness.png";
		textureFile.metallicFile = "../../models/02_Body_Metallic.png";
		textureFile.normalFile = "../../models/02_Body_Normal_DirectX.png";

		Material m;
		m.type = MaterialType::ThinDielectric;
		//m.roughness = 0.5f;
		TextureFile t;
		scene.AddMesh(
			"../../models/body.obj",
			m,
			t,
			-1, 0
		);
		textureFile.albedoFile = "../../models/03_Base_Base_Color.png";
		textureFile.metallicFile = "../../models/03_Base_Metallic.png";
		textureFile.roughnessFile = "../../models/03_Base_Roughness.png";
		textureFile.normalFile = "../../models/03_Base_Normal_DirectX.png";
		scene.AddMesh(
			"../../models/base.obj",
			material,
			textureFile,
			-1, -1
		);
		material = Material();
		material.type = MaterialType::Diffuse;
		textureFile.albedoFile = "../../models/grid.jpg";
		textureFile.roughnessFile = "../../models/rusty_metal_sheet_rough_1k.png";
		textureFile.normalFile = "";
		scene.AddMesh(
			"../../models/plane.obj",
			material,
			textureFile,
			-1, -1
		);
		scene.AddEnv("../../models/spaichingen_hill_4k.hdr");
		Camera camera = { /*from*/vec3f(0.2f, 0.2f, 0.2f),
			/* at */scene.bounds.center(),
			/* up */vec3f(0.f,1.f,0.f) };
		camera.medium = -1;

		// something approximating the scale of the world, so the
		// camera knows how much to move for any given user interaction:
		const float worldScale = length(scene.bounds.span());
		Light light;
		light.medium = -1;
		light.position = vec3f(0.5f, 0.25f, 0.0f);
		light.radius = 0.1f;
		light.radiance = vec3f(15.0f);
		//scene.AddLight(light);

		light.position = vec3f(1.0f, 0.5f, 0.0f);
		light.radius = 0.08f;
		light.radiance = vec3f(0.0f, 15.0f, 0.0f);
		//scene.AddLight(light);

		light.position = vec3f(-0.5f, 0.25f, 0.0f);
		light.radius = 0.1f;
		light.radiance = vec3f(15.0f, 0.0f, 0.0f);
		//scene.AddLight(light);

		MyWindow* window = new MyWindow("RTX_Render",
			&scene, camera, worldScale);
		//      window->enableFlyMode();
		window->enableInspectMode();

		std::cout << "Press 'a' to enable/disable accumulation/progressive refinement" << std::endl;
		std::cout << "Press ' ' to enable/disable denoising" << std::endl;
		std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
		std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
		window->run();

	}
	catch (std::runtime_error& e) {
		std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
			<< GDT_TERMINAL_DEFAULT << std::endl;
		exit(1);
	}

	return 0;
}