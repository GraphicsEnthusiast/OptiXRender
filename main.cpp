#include "Renderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

    struct SampleWindow : public GLFCameraWindow
    {
        SampleWindow(const std::string& title,
            const Scene* scene,
            const Camera& camera,
            const float worldScale)
            : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
            sample(scene)
        {
            sample.SetCamera(camera);
        }

        virtual void render() override
        {
            if (cameraFrame.modified) {
                sample.SetCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }
            sample.Render();
        }

        virtual void draw() override
        {
            sample.DownloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

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

        virtual void resize(const vec2i& newSize)
        {
            fbSize = newSize;
            sample.Resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }

        virtual void key(int key, int mods)
        {
            if (key == 'D' || key == ' ' || key == 'd') {
                sample.denoiserOn = !sample.denoiserOn;
                std::cout << "denoising now " << (sample.denoiserOn ? "ON" : "OFF") << std::endl;
            }
            if (key == 'A' || key == 'a') {
                sample.accumulate = !sample.accumulate;
                std::cout << "accumulation/progressive refinement now " << (sample.accumulate ? "ON" : "OFF") << std::endl;
            }
            if (key == ',') {
                sample.launchParams.numPixelSamples
                    = std::max(1, sample.launchParams.numPixelSamples - 1);
                std::cout << "num samples/pixel now "
                    << sample.launchParams.numPixelSamples << std::endl;
            }
            if (key == '.') {
                sample.launchParams.numPixelSamples
                    = std::max(1, sample.launchParams.numPixelSamples + 1);
                std::cout << "num samples/pixel now "
                    << sample.launchParams.numPixelSamples << std::endl;
            }
        }


        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        Renderer        sample;
        std::vector<uint32_t> pixels;
    };


    /*! main entry point to this example - initially optix, print hello
      world, then exit */
    extern "C" int main(int ac, char** av)
    {
        TextureName textureName;
        Material material;
        textureName.albedoFile = "../../models/01_Head_Base_Color.png";
        try {
            Scene scene;
            scene.AddMesh(
                "../../models/head.obj",
                material,
                textureName
            );
            textureName.albedoFile = "../../models/02_Body_Base_Color.png";
			scene.AddMesh(
				"../../models/body.obj",
				material,
				textureName
			);
            textureName.albedoFile = "../../models/03_Base_Base_Color.png";
			scene.AddMesh(
				"../../models/base.obj",
				material,
				textureName
			);
            material.type = MaterialType::Diffuse;
            textureName.albedoFile = "../../models/grid.jpg";
			scene.AddMesh(
				"../../models/plane.obj",
				material,
				textureName
			);
            scene.AddEnv("../../models/spaichingen_hill_4k.hdr");
            Camera camera = { /*from*/vec3f(0.2f, 0.2f, 0.2f),
                /* at */scene.bounds.center(),
                /* up */vec3f(0.f,1.f,0.f) };

            // something approximating the scale of the world, so the
            // camera knows how much to move for any given user interaction:
            const float worldScale = length(scene.bounds.span());
            Light light;
            light.position = vec3f(0.5f, 0.25f, 0.0f);
            light.radius = 0.1f;
            light.radiance = vec3f(15.0f);
            //scene.AddLight(light);

            //scene.AddLight(light);
            SampleWindow* window = new SampleWindow("RTRT_Render",
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
            std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
            exit(1);
        }
        return 0;
    }

} // ::osc
