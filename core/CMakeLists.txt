project ("core")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

add_library(core 
    ./cuda/Utils.h
    ./cuda/Sampling.h
    ./cuda/Microfacet.h
    ./cuda/Material.h
    ./cuda/Light.h
    ./cuda/Environment.h
    ./cuda/Filter.h
    ./cuda/Medium.h
    ./include/LaunchParams.h
    ./include/Renderer.h
    ./include/Scene.h
    ./include/SceneParser.h
    src/Renderer.cpp
    src/Scene.cpp
    src/SceneParser.cpp
)

target_include_directories(core PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ../common/3rdParty/nlohmann_json/include
)

target_link_libraries(core PUBLIC
    gdt
# optix dependencies, for rendering
    ${optix_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
# glfw and opengl, for display
    glfWindow
    glfw
    ${OPENGL_gl_LIBRARY}
)