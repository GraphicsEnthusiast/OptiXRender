project ("core")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

add_library(core 
    ./cuda/Utils.h
    ./include/LaunchParams.h
    ./include/Renderer.h
    ./include/Scene.h
    src/Renderer.cpp
    src/Scene.cpp
)

target_include_directories(core PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
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