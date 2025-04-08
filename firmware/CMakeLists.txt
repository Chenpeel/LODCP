# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.16)

# 包含 ESP-IDF 构建系统
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# 项目名称
project(lodcp)

# 添加组件目录
list(APPEND EXTRA_COMPONENT_DIRS
    components/
    $ENV{IDF_PATH}/components/
)

# 可选：全局编译选项
add_compile_options(
    -Wno-unused-variable   # 禁用特定警告
    -Wno-unused-but-set-variable
)

# 包含主程序目录
add_subdirectory(main)


#组件依赖检查
find_package(Threads REQUIRED)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/components/esp32-camera")
    message(FATAL_ERROR "esp32-camera component missing!")
endif()
