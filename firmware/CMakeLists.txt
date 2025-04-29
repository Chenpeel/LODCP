# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.16)

# 添加额外组件目录 - 必须在 include 项目前定义
set(EXTRA_COMPONENT_DIRS
    components/esp32-camera
    components/esp-dl
    components/tflite-micro
)

# 包含 ESP-IDF 构建系统 - 这必须在设置 EXTRA_COMPONENT_DIRS 之后
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# 项目名称
project(lodcp)
