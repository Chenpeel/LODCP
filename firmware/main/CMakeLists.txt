idf_component_register(
    SRCS
        "main.cpp"
        "init.cpp"
        "wifi.cpp"
        "process.cpp"
        "model_loader.cpp"
        "deep_sort.cpp"
        "semantic_seg.cpp"
        "collision.cpp"
        "image_filters.cpp"
        "video_recorder.cpp"
    INCLUDE_DIRS
        "."
        "../include"

    REQUIRES
        # 核心组件
        esp32-camera
        driver
        freertos
        nvs_flash

        # 网络相关组件
        esp_http_server
        esp_https_ota
        esp_wifi

        # 存储相关组件
        sdmmc
        fatfs

        # 工具和库
        json
        mbedtls
        esp_timer
        tflite-micro
        esp-dl
)
