idf_component_register(
    SRCS
        "main.cpp"
        "init.cpp"
        "wifi.cpp"
        "process.cpp"
    INCLUDE_DIRS
        "."
        "../components"
        "../include"

    REQUIRES
        esp32-camera
        esp_http_server
        esp_https_ota
        driver
        freertos
        nvs_flash
        sdmmc
        json               # 如需JSON处理
        mbedtls            # 加密相关

    PRIV_REQUIRES
        esp_timer          # 高精度定时器
        esp_wifi          # WiFi底层驱动
)
