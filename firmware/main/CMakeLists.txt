idf_component_register(
    SRCS
        "main.c"            # 主代码文件
        #"camera_ops.c"      # 可选：摄像头操作分离文件
        #"wifi_connect.c"    # 可选：WiFi连接模块

    INCLUDE_DIRS
        "."                 # 当前目录头文件
        "../components"     # 自定义组件头文件

    REQUIRES
        esp32-camera        # 摄像头驱动
        esp_http_server     # HTTP服务器
        esp_https_ota       # 可选：OTA更新
        driver             # ESP32 基础驱动
        freertos           # RTOS 支持
        nvs_flash          # 存储配置
        sdmmc              # 如需SD卡支持
        json               # 如需JSON处理
        mbedtls            # 加密相关

    PRIV_REQUIRES
        esp_timer          # 高精度定时器
        esp_wifi          # WiFi底层驱动
)
