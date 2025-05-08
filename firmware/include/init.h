// init.h
#ifndef INIT_H
#define INIT_H

#include "../include/wifi.h"
#include "camera_config.h"
#include "esp_err.h"
class init{
    public:
    // 初始化所有组件
    esp_err_t initialize_all();
    private:
        // 初始化NVS
        esp_err_t init_nvs();

        // 初始化摄像头
        esp_err_t init_camera();
    
        // 初始化SD卡
        esp_err_t init_sd_card();
        
        // 初始化SPIFFS文件系统
        esp_err_t init_spiffs();
    };
#endif // INIT_H
