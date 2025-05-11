#ifndef __INIT_H__
#define __INIT_H__

#include "esp_all.h"
#include "config.h"
class Init
{
public:
    Init();
    ~Init();
    public:
    esp_err_t init();
    bool force_psram_init();
    bool hardware_reset();
    bool getSDCardStatus() { return sdCardStatus; }
    private:
    bool sdCardStatus = false;
    esp_err_t init_nvs();
    esp_err_t init_sd_card();
    esp_err_t init_spiffs();
    esp_err_t init_wifi();
    esp_err_t init_camera();
    esp_err_t init_ble();
    void configure_thread_settings();
};

#endif // __INIT_H__