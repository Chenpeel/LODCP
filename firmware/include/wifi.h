#ifndef WIFI_H
#define WIFI_H

#include "esp_event.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#define WIFI_SSID "X"
#define WIFI_PASS "QWE999@@"
#define WIFI_MAXIMUM_RETRY 5

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

esp_err_t init_wifi();

#endif // WIFI_H
