#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#define WIFI_SSID "X"
#define WIFI_PASS "QWE999@@"
#define MAXIMUM_RETRY 5

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

void wifi_init_sta();
void wifi_event_handler(void *arg, esp_event_base_t event_base,
                        int32_t event_id, void *event_data);
