#include <ESPmDNS.h>
#include <WebServer.h>
#include <WiFi.h>

WebServer server(80);

void setup_webserver() {
  // 连接WiFi
  WiFi.begin("SSID", "password");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // 启动mDNS
  if (!MDNS.begin("esp32-cam")) {
    Serial.println("Error setting up MDNS responder!");
  }

  // 设置路由
  server.on("/", HTTP_GET, []() {
    String html = "<html><body>";
    html += "<img src='/stream' width='640' height='480'>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });

  server.on("/stream", HTTP_GET, []() {
    // 设置流响应头
    server.sendHeader("Content-Type",
                      "multipart/x-mixed-replace; boundary=frame");
    server.sendHeader("Cache-Control", "no-cache");
    server.sendHeader("Connection", "close");
    server.send(200, "multipart/x-mixed-replace; boundary=frame", "");

    // 持续发送视频流
    while (1) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (!fb)
        continue;

      server.sendContent("--frame\r\n");
      server.sendContent("Content-Type: image/jpeg\r\n\r\n");
      server.sendContent((char *)fb->buf, fb->len);
      server.sendContent("\r\n");

      esp_camera_fb_return(fb);
      delay(10);
    }
  });

  server.begin();
}
