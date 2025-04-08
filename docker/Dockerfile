FROM --platform=linux/amd64 espressif/idf:release-v5.1

WORKDIR /workspace

# 安装额外工具（按需添加）
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/espressif/esp32-camera.git components/esp32-camera
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
