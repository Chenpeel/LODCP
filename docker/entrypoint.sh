#!/bin/bash

# 激活 ESP-IDF 环境
. /opt/esp/idf/export.sh

# 自动检测串口设备（适用于 Mac/Linux）
if [ ! -e "$ESPPORT" ]; then
  echo "⚠️  Serial port $ESPPORT not found! Trying alternatives..."
  for dev in /dev/ttyUSB* /dev/cu.usb*; do
    if [ -e "$dev" ]; then
      export ESPPORT=$dev
      echo "🔌 Using detected serial port: $ESPPORT"
      break
    fi
  done
fi

# 执行用户命令
exec "$@"
