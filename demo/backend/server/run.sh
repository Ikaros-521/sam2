#!/bin/bash

# 初始化 conda
eval "$(/usr/local/miniconda3/bin/conda shell.bash hook)"

# 激活环境
conda activate sam2

# 默认IP地址
DEFAULT_IP="117.50.46.27"

# 提示用户输入IP地址
echo "请输入服务器IP地址 (直接回车使用默认值: $DEFAULT_IP):"
read -r USER_IP

# 如果用户没有输入，使用默认IP
if [ -z "$USER_IP" ]; then
    USER_IP="$DEFAULT_IP"
    echo "使用默认IP: $USER_IP"
else
    echo "使用输入的IP: $USER_IP"
fi

# 构建API URL
API_URL="http://$USER_IP:7263"

echo "启动服务器，API地址: $API_URL"
echo "按 Ctrl+C 停止服务器"
echo "----------------------------------------"

# 启动服务器
PYTORCH_ENABLE_MPS_FALLBACK=1 \
APP_ROOT="$(pwd)/../../../" \
API_URL="$API_URL" \
MODEL_SIZE=base_plus \
DATA_PATH="$(pwd)/../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn --worker-class gthread app:app --workers 1 --threads 2 --bind 0.0.0.0:7263 --timeout 60