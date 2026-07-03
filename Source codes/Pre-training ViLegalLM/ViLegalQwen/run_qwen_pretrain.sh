#!/bin/bash

# Tạo thư mục cần thiết
mkdir -p ./checkpoint/qwen-legal
mkdir -p ./log/qwen-legal

# Bước 1: Chuẩn bị dữ liệu cho Qwen (dữ liệu văn bản thô)
echo "===== Bước 1: Tạo tập dữ liệu cho Qwen ====="
python make_dataset_qwen.py

# Bước 2: Pre-training LLM Qwen
echo "===== Bước 2: Pre-training Qwen2.5-1.5B ====="

# Đặt CUDA_VISIBLE_DEVICES nếu cần
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Chạy pre-training
python pretrain_qwen.py

echo "===== Quá trình training hoàn tất ====="