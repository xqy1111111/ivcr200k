#!/bin/bash

# --- 0. 环境激活检查 ---
# 建议在运行脚本前先手动：conda activate ivcr

# --- 1. 核心：安装 PyTorch (CUDA 12.1) ---
# 原项目是 PyTorch 2.1.2。为了适配 4090，必须用 cu121 版本，不能用原文件的 cu113
echo ">>> [1/5] 安装 PyTorch 2.1.2 (CUDA 12.1)..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# --- 2. 核心：安装 DeepSpeed ---
# 必须在 PyTorch 之后安装。利用你系统里的 nvcc 12.5 进行编译
echo ">>> [2/5] 编译并安装 DeepSpeed 0.14.2..."
pip install deepspeed==0.14.2

# --- 3. 主要深度学习依赖 (严格对齐原版本) ---
echo ">>> [3/5] 安装核心深度学习库..."
pip install \
    transformers==4.34.0 \
    accelerate==0.23.0 \
    diffusers \
    timm==0.6.13 \
    safetensors==0.4.0 \
    tokenizers==0.14.1 \
    sentence-transformers==2.2.2 \
    peft==0.5.0 \
    bitsandbytes==0.37.0 \
    einops==0.7.0 \
    omegaconf==2.3.0

# --- 4. 视觉、视频与科学计算 (补全原文件中的依赖) ---
echo ">>> [4/5] 安装视觉与科学计算库..."
pip install \
    opencv-python==4.7.0.72 \
    pillow==10.0.1 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    pandas==2.1.1 \
    matplotlib==3.7.0 \
    av==10.0.0 \
    decord==0.6.0 \
    pytorchvideo==0.1.5 \
    moviepy \
    imageio \
    webdataset==0.2.48

# --- 5. 工具、Web与杂项 (为了防止报错，把原文件里的散碎包都装上) ---
echo ">>> [5/5] 安装工具链与杂项依赖..."
pip install \
    yacs==0.1.8 \
    termcolor==2.3.0 \
    tabulate==0.9.0 \
    gradio==3.24.1 \
    wandb==0.15.12 \
    openai==0.27.0 \
    psutil==5.9.4 \
    tqdm==4.64.1 \
    pyyaml==6.0 \
    regex \
    sentencepiece \
    ftfy \
    joblib \
    click \
    networkx \
    fsspec \
    huggingface-hub \
    jupyterlab ipykernel notebook  # 交互式开发环境

echo "=========================================="
echo "   安装完成！请运行验证脚本检查。"
echo "=========================================="