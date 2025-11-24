#!/bin/bash
echo "=== IVCR 环境检查 ==="
echo ""

# 检查conda环境
echo "1. 检查conda环境..."
conda env list | grep ivcr && echo "✓ conda环境存在" || echo "✗ conda环境不存在"

# 检查 /data1 大文件目录
echo ""
echo "2. 检查 /data1 大文件目录..."
[ -d /data1/ivcr/checkpoints ] && echo "✓ /data1/ivcr/checkpoints" || echo "✗ /data1/ivcr/checkpoints (不存在)"
[ -d /data1/ivcr/videos ] && echo "✓ /data1/ivcr/videos" || echo "✗ /data1/ivcr/videos (不存在)"

# 检查模型权重 (在 /data1)
echo ""
echo "3. 检查模型权重 (在 /data1)..."
[ -f /data1/ivcr/checkpoints/eva_vit_g.pth ] && echo "✓ EVA ViT-G" || echo "✗ EVA ViT-G"
[ -f /data1/ivcr/checkpoints/instruct_blip_vicuna7b_trimmed.pth ] && echo "✓ InstructBLIP" || echo "✗ InstructBLIP"
[ -d /data1/ivcr/checkpoints/Llama-2-7b-chat-hf ] && echo "✓ LLaMA-2-7B" || echo "✗ LLaMA-2-7B"
[ -d /data1/ivcr/checkpoints/Video-LLaMA-2-7B-Finetuned ] && echo "✓ Video-LLaMA" || echo "✗ Video-LLaMA"
[ -d /data1/ivcr/checkpoints/TimeChat-7b ] && echo "✓ TimeChat-7B" || echo "✗ TimeChat-7B"

# 检查视频数据 (在 /data1)
echo ""
echo "4. 检查视频数据 (在 /data1)..."
[ -d /data1/ivcr/videos/ivcr_compress ] && echo "✓ 压缩视频目录" || echo "✗ 压缩视频目录"

# 检查项目本地目录
echo ""
echo "5. 检查项目本地目录..."
[ -d ./data ] && echo "✓ ./data" || echo "✗ ./data (不存在)"
[ -d ./outputs ] && echo "✓ ./outputs" || echo "✗ ./outputs (不存在)"

# 检查JSON数据文件 (在项目目录)
echo ""
echo "6. 检查JSON数据文件 (在项目目录)..."
[ -f ./data/IVCR-200K.json ] && echo "✓ IVCR-200K.json" || echo "✗ IVCR-200K.json"
[ -f ./data/IVCR_no_type0_dialogues_train.json ] && echo "✓ 训练集JSON" || echo "✗ 训练集JSON (需运行convert_dataset.ipynb)"
[ -f ./data/IVCR_no_type0_no_zero_dialogues_test.json ] && echo "✓ 测试集JSON" || echo "✗ 测试集JSON (需运行convert_dataset.ipynb)"

echo ""
echo "=== 检查完成 ==="
echo ""
echo "提示："
echo "  - 大文件(模型权重、视频)应在 /data1/ivcr/"
echo "  - 小文件(JSON、结果)应在项目目录 ./data/ 和 ./outputs/"
