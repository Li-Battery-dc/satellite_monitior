#!/bin/bash

# 单次运行脚本
# 用法: ./run.sh <task> <dataset> <model>
# 示例: ./run.sh detection 激光载荷 rf

if [ $# -ne 3 ]; then
    echo "用法: $0 <task> <dataset> <model>"
    echo "  task: detection 或 identification"
    echo "  dataset: 激光载荷, 供配电, 姿轨控"
    echo "  model: mlp 或 rf"
    exit 1
fi

TASK=$1
DATASET=$2
MODEL=$3

cd "$(dirname "$0")/.."

python src/main.py --task "$TASK" --dataset "$DATASET" --model "$MODEL"
