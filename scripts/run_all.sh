#!/bin/bash

# 运行所有任务脚本

cd "$(dirname "$0")/.."

DATASETS=("激光载荷" "供配电" "姿轨控")
MODELS=("rf" "mlp")
TASKS=("detection" "identification")

for task in "${TASKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "========================================"
            echo "运行: $task - $dataset - $model"
            echo "========================================"
            python src/main.py --task "$task" --dataset "$dataset" --model "$model"
        done
    done
done

echo "所有任务完成！"
