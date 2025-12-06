#!/bin/bash

# 评测所有模型脚本
# MLP: 加载已保存的模型进行评测
# RF/XGB: 加载best_params.json训练并评测

cd "$(dirname "$0")/.."

DATASETS=("激光载荷" "供配电" "姿轨控")
MODELS=("rf" "mlp" "xgb")
TASKS=("detection" "identification")

for task in "${TASKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "========================================"
            echo "评测: $task - $dataset - $model"
            echo "========================================"
            python src/main.py --task "$task" --dataset "$dataset" --model "$model"
        done
    done
done

echo "所有评测任务完成！"
