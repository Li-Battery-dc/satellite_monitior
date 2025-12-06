#!/bin/bash

# 训练所有模型脚本
# MLP: 训练模式
# RF/XGB: 超参数调优模式

cd "$(dirname "$0")/.."

DATASETS=("激光载荷" "供配电" "姿轨控")
# MODELS=("rf" "mlp" "xgb")
MODELS=("xgb")
TASKS=("detection" "identification")

for task in "${TASKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "========================================"
            echo "训练: $task - $dataset - $model"
            echo "========================================"
            
            if [ "$model" == "mlp" ]; then
                python src/main.py --task "$task" --dataset "$dataset" --model "$model" --train
            else
                python src/main.py --task "$task" --dataset "$dataset" --model "$model" --tune
            fi
        done
    done
done

echo "所有训练任务完成！"
