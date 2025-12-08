#!/bin/bash

TASK=identification # 监测任务: identification / 预测任务: prediction
DATASET=姿轨控 # 供配电， 姿轨控， 激光载荷
MODEL=xgb # mlp or rf or xgb

if [ "$MODEL" == "mlp" ]; then
    python src/main.py --task "$TASK" --dataset "$DATASET" --model "$MODEL" --train
else
    python src/main.py --task "$TASK" --dataset "$DATASET" --model "$MODEL" --tune
fi
