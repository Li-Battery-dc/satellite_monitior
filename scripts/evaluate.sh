#!/bin/bash

TASK=identification # identification / detection
DATASET=姿轨控 # 供配电， 姿轨控， 激光载荷
MODEL=xgb # mlp or rf or xgb

python src/main.py --task "$TASK" --dataset "$DATASET" --model "$MODEL"
