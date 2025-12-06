# 卫星健康监测

## 快速开始

安装依赖：

```
pip install -r requirements.txt
```

运行示例
- 故障检测（detection）示例：

```
python3 src/main.py --task detection --dataset 激光载荷 --model rf
```

- 故障识别（identification）示例：

```
python3 src/main.py --task identification --dataset 供配电 --model mlp
```

说明
- 可用数据集位于 `data/train/` 下：`供配电`、`姿轨控`、`激光载荷`。
- `--model` 可选 `mlp` 或 `rf`（对应 MLP 或 Random Forest）。
- 输出结果会保存到 `config.py` 中配置的 `output_dir` 下。

如需更多信息或调参说明，请查看 `src/main.py` 与 `src/config.py`。
