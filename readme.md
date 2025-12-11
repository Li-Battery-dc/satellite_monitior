# 卫星健康监测

## 运行流程

### 安装依赖：

```
pip install -r requirements.txt
```

## 工作区目录结构

```
satellite_monitor/
├── readme.md                      # 项目说明文档
├── requirements.txt               # Python依赖包列表
│
├── src/                           # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── config.py                 # 配置文件
│   │
│   ├── data_preprocess/          # 数据预处理模块
│   │   ├── preprocess.ipynb      # 数据预处理Notebook
│   │   └── visualize.ipynb       # 数据可视化Notebook
│   │
│   ├── fault_detection/          # 故障检测模块
│   │   ├── __init__.py
│   │   ├── evaluate.py           # 检测评估函数
│   │   └── tune.py               # 检测超参数调优
│   │
│   ├── fault_Identification/     # 故障识别模块
│   │   ├── __init__.py
│   │   ├── evaluate.py           # 识别评估函数
│   │   └── tune.py               # 识别超参数调优
│   │
│   ├── model/                    # 机器学习模型
│   │   ├── mlp/                  # 多层感知机
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # MLP模型定义
│   │   │   ├── train.py          # MLP训练函数
│   │   │   └── utils.py          # MLP工具函数
│   │   │
│   │   ├── rf/                   # 随机森林
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # RF模型定义
│   │   │   ├── train.py          # RF训练函数
│   │   │   └── utils.py          # RF工具函数
│   │   │
│   │   └── xgb/                  # XGBoost
│   │       ├── __init__.py
│   │       ├── model.py          # XGB模型定义
│   │       ├── train.py          # XGB训练函数
│   │       └── utils.py          # XGB工具函数
│   │
│   └── utils/                    # 工具模块
│       ├── __init__.py
│       └── data_loader.py        # 数据加载工具
│
├── scripts/                       # 脚本目录
│   ├── run.sh                    # 单次训练脚本
│   ├── run_all.sh                # 批量训练脚本
│   ├── evaluate.sh               # 单次评估脚本
│   ├── evaluate_all.sh           # 批量评估脚本
│   └── save_result_to_xls.py     # 结果转Excel脚本
│
├── data/                         # 数据目录
│   └── train/                    # 训练数据
│       ├── 供配电/                # 供配电子系统数据
│       │   ├── 正常数据.csv       # 正常状态数据
│       │   ├── 故障数据.csv       # 故障状态数据
│       │   ├── processed_train.csv # 预处理后训练集
│       │   ├── processed_test.csv  # 预处理后测试集
│       │   ├── processed_all.csv   # 预处理后全部数据
│       │   ├── object_to_enum.json # 名义变量映射
│       │   ├── enum_to_object.json # 枚举特征映射
│       │   ├── atoi.json          # 标签映射(字符到整数)
│       │   └── itoa.json          # 标签映射(整数到字符)
│       │
│       ├── 激光载荷/              # 激光载荷子系统数据
│       │   └── [同供配电目录结构]
│       │
│       └── 姿轨控/               # 姿轨控子系统数据
│           └── [同供配电目录结构]
│
├── Best_params/                   # 最佳参数目录
│   ├── detection/                 # 故障检测最佳参数
│   │   ├── mlp/                   # MLP模型参数
│   │   │   ├── 供配电/             # 各子系统模型文件(model.pth)
│   │   │   ├── 激光载荷/
│   │   │   └── 姿轨控/
│   │   │
│   │   ├── rf/                    # 随机森林最佳参数
│   │   │   └── [各子系统best_params.json]
│   │   │
│   │   └── xgb/                   # XGBoost最佳参数
│   │       └── [各子系统best_params.json]
│   │
│   └── identification/            # 故障识别最佳参数
│       └── [同detection目录结构]
│
├── result_best/                   # 最佳结果目录
│   ├── detection/                 # 故障检测结果
│   │   ├── mlp/                   # MLP检测结果
│   │   │   └── [各子系统结果目录]
│   │   │       ├── results.json           # 评估结果
│   │   │       ├── feature_importance.png # 特征重要性(RF/XGB)
│   │   │       ├── confusion_matrix.png  # 混淆矩阵
│   │   │       └── roc_curve.png         # ROC曲线
│   │   │
│   │   ├── rf/                    # 随机森林检测结果
│   │   └── xgb/                   # XGBoost检测结果
│   │
│   └── identification/            # 故障识别结果
│       └── [同detection目录结构]
│
└── excel_results/                 # Excel格式结果
    └── results_summary.txt        # 结果汇总文本
```

### 数据准备

- 每个子系统放在 `data/{split}/<数据集名称>/` 目录下
- 运行data_process目录下的preprocess.ipynb文件，修改data_paths变量,运行会得到：
  - `processed_train.csv`：训练集（含 `label` 列）。
  - `processed_test.csv`：测试集（含 `label` 列）。
  - `processed_all.csv`:加载的所有数据
  - `enum_to_object.json`：名义（枚举）特征映射（用于还原/处理名义变量）。
  - `object_to_enmu.json`:名义变量到特征映射。
  - `atoi.json`:标签映射（字符串到整数）。
  - `itoa.json`:标签映射（整数到字符串），用于 identification 任务显示标签名。
- data_process会将数据随机分成 2:1的训练集和数据集，但在使用的时候如何划分由config 的 `spilt_mode`指定。

### 修改参数

- 修改conifg中这个spilt_mode: 自行划分方式使用file模式，测试train和test两个目录时使用dir模式。
- 如果要使用我已经得到的params, 将params_load_dir目录修改为我上传的Best_params目录即可。
- 修改run.sh / evaluate.sh 中的TASK, MODEL, DATA参数对指定的数据集运行训练和测试。所有结果会保存在config中指定的result_dir路径下。
- run.sh  / evaluate.sh 的区别是是否进行模型训练和超参数调优。
- 使用evaluate_all.sh脚本运行所有测试。

## 运行参数说明

- 不加train和tune参数的时候自动加载params_load_dir目录已经指定参数搜索之后的内容
- 命令行参数（脚本 `src/main.py`）：

  - `--task`：`detection` 或 `identification`。
  - `--dataset`：数据集名称（`激光载荷`、`供配电`、`姿轨控`）
  - `--model`：`mlp` / `rf` / `xgb`。
  - `--train`：仅用于 `mlp`，表示训练 MLP 并保存模型（不加则加载已保存模型进行评测）。
  - `--tune`：用于 `rf` 和 `xgb`，表示进行超参数网格搜索并保存最佳参数（不加则加载 `best_params.json` 并使用最佳参数训练后评测）。
- task参数：`detection`（故障检测，二分类）和 `identification`（故障识别，多分类）。
- 支持的模型：`mlp`、`rf`、`xgb`。

## 输出与目录

- 评估结果将保存在 `config.result_dir/{task}/{model}/{dataset}/` 下

  - `feature_importance.png`：特征重要性图（RF/XGB）。
  - `confusion_matrix.png`: 绘制得到的混淆矩阵。
  - `result.json`：保存评估输出的所有输出。
- 超参数搜索和mlp训练得到的参数保存在 `config.params_save_dir/{task}/{model}/{dataset}/`下

  - `best_params.json`：超参数搜索得到的最佳参数（当使用 `--tune` 时生成）。
  - `model.pth`：MLP 模型文件（使用 `--train`参数时训练得到）
- 仅评测时请将config目录的 `params_load_dir`修改为我给出的最佳参数。
- 使用scripts/save_result_to_xls脚本可以将我的json结果读取得到excel表格文件。使用前请修改参数：

```python
processor = ResultProcessor(
        result_base_path="./result_best", 
        result_one_vs_one="./result_one_vs_one",
        data_base_path="./data/train/", 
        save_path="./excel_results"
    )
```

## 运行示例

- 在scripts目录下，我已经写了运行的bash脚本，修改对应的bash中的参数可以方便地运行训练和评估。

在顶层目录运行

```
bash ./scripts/evaluate_all.sh
```

即可对已有的params_load_dir目录下的参数进行一次完整的评估.

- 对某个数据集进行 XGBoost 的超参数调优（Detection）：

```
python src/main.py --task detection --dataset 激光载荷 --model xgb --tune
```

- 使用params_load_dir中的参数训练并评估 XGBoost（Detection）：

（在已经存在 `params/detection/xgb/激光载荷/best_params.json` 的情况下）

```
python src/main.py --task detection --dataset 激光载荷 --model xgb
```

- 训练 MLP 并保存模型（Identification）：

```
python src/main.py --task identification --dataset 供配电 --model mlp --train
```

- 使用已保存的 MLP 模型进行评测（Identification）：

```
python src/main.py --task identification --dataset 供配电 --model mlp
```

更多细节请参见 `src/main.py`、`src/config.py` 与 `src/utils/data_loader.py`。

## 一些问题

- 现在的run脚本的主要耗时是在rf和xgb的参数搜索部分。
- 当前的MLP model进行加载的模型是config中给出的模型层数架构。如果修改这个参数可以在run.sh中单独训一遍MLP，耗时不是很长。
