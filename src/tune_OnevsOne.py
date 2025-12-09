import os
import sys
import json
import numpy as np
import xgboost as xgb
from utils.data_loader import Dataloader
from config import Config as config
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ==========================================
# 🔧 硬编码配置区域 (请根据实际情况修改)
# ==========================================

# 1. 混淆类别对 (注意：请确认是 0-indexed 还是 1-indexed)
# 如果之前的报错是因为15越界，这里请填入实际存在的ID，例如 (9, 14)
CONFUSED_PAIR = (9, 15) 

# 2. 结果保存路径
# 确保这个路径和你主程序读取 refiner_best_params.json 的路径一致
SAVE_DIR = "../params_12_8/identification/供配电/xgb"
SAVE_FILE = "refiner_best_params.json"

# 3. 搜索网格 (针对二分类精修的保守参数范围)
PARAM_GRID = {
    'max_depth': [5, 6, 7, 9],            
    'learning_rate': [0.01, 0.03, 0.05], # 低学习率：学得细致
    'n_estimators': [200, 300, 500],    # 配合低学习率
    'min_child_weight': [2, 3, 4],      # 提高叶子节点门槛，防过拟合
    'reg_lambda': [1, 3, 5],            # L2正则化强度
    'subsample': [0.6, 0.8, 1.0],            # 样本采样
    'colsample_bytree': [0.6, 0.8, 1.0]      # 特征采样
}

# 4. 线程数与GPU设置
N_JOBS = 1
USE_GPU = True
SEED = 42

# ==========================================
# 📥 数据加载区域 (需要你填写)
# ==========================================

def load_your_data(dataset: str = '供配电'):
    """
    请在这里复制你 main.py 中加载供配电数据的代码
    返回: X_train (numpy array), y_train (numpy array)
    """
    print("正在加载数据...")
    data_loader = Dataloader(data_root=config.data_root, data_name=dataset, split_mode=config.split_mode)
    data_loader.load_data(dataset, fault_only=True)
    data = data_loader.get_data(dataset)
    
    X_train = data['X_train']
    y_train = data['y_train']
    return X_train, y_train


# ==========================================
# 🚀 主执行逻辑
# ==========================================

def run_search():
    # 1. 准备目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, SAVE_FILE)

    # 2. 加载数据
    try:
        X_train, y_train = load_your_data()
    except NotImplementedError as e:
        print(f"\n❌ 错误: {e}")
        print("提示: 请打开此脚本，在 load_your_data 函数中粘贴 main.py 里的数据加载代码。\n")
        return

    print(f"原始训练集形状: {X_train.shape}")
    
    # 3. 筛选混淆类别数据
    c1, c2 = CONFUSED_PAIR
    print(f"正在筛选混淆类别: {c1} 和 {c2} ...")
    
    mask = np.isin(y_train, [c1, c2])
    X_sub = X_train[mask]
    y_sub = y_train[mask]
    
    if len(y_sub) == 0:
        print(f"❌ 错误: 训练集中未找到类别 {c1} 或 {c2}。请检查 CONFUSED_PAIR 设置。")
        print(f"当前数据集包含的类别: {np.unique(y_train)}")
        return

    # 4. 转换为二分类标签 (0: c1, 1: c2)
    # XGBoost 二分类要求标签必须是 0 和 1
    y_sub_binary = (y_sub == c2).astype(int)
    
    print(f"精修模型训练样本数: {len(X_sub)}")
    print(f"类别分布: {c1}(0): {np.sum(y_sub_binary==0)}, {c2}(1): {np.sum(y_sub_binary==1)}")

    # 5. 初始化模型
    device = 'cuda' if USE_GPU else 'cpu'
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device=device,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbosity=1
    )

    # 6. 配置 Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring='f1', # 对于混淆对，Accuracy 通常足够，也可以用 'f1'
        n_jobs=N_JOBS,
        verbose=1
    )

    # 7. 开始搜索
    print("\n>>> 开始 Grid Search (Refiner)...")
    grid_search.fit(X_sub, y_sub_binary)

    # 8. 输出结果
    print("\n" + "="*50)
    print(f"搜索完成！")
    print(f"最佳准确率 (CV): {grid_search.best_score_:.4f}")
    print(f"最佳参数: {json.dumps(grid_search.best_params_, indent=2)}")
    print("="*50)

    # 9. 保存参数
    # 这里我们只保存搜索到的参数，其他参数(如 objective)由主程序运行时决定
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(grid_search.best_params_, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 最佳参数已保存至: {save_path}")
    print("现在你可以直接运行 main.py (不带 --tune)，它会自动加载这个文件。")

if __name__ == "__main__":
    run_search()