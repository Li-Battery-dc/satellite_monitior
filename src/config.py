class MLPConfig:
    hidden_layers = [512, 256, 128, 64, 32]
    dropout_rate = 0.3
    use_batchNorm = True
    lr = 0.005
    epochs = 100
    batch_size = 512
    seed = 42
    use_cosine_annealing = True
    eta_min = 1e-6


class RFConfig:
    # 以下参数不在param_grid中，需要手动指定
    seed = 42
    n_jobs = 16
    class_weight = 'balanced'
    
    # 超参数搜索空间
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [1, 5, 10],
        'min_samples_leaf': [1, 3, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3, 0.5]
    }


class XGBConfig:
    # 以下参数不在param_grid中，需要手动指定
    seed = 42
    n_jobs = 16 
    use_gpu = False # 数据量很小，不用GPU搜索快的多

    confused_pairs = [
        (9, 15),  # 供配电： 单体电池短路vs蓄电池组加热带误通
        (3, 4)     # 姿轨控：姿态控制系统故障vs轨道控制系统故障
    ]

    # 超参数搜索空间
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    }


class Config:
    mlp = MLPConfig()
    rf = RFConfig()
    xgb = XGBConfig()
    
    datasets = ['激光载荷', '供配电', '姿轨控']
    data_root = './data/'
    split_mode = 'file'  # 'file' 或 'dir'
    result_dir = './result_12_9'  # 保存结果的根目录
    params_save_dir = './params' # 保存参数搜索和模型训练的结果
    params_load_dir = './params_best'
