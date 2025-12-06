class MLPConfig:
    hidden_layers = [256, 128, 64, 32]
    dropout_rate = 0.3
    use_batchNorm = True
    lr = 0.001
    epochs = 100
    batch_size = 32
    seed = 42


class RFConfig:
    n_estimators = 100
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    max_features = 'sqrt'
    seed = 42
    n_jobs = -1
    class_weight = 'balanced'
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }


class Config:
    mlp = MLPConfig()
    rf = RFConfig()
    
    datasets = ['激光载荷', '供配电', '姿轨控']
    data_root = './data/train'
    output_dir = './result'
