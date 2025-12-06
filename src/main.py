import argparse
import os
import json
import numpy as np

from fault_detection.random_forest import RandomForestDetector
from fault_detection.MLP import MLPDetector
from fault_detection.xgboost import XGBoostDetector
from fault_Identification.MLP import MLPIdentifier
from fault_Identification.random_forest import RandomForestIdentifier
from fault_Identification.xgboost import XGBoostIdentifier

from utils.data_loader import Dataloader
from utils.evaluation import Evaluator
from config import Config


def get_model_dir(config: Config, task: str, dataset: str, model: str) -> str:
    """获取模型保存/加载目录
    
    目录结构: result/{task}/{model}/{dataset}/
    """
    return os.path.join(config.output_dir, task, model, dataset)


def run_detection(dataset: str, model: str, config: Config, train: bool = True, tune: bool = False):
    print(f"\n{'='*50}")
    print(f"任务: Detection | 数据集: {dataset} | 模型: {model}")
    print(f"模式: {'训练' if train else '评测'}" + (f" | 超参数调优: {'是' if tune else '否'}" if model in ['rf', 'xgb'] else ""))
    print(f"{'='*50}")
    
    data_loader = Dataloader(data_root=config.data_root, data_name=dataset)
    data = data_loader.get_data(dataset)
    
    X_train = data['X_train'] 
    y_train = (data['y_train'] != 0).astype(int)  # detection只用2值标签
    X_test = data['X_test']
    y_test = (data['y_test'] != 0).astype(int)

    feature_names = data['feature_names']
    
    # 模型保存目录: result/detection/{model}/{dataset}/
    output_dir = get_model_dir(config, 'detection', dataset, model)
    os.makedirs(output_dir, exist_ok=True)
    
    if model == 'mlp':
        model_path = os.path.join(output_dir, 'model.pth')
        
        if train:
            # 训练模式
            detector = MLPDetector(
                input_dim=X_train.shape[1],
                hidden_layers=config.mlp.hidden_layers,
                dropout_rate=config.mlp.dropout_rate,
                use_batchNorm=config.mlp.use_batchNorm,
                lr=config.mlp.lr,
                seed=config.mlp.seed
            )
            detector.fit(
                X_train, y_train,
                epochs=config.mlp.epochs,
                batch_size=config.mlp.batch_size,
                use_cosine_annealing=config.mlp.use_cosine_annealing,
                eta_min=config.mlp.eta_min
            )
            detector.save(model_path)
        else:
            # 评测模式：加载已保存的模型
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}，请先使用 --train 训练模型")
            detector = MLPDetector.load(model_path)
    elif model == 'rf':
        # 随机森林
        best_params_path = os.path.join(output_dir, 'best_params.json')
        
        if tune:
            # 超参数调优模式：使用GridSearchCV寻找最佳参数
            detector = RandomForestDetector(
                seed=config.rf.seed,
                n_jobs=config.rf.n_jobs,
                class_weight=config.rf.class_weight
            )
            tuning_results = detector.hyperparameter_tuning(
                X_train, y_train,
                param_grid=config.rf.param_grid,
                cv_folds=5,
                scoring='f1'
            )
            detector.feature_names = feature_names
            # 保存最佳参数
            with open(best_params_path, 'w', encoding='utf-8') as f:
                json.dump(tuning_results['best_params'], f, indent=2, ensure_ascii=False)
            print(f"最佳参数已保存至: {best_params_path}")
        else:
            # 非调优模式：加载best_params.json并使用最佳参数训练
            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"参数文件不存在: {best_params_path}，请先使用 --tune 进行超参数调优")
            
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"已加载最佳参数: {best_params}")
            
            detector = RandomForestDetector(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                max_features=best_params.get('max_features', 'sqrt'),
                seed=config.rf.seed,
                n_jobs=config.rf.n_jobs,
                class_weight=config.rf.class_weight
            )
            detector.fit(X_train, y_train, feature_names=feature_names)
        
        # 保存特征重要性图
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        detector.plot_feature_importance(save_path=importance_path)
    else:
        # XGBoost
        best_params_path = os.path.join(output_dir, 'best_params.json')
        
        if tune:
            # 超参数调优模式
            detector = XGBoostDetector(
                seed=config.xgb.seed,
                n_jobs=config.xgb.n_jobs,
                use_gpu=config.xgb.use_gpu
            )
            tuning_results = detector.hyperparameter_tuning(
                X_train, y_train,
                param_grid=config.xgb.param_grid,
                cv_folds=5,
                scoring='f1',
                auto_balance=True
            )
            detector.feature_names = feature_names
            # 保存最佳参数
            with open(best_params_path, 'w', encoding='utf-8') as f:
                json.dump(tuning_results['best_params'], f, indent=2, ensure_ascii=False)
            print(f"最佳参数已保存至: {best_params_path}")
        else:
            # 非调优模式：加载best_params.json并使用最佳参数训练
            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"参数文件不存在: {best_params_path}，请先使用 --tune 进行超参数调优")
            
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"已加载最佳参数: {best_params}")
            
            detector = XGBoostDetector(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 6),
                learning_rate=best_params.get('learning_rate', 0.1),
                subsample=best_params.get('subsample', 0.8),
                colsample_bytree=best_params.get('colsample_bytree', 0.8),
                min_child_weight=best_params.get('min_child_weight', 1),
                seed=config.xgb.seed,
                n_jobs=config.xgb.n_jobs,
                use_gpu=config.xgb.use_gpu
            )
            detector.fit(X_train, y_train, feature_names=feature_names, auto_balance=True)
        
        # 保存特征重要性图
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        detector.plot_feature_importance(save_path=importance_path)
    
    # 预测和评估
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_prob(X_test)
    
    label_map = {0: '正常', 1: '故障'}
    evaluator = Evaluator(y_test, y_pred, y_prob, label_map)
    evaluator.print_results()
    evaluator.save_results(output_dir, 'results')


def run_identification(dataset: str, model: str, config: Config, train: bool = True, tune: bool = False):
    print(f"\n{'='*50}")
    print(f"任务: Identification | 数据集: {dataset} | 模型: {model}")
    print(f"模式: {'训练' if train else '评测'}" + (f" | 超参数调优: {'是' if tune else '否'}" if model in ['rf', 'xgb'] else ""))
    print(f"{'='*50}")
    
    data_loader = Dataloader(data_root=config.data_root, data_name=dataset)
    data_loader.load_data(dataset, fault_only=True)
    data = data_loader.get_data(dataset)
    
    label_map = data['label_map']
    feature_names = data['feature_names']
    num_classes = len(label_map)
    
    X_train = data['X_train']
    y_train = data['y_train'] - 1  # 训练时要求标签从0开始
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 模型保存目录: result/identification/{model}/{dataset}/
    output_dir = get_model_dir(config, 'identification', dataset, model)
    os.makedirs(output_dir, exist_ok=True)
    
    if model == 'mlp':
        model_path = os.path.join(output_dir, 'model.pth')
        
        if train:
            # 训练模式
            identifier = MLPIdentifier(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                hidden_layers=config.mlp.hidden_layers,
                dropout_rate=config.mlp.dropout_rate,
                use_batchNorm=config.mlp.use_batchNorm,
                lr=config.mlp.lr,
                seed=config.mlp.seed
            )
            identifier.fit(
                X_train, y_train,
                epochs=config.mlp.epochs,
                batch_size=config.mlp.batch_size,
                use_cosine_annealing=config.mlp.use_cosine_annealing,
                eta_min=config.mlp.eta_min
            )
            identifier.save(model_path)
        else:
            # 评测模式：加载已保存的模型
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}，请先使用 --train 训练模型")
            identifier = MLPIdentifier.load(model_path)
    elif model == 'rf':
        # 随机森林
        best_params_path = os.path.join(output_dir, 'best_params.json')
        
        if tune:
            # 超参数调优模式：使用GridSearchCV寻找最佳参数
            identifier = RandomForestIdentifier(
                num_classes=num_classes,
                seed=config.rf.seed,
                n_jobs=config.rf.n_jobs,
                class_weight=config.rf.class_weight
            )
            tuning_results = identifier.hyperparameter_tuning(
                X_train, y_train,
                param_grid=config.rf.param_grid,
                cv_folds=5,
                scoring='accuracy'  # 多分类任务使用accuracy
            )
            identifier.feature_names = feature_names
            # 保存最佳参数
            with open(best_params_path, 'w', encoding='utf-8') as f:
                json.dump(tuning_results['best_params'], f, indent=2, ensure_ascii=False)
            print(f"最佳参数已保存至: {best_params_path}")
        else:
            # 非调优模式：加载best_params.json并使用最佳参数训练
            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"参数文件不存在: {best_params_path}，请先使用 --tune 进行超参数调优")
            
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"已加载最佳参数: {best_params}")
            
            identifier = RandomForestIdentifier(
                num_classes=num_classes,
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', None),
                min_samples_split=best_params.get('min_samples_split', 2),
                min_samples_leaf=best_params.get('min_samples_leaf', 1),
                max_features=best_params.get('max_features', 'sqrt'),
                seed=config.rf.seed,
                n_jobs=config.rf.n_jobs,
                class_weight=config.rf.class_weight
            )
            identifier.fit(X_train, y_train, feature_names=feature_names)
        
        # 保存特征重要性图
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        identifier.plot_feature_importance(save_path=importance_path)
    else:
        # XGBoost
        best_params_path = os.path.join(output_dir, 'best_params.json')
        
        if tune:
            # 超参数调优模式
            identifier = XGBoostIdentifier(
                num_classes=num_classes,
                seed=config.xgb.seed,
                n_jobs=config.xgb.n_jobs,
                use_gpu=config.xgb.use_gpu
            )
            tuning_results = identifier.hyperparameter_tuning(
                X_train, y_train,
                param_grid=config.xgb.param_grid,
                cv_folds=5,
                scoring='accuracy',  # 多分类任务使用accuracy
                auto_balance=False  # 多分类不使用
            )
            identifier.feature_names = feature_names
            # 保存最佳参数
            with open(best_params_path, 'w', encoding='utf-8') as f:
                json.dump(tuning_results['best_params'], f, indent=2, ensure_ascii=False)
            print(f"最佳参数已保存至: {best_params_path}")
        else:
            # 非调优模式：加载best_params.json并使用最佳参数训练
            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"参数文件不存在: {best_params_path}，请先使用 --tune 进行超参数调优")
            
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"已加载最佳参数: {best_params}")
            
            identifier = XGBoostIdentifier(
                num_classes=num_classes,
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 6),
                learning_rate=best_params.get('learning_rate', 0.1),
                subsample=best_params.get('subsample', 0.8),
                colsample_bytree=best_params.get('colsample_bytree', 0.8),
                min_child_weight=best_params.get('min_child_weight', 1),
                seed=config.xgb.seed,
                n_jobs=config.xgb.n_jobs,
                use_gpu=config.xgb.use_gpu
            )
            identifier.fit(X_train, y_train, feature_names=feature_names, auto_balance=False)
        
        # 保存特征重要性图
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        identifier.plot_feature_importance(save_path=importance_path)
    
    # 预测和评估
    y_pred = identifier.predict(X_test) + 1  # 预测结果还原标签
    
    evaluator = Evaluator(y_test, y_pred, None, label_map)
    evaluator.print_results()
    evaluator.save_results(output_dir, 'results')


def main():
    parser = argparse.ArgumentParser(description='故障检测与识别系统')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'identification'],
                        help='任务类型: detection 或 identification')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称: 激光载荷, 供配电, 姿轨控')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'rf', 'xgb'],
                        help='模型类型: mlp, rf 或 xgb')
    parser.add_argument('--train', action='store_true',
                        help='MLP训练模式：训练新模型并保存。不指定则加载已保存的模型进行预测')
    parser.add_argument('--tune', action='store_true',
                        help='RF/XGB超参数调优模式：进行网格搜索并保存最佳参数。不指定则加载已保存的best_params进行训练')
    
    args = parser.parse_args()
    config = Config()
    
    # 检查参数合法性
    if args.tune and args.model not in ['rf', 'xgb']:
        print("警告: --tune 参数仅对 rf 和 xgb 模型有效")
    if args.train and args.model != 'mlp':
        print("警告: --train 参数仅对MLP模型有效，rf/xgb请使用 --tune")
    
    if args.task == 'detection':
        run_detection(args.dataset, args.model, config, args.train, args.tune)
    else:
        run_identification(args.dataset, args.model, config, args.train, args.tune)


if __name__ == "__main__":
    main()