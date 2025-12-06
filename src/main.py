import argparse
import os
import numpy as np

from fault_detection.random_forest import RandomForestDetector
from fault_detection.MLP import MLPDetector
from fault_Identification.MLP import MLPIdentifier
from fault_Identification.random_forest import RandomForestIdentifier

from utils.data_loader import Dataloader
from utils.evaluation import Evaluator
from config import Config


def run_detection(dataset: str, model: str, config: Config):
    print(f"\n{'='*50}")
    print(f"任务: Detection | 数据集: {dataset} | 模型: {model}")
    print(f"{'='*50}")
    
    data_loader = Dataloader(data_root=config.data_root, data_name=dataset)
    data = data_loader.get_data(dataset)
    
    X_train = data['X_train'] 
    y_train = (data['y_train'] != 0).astype(int) # detection只用2值标签
    X_test = data['X_test']
    y_test = (data['y_test'] != 0).astype(int)

    feature_names = data['feature_names']
    
    output_dir = os.path.join(config.output_dir, 'detection', dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    if model == 'mlp':
        detector = MLPDetector(
            input_dim=X_train.shape[1],
            hidden_layers=config.mlp.hidden_layers,
            dropout_rate=config.mlp.dropout_rate,
            use_batchNorm=config.mlp.use_batchNorm,
            lr=config.mlp.lr,
            seed=config.mlp.seed
        )
        detector.fit(X_train, y_train, epochs=config.mlp.epochs, batch_size=config.mlp.batch_size)
    else:
        detector = RandomForestDetector(
            n_estimators=config.rf.n_estimators,
            max_depth=config.rf.max_depth,
            seed=config.rf.seed,
            n_jobs=config.rf.n_jobs,
            class_weight=config.rf.class_weight
        )
        importance_path = os.path.join(output_dir, f"{model}_feature_importance.png")
        detector.fit(X_train, y_train, feature_names=feature_names)
        detector.plot_feature_importance(save_path=importance_path)
    
    y_pred = detector.predict(X_test)
    y_prob = detector.predict_prob(X_test)
    
    label_map = {0: '正常', 1: '故障'}
    evaluator = Evaluator(y_test, y_pred, y_prob, label_map)
    evaluator.print_results()
    evaluator.save_results(output_dir, model)


def run_identification(dataset: str, model: str, config: Config):
    print(f"\n{'='*50}")
    print(f"任务: Identification | 数据集: {dataset} | 模型: {model}")
    print(f"{'='*50}")
    
    data_loader = Dataloader(data_root=config.data_root, data_name=None)
    data_loader.load_data(dataset, fault_only=True)
    data = data_loader.get_data(dataset)
    
    label_map = data['label_map']
    feature_names = data['feature_names']
    num_classes = len(label_map)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    output_dir = os.path.join(config.output_dir, 'identification', dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    if model == 'mlp':
        identifier = MLPIdentifier(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_layers=config.mlp.hidden_layers,
            dropout_rate=config.mlp.dropout_rate,
            use_batchNorm=config.mlp.use_batchNorm,
            lr=config.mlp.lr,
            seed=config.mlp.seed
        )
        identifier.fit(X_train, y_train, epochs=config.mlp.epochs, batch_size=config.mlp.batch_size)
    else:
        identifier = RandomForestIdentifier(
            num_classes=num_classes,
            n_estimators=config.rf.n_estimators,
            max_depth=config.rf.max_depth,
            seed=config.rf.seed,
            n_jobs=config.rf.n_jobs,
            class_weight=config.rf.class_weight
        )
        identifier.fit(X_train, y_train, feature_names=feature_names)
        importance_path = os.path.join(output_dir, f"{model}_feature_importance.png")
        identifier.plot_feature_importance(save_path=importance_path)
    
    y_pred = identifier.predict(X_test)
    
    evaluator = Evaluator(y_test, y_pred, None, label_map)
    evaluator.print_results()
    evaluator.save_results(output_dir, model)


def main():
    parser = argparse.ArgumentParser(description='故障检测与识别系统')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'identification'],
                        help='任务类型: detection 或 identification')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称: 激光载荷, 供配电, 姿轨控')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'rf'],
                        help='模型类型: mlp 或 rf')
    
    args = parser.parse_args()
    config = Config()
    
    if args.task == 'detection':
        run_detection(args.dataset, args.model, config)
    else:
        run_identification(args.dataset, args.model, config)


if __name__ == "__main__":
    main()