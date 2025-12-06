import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt

from utils.evaluation import Evaluator


class RandomForestDetector:
    """随机森林故障检测器"""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', seed: int = 42,
                 n_jobs: int = -1, class_weight: str = 'balanced'):
        """
        初始化随机森林检测器

        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂内部节点所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分割时考虑的特征数量
            seed: 随机种子
            n_jobs: 并行数
            class_weight: 类别权重
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.n_jobs = n_jobs
        self.class_weight = class_weight

        # 初始化随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_jobs,
            class_weight=class_weight
        )

        self.is_fitted = False
        self.feature_names = None
        self.cv_results = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: Optional[List[str]] = None,
            use_cv: bool = True, cv_folds: int = 5) -> Dict:
        """
        训练模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            feature_names: 特征名称列表（可选）
            use_cv: 是否使用交叉验证
            cv_folds: 交叉验证折数

        Returns:
            train_metrics: 训练集评估结果
        """
        # 保存特征名称
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        print("开始训练随机森林模型...")
        print(f"训练样本数: {len(X_train)}")
        print(f"特征数量: {X_train.shape[1]}")
        print(f"正常样本数: {np.sum(y_train == 0)}")
        print(f"故障样本数: {np.sum(y_train == 1)}")
        print(f"正常样本比例: {np.sum(y_train == 0) / len(y_train) * 100:.2f}%")
        print("-" * 50)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # 获取训练集预测与概率
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]

        evaluator = Evaluator(y_train, train_pred, train_proba)
        train_metrics = evaluator.results

        # 打印主要指标
        print(f"训练集准确率: {train_metrics.get('accuracy', None):.4f}")
        if 'auc' in train_metrics and train_metrics['auc'] is not None:
            print(f"训练集AUC: {train_metrics['auc']:.4f}")
        if 'TP' in train_metrics:
            print(f"TP: {train_metrics['TP']}, TN: {train_metrics['TN']}, FP: {train_metrics['FP']}, FN: {train_metrics['FN']}")

        print("训练完成！")

        return train_metrics

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 对于随机森林，我们直接使用predict方法，内部会使用0.5作为阈值
        predictions = self.model.predict(X)

        # 如果指定了不同的阈值，使用概率进行预测
        if threshold != 0.5:
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

        return predictions
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """
        获取特征重要性

        Args:
            top_k: 返回前k个重要特征（可选）

        Returns:
            importance_dict: 特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))

        # 按重要性排序
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # 如果指定了top_k，只返回前k个
        if top_k is not None:
            importance_dict = dict(list(importance_dict.items())[:top_k])

        return importance_dict

    def plot_feature_importance(self, top_k: int = 20, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性图

        Args:
            top_k: 显示前k个重要特征
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        importance_dict = self.get_feature_importance(top_k)

        # 转换为DataFrame
        importance_df = pd.DataFrame(list(importance_dict.items()),
                                    columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=True)

        plt.figure(figsize=figsize)

        # 水平条形图
        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                        color='skyblue', edgecolor='navy', alpha=0.7)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(importance_df['importance']) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)

        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                        param_grid: Optional[Dict] = None,
                        cv_folds: int = 5, scoring: str = 'f1') -> Dict:
        """
        超参数调优

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            param_grid: 参数网格（可选）
            cv_folds: 交叉验证折数
            scoring: 评分指标

        Returns:
            tuning_results: 调优结果
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

        print("开始超参数调优...")
        print(f"参数网格: {param_grid}")
        print("-" * 50)

        # 网格搜索
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        # 使用 GridSearchCV 的 best_estimator_ 作为最终模型（已在训练数据上 refit）
        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        self.is_fitted = True

        print("超参数调优完成！")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳{scoring}分数 (CV): {grid_search.best_score_:.4f}")

        # 计算最佳模型在训练集上的 F1 分数并作为 train_metric 返回
        train_pred = self.model.predict(X_train)
        try:
            train_proba = self.model.predict_proba(X_train)[:, 1]
        except Exception:
            train_proba = None

        train_metrics = evaluate_model(y_train, train_pred, train_proba)

        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'train_metrics': train_metrics
        }

        return tuning_results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        # 精简输出，仅返回必要的模型参数与状态
        info = {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'class_weight': self.class_weight
            },
            'n_features': len(self.feature_names) if self.feature_names else None,
            'is_fitted': self.is_fitted
        }

        if self.is_fitted:
            info['n_trees_actual'] = len(self.model.estimators_)

        return info