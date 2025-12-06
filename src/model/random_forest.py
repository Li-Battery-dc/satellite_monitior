import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt

from utils.evaluation import Evaluator


class RFClassifier:

    def __init__(self, num_classes: int = 2, n_estimators: int = 100, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', seed: int = 42,
                 n_jobs: int = -1, class_weight: str = 'balanced'):
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.n_jobs = n_jobs
        self.class_weight = class_weight

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
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        print("开始训练随机森林模型...")
        print(f"训练样本数: {len(X_train)}")
        print(f"特征数量: {X_train.shape[1]}")
        print(f"类别数量: {self.num_classes}")
        print("-" * 50)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        train_pred = self.model.predict(X_train)
        train_proba = self._predict_prob_all(X_train)
        
        if self.num_classes == 2:
            evaluator = Evaluator(y_train, train_pred, train_proba)
        else:
            evaluator = Evaluator(y_train, train_pred, None)
        train_metrics = evaluator.results

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

        if self.num_classes == 2 and threshold != 0.5:
            probabilities = self._predict_prob_all(X)
            predictions = (probabilities >= threshold).astype(int)
        else:
            predictions = self.model.predict(X)

        return predictions

    def _predict_prob_all(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        proba = self.model.predict_proba(X)
        if self.num_classes == 2:
            return proba[:, 1]
        return proba

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)

    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))

        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        if top_k is not None:
            importance_dict = dict(list(importance_dict.items())[:top_k])

        return importance_dict

    def plot_feature_importance(self, top_k: int = 20, figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        importance_dict = self.get_feature_importance(top_k)

        importance_df = pd.DataFrame(list(importance_dict.items()),
                                    columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=True)

        plt.figure(figsize=figsize)

        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                        color='skyblue', edgecolor='navy', alpha=0.7)

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

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        self.model = best_estimator
        self.is_fitted = True

        print("超参数调优完成！")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳{scoring}分数 (CV): {grid_search.best_score_:.4f}")

        train_pred = self.model.predict(X_train)
        train_proba = self._predict_prob_all(X_train)

        if self.num_classes == 2:
            evaluator = Evaluator(y_train, train_pred, train_proba)
        else:
            evaluator = Evaluator(y_train, train_pred, None)
        train_metrics = evaluator.results

        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'train_metrics': train_metrics
        }

        return tuning_results

    def get_model_info(self) -> Dict:
        info = {
            'model_type': 'RandomForest',
            'num_classes': self.num_classes,
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
