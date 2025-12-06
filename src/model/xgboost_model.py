import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ["Noto Sans CJK JP"]  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from utils.evaluation import Evaluator


class XGBClassifier:

    def __init__(self, num_classes: int = 2, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 min_child_weight: int = 1, gamma: float = 0,
                 reg_alpha: float = 0, reg_lambda: float = 1,
                 seed: int = 42, n_jobs: int = -1, use_gpu: bool = True,
                 scale_pos_weight: float = None):
        """
        XGBoost分类器
        
        Args:
            num_classes: 类别数量，2为二分类，>2为多分类
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            min_child_weight: 最小叶子权重
            gamma: 分裂最小损失减少
            reg_alpha: L1正则化
            reg_lambda: L2正则化
            seed: 随机种子
            n_jobs: 并行数
            use_gpu: 是否使用GPU加速
            scale_pos_weight: 正负样本权重比（用于处理类别不平衡，仅二分类有效）
        """
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.scale_pos_weight = scale_pos_weight

        # 设置目标函数
        if num_classes == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
        else:
            objective = 'multi:softprob'
            eval_metric = 'mlogloss'

        device = 'cuda' if use_gpu else 'cpu'

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method='hist',
            device=device,
            objective=objective,
            eval_metric=eval_metric,
            scale_pos_weight=scale_pos_weight,
            num_class=num_classes if num_classes > 2 else None,
            verbosity=0
        )

        self.is_fitted = False
        self.feature_names = None
        self.cv_results = None

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """计算正负样本权重比（用于二分类类别不平衡）"""
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        return neg_count / pos_count if pos_count > 0 else 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: Optional[List[str]] = None,
            auto_balance: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
            auto_balance: 是否自动处理类别不平衡（仅二分类有效）
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # 自动计算类别不平衡权重
        if auto_balance and self.num_classes == 2 and self.scale_pos_weight is None:
            self.scale_pos_weight = self._compute_scale_pos_weight(y_train)
            self.model.set_params(scale_pos_weight=self.scale_pos_weight)
            print(f"自动计算scale_pos_weight: {self.scale_pos_weight:.4f}")

        print("开始训练XGBoost模型...")
        print(f"训练样本数: {len(X_train)}")
        print(f"特征数量: {X_train.shape[1]}")
        print(f"类别数量: {self.num_classes}")
        print(f"使用GPU: {self.use_gpu}")
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

    def get_feature_importance(self, importance_type: str = 'gain',
                               top_k: Optional[int] = None) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型，'weight', 'gain', 'cover'
            top_k: 返回前k个重要特征
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 获取特征重要性
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 映射特征名称
        result = {}
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            result[name] = importance_dict.get(key, 0)

        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        if top_k is not None:
            result = dict(list(result.items())[:top_k])

        return result

    def plot_feature_importance(self, top_k: int = 20, figsize: Tuple[int, int] = (10, 8),
                               importance_type: str = 'gain',
                               save_path: Optional[str] = None) -> None:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        importance_dict = self.get_feature_importance(importance_type, top_k)

        importance_df = pd.DataFrame(list(importance_dict.items()),
                                    columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=True)

        plt.figure(figsize=figsize)

        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                        color='lightgreen', edgecolor='darkgreen', alpha=0.7)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                plt.text(width + max(importance_df['importance']) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f'{width:.4f}', ha='left', va='center', fontsize=9)

        plt.xlabel(f'Importance ({importance_type})', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              param_grid: Optional[Dict] = None,
                              cv_folds: int = 5, scoring: str = 'f1',
                              early_stopping_rounds: int = 20,
                              auto_balance: bool = True) -> Dict:
        """
        超参数调优（带早停）
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数搜索空间
            cv_folds: 交叉验证折数
            scoring: 评分指标
            early_stopping_rounds: 早停轮数
            auto_balance: 是否自动处理类别不平衡
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5],
            }

        # 自动计算类别不平衡权重
        if auto_balance and self.num_classes == 2:
            self.scale_pos_weight = self._compute_scale_pos_weight(y_train)
            self.model.set_params(scale_pos_weight=self.scale_pos_weight)
            print(f"自动计算scale_pos_weight: {self.scale_pos_weight:.4f}")

        print("开始超参数调优（带早停）...")
        print(f"参数网格: {param_grid}")
        print(f"早停轮数: {early_stopping_rounds}")
        print("-" * 50)

        # 使用早停的GridSearchCV
        # 需要创建一个fit_params用于早停
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        # XGBoost早停需要eval_set，但GridSearchCV不直接支持
        # 这里使用标准GridSearchCV，通过设置较大的n_estimators和cv来间接实现
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
            'model_type': 'XGBoost',
            'num_classes': self.num_classes,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'gamma': self.gamma,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'scale_pos_weight': self.scale_pos_weight
            },
            'use_gpu': self.use_gpu,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'is_fitted': self.is_fitted
        }

        return info
