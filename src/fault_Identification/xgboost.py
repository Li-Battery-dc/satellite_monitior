import numpy as np
import xgboost as xgb
from typing import Tuple, Optional, List
from sklearn.model_selection import GridSearchCV

from model.xgboost_model import XGBClassifier 

class XGBoostIdentifier(XGBClassifier):

    def __init__(self, num_classes: int, 
                 confused_pair: Optional[Tuple[int, int]] = None,
                 refiner_params: Optional[dict] = None,
                 **kwargs):
        """
        Args:
            num_classes: 类别总数
            confused_pair: 一个元组 (class_A, class_B)，指定需要单独处理的两个混淆类别ID
            refiner_params: 专门用于精修模型的参数字典
            **kwargs: 传递给父类的参数
        """
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.confused_pair = confused_pair
        self.refiner_model = None
        self.refiner_params = refiner_params if refiner_params else kwargs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: Optional[List[str]] = None,
            auto_balance: bool = True) -> dict:
        
        # 1. 训练主模型 (调用父类逻辑)
        print(">>> 阶段 1: 训练主分类器 (Global Multi-class)")
        metrics = super().fit(X_train, y_train, feature_names, auto_balance)
        
        # 2. 训练精修模型 (如果有指定的混淆对)
        if self.confused_pair:
            c1, c2 = self.confused_pair
            print(f"\n>>> 阶段 2: 训练精修分类器 (Refiner) 针对类别 {c1} 和 {c2}")
            
            # 筛选只包含这两个类别的样本
            mask = np.isin(y_train, [c1, c2])
            X_sub = X_train[mask]
            y_sub = y_train[mask]
            
            if len(y_sub) == 0:
                print("警告: 训练集中不存在指定的混淆类别样本，跳过精修模型训练。")
                self.refiner_model = None
                return metrics

            # 将标签重映射为 0 和 1 (0对应c1, 1对应c2)
            y_sub_binary = (y_sub == c2).astype(int)
            
            # 初始化并训练精修模型 (使用原生XGB以避免递归调用父类逻辑)
            # 从 refiner_params 中提取有效的 XGBoost 参数
            refiner_args = {k: v for k, v in self.refiner_params.items() 
                           if k in ['n_estimators', 'max_depth', 'learning_rate', 
                                    'subsample', 'colsample_bytree', 'gamma', 
                                    'reg_alpha', 'reg_lambda', 'min_child_weight',
                                    'n_jobs', 'random_state']}
            
            # 添加固定参数
            refiner_args.update({
                'random_state': self.seed,
                'n_jobs': self.n_jobs,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'device': 'cuda' if self.use_gpu else 'cpu',
                'verbosity': 0
            })
            
            self.refiner_model = xgb.XGBClassifier(**refiner_args)
            
            self.refiner_model.fit(X_sub, y_sub_binary)
            
            # 简单评估精修效果
            acc = self.refiner_model.score(X_sub, y_sub_binary)
            print(f"精修模型 (Binary {c1} vs {c2}) 内部准确率: {acc:.4f}")
            
        return metrics

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        # 1. 获取主模型的预测
        main_preds = super().predict(X, threshold)
        
        # 2. 如果存在精修模型，对特定预测结果进行“拦截修正”
        if self.refiner_model and self.confused_pair:
            c1, c2 = self.confused_pair
            
            # 找出主模型认为是 c1 或 c2 的样本索引
            mask_indices = np.where(np.isin(main_preds, [c1, c2]))[0]
            
            if len(mask_indices) > 0:
                X_confused = X[mask_indices]
                
                # 精修模型预测 (0 -> c1, 1 -> c2)
                refiner_preds_binary = self.refiner_model.predict(X_confused)
                
                # 将 0/1 映射回原始类别 ID
                refined_preds = np.where(refiner_preds_binary == 0, c1, c2)
                
                # 更新最终预测结果
                main_preds[mask_indices] = refined_preds
        
        return main_preds