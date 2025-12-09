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
            refiner_params: 专门用于精修模型的参数字典（可选，默认沿用主模型参数）
            **kwargs: 传递给父类的参数
        """
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.confused_pair = confused_pair
        self.refiner_model = None
        self.refiner_params = refiner_params if refiner_params else kwargs
        
        # 确保精修模型是二分类逻辑
        self.refiner_params = self.refiner_params.copy()
        self.refiner_params['objective'] = 'binary:logistic'
        self.refiner_params['num_class'] = None # 清除多分类参数

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
            # refiner_args = {k: v for k, v in self.refiner_params.items() 
            #                if k in ['n_estimators', 'max_depth', 'learning_rate', 
            #                         'subsample', 'colsample_bytree', 'gamma', 
            #                         'reg_alpha', 'reg_lambda', 'n_jobs', 'random_state']}
            
            # refiner_args['max_depth'] = refiner_args.get('max_depth', 6) + 1 
            
            # 不用大分类器的参数，在单独二分类上要求完全不同
            refiner_args = {
                'max_depth': 6,          
                'learning_rate': 0.02,    
                'n_estimators': 500,      
                'min_child_weight': 3,    
                'reg_lambda': 5,         
                'subsample': 0.7,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'n_jobs': self.n_jobs,
                'random_state': self.seed
            }
            self.refiner_model = xgb.XGBClassifier(
                **refiner_args,
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                device='cuda' if self.use_gpu else 'cpu'
            )
            
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