import numpy as np
import sys
sys.path.append('..')

from model.xgboost_model import XGBClassifier


class XGBoostDetector(XGBClassifier):

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, min_child_weight: int = 1,
                 gamma: float = 0, reg_alpha: float = 0, reg_lambda: float = 1,
                 seed: int = 42, n_jobs: int = -1, use_gpu: bool = True,
                 scale_pos_weight: float = None):
        super().__init__(
            num_classes=2,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            seed=seed,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            scale_pos_weight=scale_pos_weight
        )

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)
