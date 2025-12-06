import numpy as np
import sys
sys.path.append('..')

from model.random_forest import RFClassifier


class RandomForestIdentifier(RFClassifier):

    def __init__(self, num_classes: int, n_estimators: int = 100, max_depth=None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', seed: int = 42,
                 n_jobs: int = -1, class_weight: str = 'balanced'):
        super().__init__(
            num_classes=num_classes,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            seed=seed,
            n_jobs=n_jobs,
            class_weight=class_weight
        )

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)
