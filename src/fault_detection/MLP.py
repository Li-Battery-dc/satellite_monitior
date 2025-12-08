import numpy as np
import sys
sys.path.append('..')

from model.mlp import MLPClassifier


class MLPDetector(MLPClassifier):

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batchNorm: bool = True,
                 lr: float = 0.001, seed: int = 42):
        super().__init__(
            input_dim=input_dim,
            num_classes=2, # 借口兼容加载模式，但是num_classes固定为2。
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batchNorm=use_batchNorm,
            lr=lr,
            seed=seed
        )

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)