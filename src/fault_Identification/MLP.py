import numpy as np
import sys
sys.path.append('..')

from model.mlp import MLPClassifier


class MLPIdentifier(MLPClassifier):

    def __init__(self, input_dim: int, num_classes: int, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batchNorm: bool = True,
                 lr: float = 0.001, seed: int = 42):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batchNorm=use_batchNorm,
            lr=lr,
            seed=seed
        )

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)
