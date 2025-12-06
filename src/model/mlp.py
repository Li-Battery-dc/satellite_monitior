import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt


class MLPNetwork(nn.Module):

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super(MLPNetwork, self).__init__()

        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()

        # 输出维度：二分类用1，多分类用num_classes
        output_dim = 1 if num_classes == 2 else num_classes
        layer_dims = [input_dim] + hidden_layers + [output_dim]

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < len(layer_dims) - 2:
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))
                self.dropouts.append(nn.Dropout(dropout_rate))

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropouts[i](x)

        if self.num_classes == 2:
            return x.squeeze(-1)
        return x


class MLPClassifier:

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batchNorm: bool = True,
                 lr: float = 0.001, seed: int = 42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batchNorm = use_batchNorm
        self.lr = lr
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"training on : {self.device}")

        self.network = MLPNetwork(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batchNorm
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        if num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.is_fitted = False

    def _get_torchloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        X_tensor = torch.FloatTensor(X)
        if self.num_classes == 2:
            y_tensor = torch.FloatTensor(y.reshape(-1))
        else:
            y_tensor = torch.LongTensor(y.reshape(-1))

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = True) -> Dict:
        train_dataloader = self._get_torchloader(X_train, y_train, batch_size)

        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_dataloader = self._get_torchloader(X_val, y_val, batch_size)

        self.train_losses = []
        self.val_losses = []

        if verbose:
            print("开始训练MLP模型...")
            print(f"训练样本数: {len(X_train)}")
            if X_val is not None:
                print(f"验证样本数: {len(X_val)}")
            output_dim = 1 if self.num_classes == 2 else self.num_classes
            print(f"网络结构: {self.input_dim} -> {' -> '.join(map(str, self.hidden_layers))} -> {output_dim}")
            print("-" * 50)

        for epoch in range(epochs):
            self.network.train()
            train_loss = 0.0

            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(batch_X)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)
            self.train_losses.append(avg_train_loss)

            val_loss = None
            if val_dataloader is not None:
                self.network.eval()
                val_loss_sum = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.network(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss_sum += loss.item()

                val_loss = val_loss_sum / len(val_dataloader)
                self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.6f}")

        self.is_fitted = True

        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_trained': epoch + 1
        }

        if verbose:
            print(f"训练完成！总共训练 {epoch + 1} 轮")

        return training_history

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)

            if self.num_classes == 2:
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= threshold).float().cpu().numpy().flatten()
            else:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return predictions.astype(int)

    def _predict_prob_all(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.network(X_tensor)

            if self.num_classes == 2:
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return self._predict_prob_all(X)

    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4),
                             save_path: Optional[str] = None) -> None:
        if not self.train_losses:
            print("没有训练历史可显示")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        if self.val_losses:
            axes[1].plot(self.train_losses, label='Training Loss', color='blue')
            axes[1].plot(self.val_losses, label='Validation Loss', color='red')
            axes[1].set_title('Training and Validation Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No validation data available',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Validation Loss')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        info = {
            'model_type': 'MLP',
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_layers': self.hidden_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batchNorm,
            'learning_rate': self.lr,
            'is_fitted': self.is_fitted
        }

        return info
