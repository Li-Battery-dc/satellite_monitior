import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt


class MLPNetwork(nn.Module):
    """MLP神经网络"""

    def __init__(self, input_dim: int, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """
        初始化MLP网络

        Args:
            input_dim: 输入特征维度
            hidden_layers: 隐藏层神经元数量列表
            dropout_rate: Dropout比例
            use_batch_norm: 是否使用批归一化
        """
        super(MLPNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()

        # 构建网络层
        layer_dims = [input_dim] + hidden_layers + [1]  # 最后一层是输出层

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # 除了输出层外，都添加激活函数、批归一化和Dropout
            if i < len(layer_dims) - 2:
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))
                self.dropouts.append(nn.Dropout(dropout_rate))

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # 如果不是输出层
            if i < len(self.layers) - 1:
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropouts[i](x)

        # 输出层不加激活函数，配合BCEWithLogitsLoss使用
        return x.squeeze(-1)


class MLPDetector:
    """MLP故障检测器"""

    def __init__(self, input_dim: int, hidden_layers: list = [64, 32],
                 dropout_rate: float = 0.3, use_batchNorm: bool = True,
                 lr: float = 0.001, seed: int = 42):
        """
        初始化MLP检测器

        Args:
            input_dim: 输入特征维度
            hidden_layers: 隐藏层神经元数量列表
            dropout_rate: Dropout比例
            use_batch_norm: 是否使用批归一化
            learning_rate: 学习率
            random_state: 随机种子
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batchNorm = use_batchNorm
        self.lr = lr
        self.seed = seed

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"training on : {self.device}")

        # 初始化网络并移动到设备
        self.network = MLPNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batchNorm
        ).to(self.device)

        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.is_fitted = False

    def get_torchloader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[DataLoader, Optional[DataLoader]]:
        """使用torch.utils.data.DataLoader支持batch training"""
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1))

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = True) -> Dict:
        """
        训练模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            training_history: 训练历史
        """
        train_dataloader = self.get_torchloader(X_train, y_train, batch_size)

        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_dataloader = self.get_torchloader(X_val, y_val, batch_size)

        self.train_losses = []
        self.val_losses = []

        if verbose:
            print("开始训练MLP模型...")
            print(f"训练样本数: {len(X_train)}")
            if X_val is not None:
                print(f"验证样本数: {len(X_val)}")
            print(f"网络结构: {self.input_dim} -> {' -> '.join(map(str, self.hidden_layers))} -> 1")
            print("-" * 50)

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
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

            # 验证阶段
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

            # 打印训练进度
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
        """
        预测类别

        Args:
            X: 特征数据
            threshold: 分类阈值

        Returns:
            predictions: 预测类别
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.network(X_tensor)
            probabilities = torch.sigmoid(logits)  # 将logits转换为概率
            predictions = (probabilities >= threshold).float().cpu().numpy().flatten()

        return predictions.astype(int)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        输出每个样本属于正类的概率（用于绘制ROC曲线）

        Args:
            X: 特征数据

        Returns:
            probabilities: 每个样本属于正类的概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.network(X_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        return probabilities

    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4),
                             save_path: Optional[str] = None) -> None:
        """
        绘制训练历史

        Args:
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if not self.train_losses:
            print("没有训练历史可显示")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 训练损失
        axes[0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 验证损失（如果有）
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
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        info = {
            'model_type': 'MLP',
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batchNorm,
            'learning_rate': self.lr,
            'is_fitted': self.is_fitted
        }

        return info