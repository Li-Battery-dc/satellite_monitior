import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
import os
import json


class Dataloader:
    """数据加载和预处理类"""

    def __init__(self, data_root: str = "../data", data_name: Optional[str] = None, split_mode: str = "file"):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录
            data_name: 子系统名称，如果为None则加载所有子系统
            split_mode: 数据划分模式
                - 'file': 从 data_root/train/{dataset}/processed_train.csv 和 processed_test.csv 加载
                - 'dir': 从 data_root/train/{dataset}/processed_all.csv 和 data_root/test/{dataset}/processed_all.csv 加载
        """
        self.data_root = data_root
        self.split_mode = split_mode
        self.data = {}

        if data_name:
            self.load_data(data_name)
        else:
            for name in ['激光载荷', '供配电', '姿轨控']:
                self.load_data(name)

    def _prepare_features_targets(self, df: pd.DataFrame, fault_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和目标变量"""
        if fault_only:
            df = df[df['label'] != 0]
        X = df.drop(columns=['label']).values
        y = df['label'].values
        return X, y

    def load_data(self, data_name: str, fault_only: bool = False) -> Dict:
        """
        加载并处理单个子系统的数据

        Args:
            data_name: 子系统名称
            fault_only: 是否只加载故障样本（用于identification任务）
        Returns:
            data_dict: 用于训练的数据内容
        """

        # 根据 split_mode 加载数据
        if self.split_mode == "file":
            # file 模式：从 train 目录下的 processed_train.csv 和 processed_test.csv 加载
            base_path = os.path.join(self.data_root, "train", data_name)
            train_path = os.path.join(base_path, "processed_train.csv")
            test_path = os.path.join(base_path, "processed_test.csv")
        elif self.split_mode == "dir":
            # dir 模式：从 train/processed_all.csv 和 test/processed_all.csv 加载
            base_path = os.path.join(self.data_root, "train", data_name)  # 用于加载映射文件
            train_path = os.path.join(self.data_root, "train", data_name, "processed_all.csv")
            test_path = os.path.join(self.data_root, "test", data_name, "processed_all.csv")
        else:
            raise ValueError(f"不支持的 split_mode: {self.split_mode}，请使用 'file' 或 'dir'")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # 提取出标签和特征
        X_train, y_train = self._prepare_features_targets(train_df, fault_only)
        X_test, y_test = self._prepare_features_targets(test_df, fault_only)

        # 获取特征名称, 名义变量映射, 标签映射
        feature_names = [col for col in train_df.columns if col != 'label']
        with open(os.path.join(base_path, "enum_to_object.json"), 'r') as f:
            enum_to_object = json.load(f)

        # 加载标签映射 (itoa.json)
        itoa_path = os.path.join(base_path, "itoa.json")
        label_map = {}
        if os.path.exists(itoa_path):
            with open(itoa_path, 'r') as f:
                itoa_data = json.load(f)
            # 将字符串键转换为整数键
            if fault_only:
                label_map = {int(k): v for k, v in itoa_data.items() if int(k) != 0} # 0标签被移除
            else:
                label_map = {int(k): v for k, v in itoa_data.items()}

        
        
        object_features = list(enum_to_object.keys())
        
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'object_features': object_features,
            'enum_to_object_map': enum_to_object,
            'label_map': label_map,
            'use_pca': False,
            'data_name': data_name,
            'data_shapes': {
                'train': train_df.shape,
                'test': test_df.shape
            }
        }

        self.data[data_name] = data_dict

        print(f"成功加载数据集: {data_name}")

        return data_dict
    
    def _apply_pca(self, X, n_components: Optional[int] = None):
        """应用PCA降维"""
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 应用PCA
        pca_model = PCA(n_components=n_components)
        X_pca = pca_model.fit_transform(X_scaled)
        print(f"PCA降维结果:")
        print(f"- 原始特征维度: {X.shape[1]}")
        print(f"- 降维后特征维度: {X_pca.shape[1]}")
        print(f"- 解释方差比例: {np.sum(pca_model.explained_variance_ratio_):.4f}")

        return X_pca, pca_model, scaler # 返回相关model用于测试集

    def apply_pca_to_data(self, data_name: str, n_components: Optional[int] = 10) -> Dict: 
        """对指定数据集应用PCA降维"""
        if data_name not in self.data:
            raise ValueError(f"数据 {data_name} 尚未加载，请先调用 load_data")

        data_dict = self.data[data_name]
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']

        # 对训练和测试数据应用PCA
        # 测试数据使用训练数据的PCA模型进行转换
        X_train_pca, pca_model, scaler = self._apply_pca(X_train, n_components)
        X_test_pca = pca_model.transform(scaler.transform(X_test))

        # 更新数据字典
        data_dict['X_train'] = X_train_pca
        data_dict['X_test'] = X_test_pca
        data_dict['use_pca'] = True
        data_dict['pca_model'] = pca_model

        self.data[data_name] = data_dict

        return data_dict
    
    def get_data(self, data_name: str) -> Dict:
        """获取已处理的数据"""
        if data_name not in self.data:
            raise ValueError(f"数据 {data_name} 尚未加载，请先调用 load_and_process_data")
        return self.data[data_name]

    def get_all_data(self) -> Dict[str, Dict]:
        """获取所有已处理的数据"""
        return self.data
    
    def get_data_info(self, data_name: str) -> str:
        """获取数据信息摘要"""
        if data_name not in self.data:
            raise ValueError(f"数据 {data_name} 尚未加载")

        data = self.data[data_name]
        info = f"""
数据集信息: {data_name}
{'='*40}
数据形状:
- 训练集: {data['data_shapes']['train']}
- 测试集: {data['data_shapes']['test']}
特征维度: {data['X_train'].shape[1]}
use_PCA: {data['use_pca']}
            """
        return info
    
    def print_all_info(self):
        """打印所有数据集的信息摘要"""
        for data_name in self.data.keys():
            info = self.get_data_info(data_name)
            print(info)


if __name__ == "__main__":
    # 测试数据加载器类
    print("测试 file 模式加载单个数据集...")
    loader1 = Dataloader(data_root='../data', data_name='激光载荷', split_mode='file')
    data1 = loader1.get_data('激光载荷')
    print(f"加载成功: {data1['X_train'].shape}")

    print("\n测试 dir 模式加载所有数据集...")
    loader2 = Dataloader(data_root='../data', split_mode='dir')
    loader2.print_all_info()

    print("\n测试PCA..")
    loader1.apply_pca_to_data('激光载荷', n_components=10)
    data_pca = loader1.get_data('激光载荷')
    print(f"PCA降维后形状: {data_pca['X_train'].shape}")