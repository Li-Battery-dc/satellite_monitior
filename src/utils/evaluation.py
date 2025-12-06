import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ["Noto Sans CJK JP"]  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import seaborn as sns
import json
import os
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc
)
from typing import Dict, Optional, Tuple, Union, List

class Evaluator:
   
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray ,
                    label_map: Optional[Dict[int, str]] = None) -> Dict:
        """
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 正类的预测概率（仅二分类时使用，可选）
            label_map: 标签到名称的映射字典

        self.results: 包含评估结果的字典，包括：
            - accuracy: 准确率
            - confusion_matrix: 混淆矩阵
            - TP, TN, FP, FN (二分类时)
            - 若提供 y_proba: fpr, tpr, auc
        """
        results: Dict = {}

        # 基本指标
        results['accuracy'] = float(accuracy_score(y_true, y_pred))

        # 混淆矩阵
        labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = np.sort(labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results['confusion_matrix'] = cm
        results['labels'] = labels  # 保存实际的标签值用于绘图

        # 二分类时，返回 TP/TN/FP/FN 以及单阈值的 TPR/FPR
        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            results.update({'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)})

            # 若提供了概率，则计算完整 ROC 与 AUC
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            results['fpr'] = fpr
            results['tpr'] = tpr
            results['auc'] = float(roc_auc)

        self.results = results
        self.label_map = label_map

    def print_results(self) -> Dict:
        """打印相关信息"""
        print("评估结果:")
        print(f"准确率: {self.results.get('accuracy', None)}")

        if 'TP' in self.results:
            print(f"TP: {self.results['TP']}, TN: {self.results['TN']}, FP: {self.results['FP']}, FN: {self.results['FN']}")

        if 'auc' in self.results and self.results['auc'] is not None:
            print(f"AUC: {self.results['auc']}")

        return self.results
        
    def plot_confusion_matrix(self,
                            title: str = "Confusion Matrix",
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        绘制混淆矩阵热力图（支持多分类）

        Args:
            cm: 混淆矩阵数组
            label_map: 标签到名称的映射字典，例如 {0: 'Normal', 1: 'Fault A', 2: 'Fault B'}
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        cm = self.results['confusion_matrix']
        labels = self.results['labels']  # 使用实际的标签值而非矩阵索引

        # 确定类别名称
        if self.label_map is not None:
            class_names = [self.label_map.get(label, f'Class {label}') for label in labels]
        else:
            class_names = [f'Class {label}' for label in labels]
        
        plt.figure(figsize=figsize)

        # 创建热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    square=True, cbar_kws={'shrink': 0.8},
                    xticklabels=class_names, yticklabels=class_names)

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else: 
            plt.show()


    def plot_roc_curve(self,
                    title: str = "ROC Curve", figsize: Tuple[int, int] = (8, 6),
                    save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        绘制二分类ROC曲线

        Args:
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if 'fpr' not in self.results or 'tpr' not in self.results or 'auc' not in self.results:
            raise ValueError("ROC曲线仅适用于二分类且需提供预测概率")
        fpr = self.results['fpr']
        tpr = self.results['tpr']
        roc_auc = self.results['auc']

        plt.figure(figsize=figsize)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')

        # 绘制随机分类器线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5000)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else: 
            plt.show()

    def save_results(self, save_dir: str, prefix: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        results_to_save = {
            'accuracy': self.results.get('accuracy'),
            'confusion_matrix': self.results['confusion_matrix'].tolist(),
        }
        
        if 'TP' in self.results:
            results_to_save.update({
                'TP': self.results['TP'],
                'TN': self.results['TN'],
                'FP': self.results['FP'],
                'FN': self.results['FN'],
            })
        
        if 'auc' in self.results:
            results_to_save['auc'] = self.results['auc']
        
        json_path = os.path.join(save_dir, f"{prefix}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        cm_path = os.path.join(save_dir, f"{prefix}_confusion_matrix.png")
        self.plot_confusion_matrix(save_path=cm_path)
        
        if 'auc' in self.results:
            roc_path = os.path.join(save_dir, f"{prefix}_roc_curve.png")
            self.plot_roc_curve(save_path=roc_path)
        
        print(f"结果已保存到: {save_dir}")