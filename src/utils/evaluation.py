import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc
)
from typing import Dict, Optional, Tuple, Union, List


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算预测准确率"""
    return accuracy_score(y_true, y_pred)


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    计算二分类的 TP, TN, FP, FN

    Args:
        y_true: 真实标签 (0 或 1)
        y_pred: 预测标签 (0 或 1)

    Returns:
        包含 TP, TN, FP, FN 的字典
    """
    if len(np.unique(y_true)) > 2 or len(np.unique(y_pred)) > 2:
        raise ValueError("此函数仅支持二分类，请确保标签只有两个类别")
    
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()

    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         label_map: Optional[Dict[int, str]] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> np.ndarray:
    """
    绘制混淆矩阵热力图（支持多分类）

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_map: 标签到名称的映射字典，例如 {0: 'Normal', 1: 'Fault A', 2: 'Fault B'}
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径（可选）

    Returns:
        cm: 混淆矩阵
    """
    # 获取所有唯一标签并排序
    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.sort(labels)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 确定类别名称
    if label_map is not None:
        class_names = [label_map.get(label, f'Class {label}') for label in labels]
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
    plt.show()
    
    return cm


def calculate_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算二分类的ROC曲线和AUC

    Args:
        y_true: 真实标签 (0 或 1)
        y_proba: 正类的预测概率

    Returns:
        fpr: 假正率
        tpr: 真正率
        roc_auc: AUC值
    """
    if len(np.unique(y_true)) > 2:
        raise ValueError("此函数仅支持二分类，请确保标签只有两个类别")
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  title: str = "ROC Curve", figsize: Tuple[int, int] = (8, 6),
                  save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    绘制二分类ROC曲线

    Args:
        y_true: 真实标签 (0 或 1)
        y_proba: 正类的预测概率
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径（可选）

    Returns:
        fpr: 假正率
        tpr: 真正率
        roc_auc: AUC值
    """
    fpr, tpr, roc_auc = calculate_roc_auc(y_true, y_proba)

    plt.figure(figsize=figsize)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')

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
    plt.show()

    return fpr, tpr, roc_auc


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None,
                  label_map: Optional[Dict[int, str]] = None) -> Dict:
    """
    评估模型的性能

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 正类的预测概率（仅二分类时使用，可选）
        label_map: 标签到名称的映射字典

    Returns:
        results: 评估结果字典
    """
    results = {}

    # 计算准确率
    results['accuracy'] = calculate_accuracy(y_true, y_pred)

    # 计算混淆矩阵
    labels = np.unique(np.concatenate([y_true, y_pred]))
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 如果是二分类，计算 TP, TN, FP, FN
    if len(np.unique(y_true)) == 2:
        binary_metrics = calculate_binary_metrics(y_true, y_pred)
        results.update(binary_metrics)
        
        # 计算ROC和AUC（如果提供了概率）
        if y_proba is not None:
            try:
                fpr, tpr, roc_auc = calculate_roc_auc(y_true, y_proba)
                results['fpr'] = fpr
                results['tpr'] = tpr
                results['auc'] = roc_auc
            except Exception as e:
                print(f"ROC计算失败: {e}")
                results['auc'] = None

    return results


if __name__ == "__main__":
    # 测试代码
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    print("=" * 50)
    print("测试二分类场景")
    print("=" * 50)
    
    # 生成二分类测试数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                             n_redundant=5, n_clusters_per_class=1,
                             weights=[0.9, 0.1], flip_y=0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                       random_state=42, stratify=y)

    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # 评估模型
    label_map_binary = {0: 'Normal', 1: 'Fault'}
    results = evaluate_model(y_test, y_pred, y_proba, label_map_binary)
    print("评估结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"TP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")

    # 绘制ROC曲线
    plot_roc_curve(y_test, y_proba, title="Binary Classification ROC Curve")

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, label_map=label_map_binary, title="Binary Confusion Matrix")
    
    print("\n" + "=" * 50)
    print("测试多分类场景")
    print("=" * 50)
    
    # 生成多分类测试数据
    X_multi, y_multi = make_classification(n_samples=1000, n_features=20, n_informative=10,
                                           n_redundant=5, n_classes=4, n_clusters_per_class=1,
                                           random_state=42)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.3,
                                                                 random_state=42, stratify=y_multi)

    # 训练多分类模型
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_multi.fit(X_train_m, y_train_m)
    y_pred_m = rf_multi.predict(X_test_m)

    # 使用 label_map 绘制多分类混淆矩阵
    label_map_multi = {0: 'Normal', 1: 'Fault A', 2: 'Fault B', 3: 'Fault C'}
    plot_confusion_matrix(y_test_m, y_pred_m, label_map=label_map_multi, 
                         title="Multi-class Confusion Matrix")
    
    # 不使用 label_map 绘制
    plot_confusion_matrix(y_test_m, y_pred_m, title="Multi-class Confusion Matrix (No Label Map)")