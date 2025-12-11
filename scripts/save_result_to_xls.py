#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果处理脚本 - 将detection和identification任务的结果保存为Excel文件
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


class ResultProcessor:
    def __init__(self, result_base_path: str = "result_best", data_base_path: str = "data/train", save_path: str = "."):
        self.result_base_path = Path(result_base_path)
        self.data_base_path = Path(data_base_path)
        self.save_path = Path(save_path)
        # 确保保存路径存在
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.datasets = ["激光载荷", "供配电", "姿轨控"]
        self.methods = ["xgb"] # 只保存最佳模型的结果
        self.tasks = ["detection", "identification"]

    def load_fault_names(self, dataset: str) -> Dict[str, str]:
        """加载指定数据集的故障类型名称"""
        itoa_path = self.data_base_path / dataset / "itoa.json"
        if not itoa_path.exists():
            raise FileNotFoundError(f"故障类型文件不存在: {itoa_path}")

        with open(itoa_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_results(self, task: str, method: str, dataset: str) -> Dict[str, Any]:
        """加载指定任务、方法和数据集的结果"""
        result_path = self.result_base_path / task / method / dataset / "results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {result_path}")

        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_detection_results(self) -> pd.DataFrame:
        """处理detection任务的结果，提取所有参数"""
        detection_data = []

        for dataset in self.datasets:
            for method in self.methods:
                try:
                    result = self.load_results("detection", method, dataset)
                    row = {
                        "数据集": dataset,
                        "方法": method.upper(),
                        "准确率": result.get("accuracy", 0),
                        "真正例(TP)": result.get("TP", 0),
                        "真负例(TN)": result.get("TN", 0),
                        "假正例(FP)": result.get("FP", 0),
                        "假负例(FN)": result.get("FN", 0),
                        "虚警率": result.get("虚警率", 0),
                        "漏警率": result.get("漏警率", 0),
                        "AUC": result.get("auc", 0)
                    }
                    detection_data.append(row)
                except FileNotFoundError:
                    print(f"警告: 未找到检测任务结果 - {dataset}/{method}")
                    continue

        return pd.DataFrame(detection_data)

    def process_identification_results(self) -> Dict[str, pd.DataFrame]:
        """处理identification任务的结果，生成混淆矩阵表格"""
        identification_results = {}

        for dataset in self.datasets:
            # 获取故障类型名称
            fault_names = self.load_fault_names(dataset)
            # 移除"正常"类型，因为identification任务不包含正常样本
            fault_names_filtered = {k: v for k, v in fault_names.items() if k != "0"}

            # 创建该数据集的DataFrame
            identification_data = []

            for method in self.methods:
                try:
                    result = self.load_results("identification", method, dataset)
                    confusion_matrix = np.array(result.get("confusion_matrix", []))

                    if confusion_matrix.size == 0:
                        continue

                    # 为每种故障类型创建一行数据
                    for i, (fault_idx, fault_name) in enumerate(fault_names_filtered.items()):
                        row = {
                            "数据集": dataset,
                            "方法": method.upper(),
                            "故障类型": fault_name,
                        }

                        # 对角线元素为预测正确的样本数
                        if i < len(confusion_matrix):
                            row["该类总样本数"] = np.sum(confusion_matrix[i, :])
                            row["预测正确"] = confusion_matrix[i, i]
                            # 计算分类错误的总和
                            row["预测错误"] = np.sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
                            row["准确率"] = (confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
                                           if np.sum(confusion_matrix[i, :]) > 0 else 0)
                            


                        identification_data.append(row)

                except FileNotFoundError:
                    print(f"警告: 未找到识别任务结果 - {dataset}/{method}")
                    continue

            if identification_data:
                identification_results[dataset] = pd.DataFrame(identification_data)

        return identification_results

    def save_to_excel(self, output_file: str = "satellite_monitor_results.xlsx"):
        """将所有结果保存到Excel文件"""
        # 使用 save_path 中指定的目录
        full_output_path = self.save_path / output_file
        with pd.ExcelWriter(full_output_path, engine='openpyxl') as writer:
            # 处理并保存detection结果
            print("处理检测任务结果...")
            detection_df = self.process_detection_results()
            detection_df.to_excel(writer, sheet_name="检测结果", index=False)
            print(f"检测结果已保存，共{len(detection_df)}条记录")

            # 处理并保存identification结果
            print("处理识别任务结果...")
            identification_results = self.process_identification_results()

            for dataset, df in identification_results.items():
                # Excel sheet名称不能超过31个字符
                sheet_name = f"识别结果_{dataset}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"识别结果({dataset})已保存，共{len(df)}条记录")

        print(f"\n所有结果已保存到: {full_output_path}")
        return str(full_output_path)

    def generate_summary_report(self, output_file: str = "results_summary.txt"):
        """生成文本格式的汇总报告"""
        # 使用 save_path 中指定的目录
        full_output_path = self.save_path / output_file
        with open(full_output_path, 'w', encoding='utf-8') as f:
            f.write("卫星监测任务结果汇总报告\n")
            f.write("=" * 50 + "\n\n")

            # Detection结果汇总
            f.write("1. 检测任务结果汇总\n")
            f.write("-" * 30 + "\n")
            detection_df = self.process_detection_results()

            for dataset in self.datasets:
                dataset_data = detection_df[detection_df["数据集"] == dataset]
                if not dataset_data.empty:
                    f.write(f"\n{dataset}数据集:\n")
                    for _, row in dataset_data.iterrows():
                        f.write(f"  {row['方法']}: 准确率={row['准确率']:.4f}, "
                               f"虚警率={row['虚警率']:.4f}, "
                               f"漏警率={row['漏警率']:.4f}, "
                               f"AUC={row['AUC']:.4f}\n")

            # Identification结果汇总
            f.write(f"\n\n2. 识别任务结果汇总\n")
            f.write("-" * 30 + "\n")
            identification_results = self.process_identification_results()

            for dataset, df in identification_results.items():
                f.write(f"\n{dataset}数据集:\n")
                fault_names = self.load_fault_names(dataset)
                fault_names_filtered = {k: v for k, v in fault_names.items() if k != "0"}

                for method in ["RF", "MLP", "XGB"]:
                    method_data = df[df["方法"] == method]
                    if not method_data.empty:
                        total_correct = method_data["预测正确"].sum()
                        total_samples = method_data["该类总样本数"].sum()
                        accuracy = total_correct / total_samples if total_samples > 0 else 0
                        f.write(f"  {method}: 总体准确率={accuracy:.4f} "
                               f"({total_correct}/{total_samples})\n")

        print(f"汇总报告已保存到: {full_output_path}")
        return str(full_output_path)


def main():
    """主函数"""
    print("开始处理卫星监测任务结果...")

    # 创建结果处理器
    processor = ResultProcessor(
        result_base_path="./result_best", 
        data_base_path="./data/train/", 
        save_path="./excel_results"
    )

    # 保存Excel结果
    excel_file = processor.save_to_excel()

    # 生成汇总报告
    report_file = processor.generate_summary_report()

    print("\n处理完成!")
    print(f"Excel文件: {excel_file}")
    print(f"汇总报告: {report_file}")


if __name__ == "__main__":
    main()