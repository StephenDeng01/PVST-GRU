import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_random_sample_psr(psr_results_folder, psr_length=757):
    # 搜集所有CSV文件路径
    csv_files = []
    for root, _, files in os.walk(psr_results_folder):
        for f in files:
            if f.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        print("未找到任何CSV文件！")
        return

    # 随机选择一个CSV文件
    chosen_file = random.choice(csv_files)
    print(f"随机选择文件: {chosen_file}")

    df = pd.read_csv(chosen_file)
    if df.empty:
        print("文件为空！")
        return

    # 随机选择一行（一个心拍）
    sample_row = df.sample(n=1).iloc[0]
    beat_id = sample_row['beat_id'] if 'beat_id' in df.columns else '未知beat_id'
    label = sample_row['label'] if 'label' in df.columns else '未知label'

    # 12导联名称
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    plt.figure(figsize=(15, 10))
    for idx, lead in enumerate(leads, 1):
        # 构建导联对应列名列表
        cols = [f"{lead}_p{i+1}" for i in range(psr_length)]
        if not set(cols).issubset(sample_row.index):
            print(f"导联 {lead} 的列在该文件中不完整，跳过绘制")
            continue
        lead_data = sample_row[cols].values.astype(float)
        plt.subplot(4, 3, idx)
        plt.plot(lead_data, linewidth=1)
        plt.title(f"{lead}")
        plt.grid(True)
        plt.tight_layout()

    plt.suptitle(f"PSR特征归一化后 - 心拍ID: {beat_id}, 标签: {label}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    psr_folder = r"F:\heartbeat_csv\PSR\negative"  # PSR批量处理结果目录
    visualize_random_sample_psr(psr_folder, psr_length=757)
