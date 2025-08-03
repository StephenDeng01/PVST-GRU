import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_specified_file(csv_path, sample_idx=0):
    """
    绘制指定CSV文件中特定样本的12导联PCA特征图像
    :param csv_path: 处理后的CSV文件路径（完整路径）
    :param sample_idx: 要可视化的样本索引（默认第0个样本）
    """
    # 读取文件
    try:
        df = pd.read_csv(csv_path, header=None, dtype=np.float64)
    except FileNotFoundError:
        print(f"错误：文件 {csv_path} 不存在")
        return
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return

    # 验证样本索引有效性
    if sample_idx < 0 or sample_idx >= len(df):
        print(f"错误：样本索引无效，该文件共有 {len(df)} 个样本（索引0到{len(df) - 1}）")
        return

    # 提取样本数据
    sample = df.iloc[sample_idx]
    beat_id = sample[0]
    features = sample[1:]  # 排除第一列的beat_id

    # 创建画布
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'文件：{csv_path.split(os.sep)[-1]} 心拍编号：{beat_id:.0f}', fontsize=16)

    # 绘制12个导联
    for lead in range(12):
        row, col = lead // 4, lead % 4
        ax = axes[row, col]

        # 提取当前导联的757个特征点
        start = lead * 757
        end = start + 757
        lead_data = features[start:end].values

        # 绘图
        ax.plot(lead_data, color='#1f77b4', linewidth=1.2)
        ax.set_title(f'导联 {lead + 1}', fontsize=12)
        ax.set_xlabel('样本点 (0-756)')
        ax.set_ylabel('归一化PCA特征')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)  # 限制Y轴范围，突出细节

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，避免标题被遮挡
    plt.show()


if __name__ == "__main__":
    # --------------------------
    # 请修改以下两个参数
    # --------------------------
    # 要可视化的CSV文件完整路径
    target_csv = r"F:\heartbeat_csv\SEE\negative\关秋雪_denoised.csv"  # 例如：r"F:\heartbeat_csv\SEE\patient1.csv"
    # 要查看的样本索引（从0开始）
    target_sample_idx = 3

    # 调用绘图函数
    plot_specified_file(target_csv, target_sample_idx)
