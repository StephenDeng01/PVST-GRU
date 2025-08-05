import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from OCCO import OCCO

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_data(filename):
    """读取JSON文件并提取ECG数据（保持不变）"""
    try:
        with open(filename, 'r') as f:
            ecg_data = json.load(f)

        if "beats" not in ecg_data:
            print("错误: JSON文件中未找到外层的'beats'关键字")
            return None, None

        beat_content = ecg_data["beats"]
        raw_leads = {}
        processed_leads = {}
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        for beat_id, beat_data in beat_content.items():
            for lead in leads:
                if lead in beat_data:
                    lead_data = beat_data[lead]
                    if isinstance(lead_data, str):
                        data_values = list(map(float, lead_data.split()))
                    elif isinstance(lead_data, list):
                        data_values = [float(val) for val in lead_data]
                    else:
                        print(f"警告: 导联 {lead} 在beat {beat_id} 中的数据格式不支持")
                        continue

                    key = f"{lead}_{beat_id}"
                    raw_series = pd.Series(data_values)
                    raw_leads[key] = raw_series
                    processed_series = OCCO.MC(raw_series)
                    processed_leads[key] = processed_series
                else:
                    print(f"警告: 导联 {lead} 在beat {beat_id} 中未找到")

        if not raw_leads:
            print("警告: 未解析到任何有效数据")
            return None, None

        return processed_leads, raw_leads

    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        return None, None


def test_single_file(file_path):
    """测试单个ECG文件的处理（调用三行四列可视化）"""
    print(f"正在处理文件: {file_path}")
    print("=" * 50)

    processed_leads, raw_leads = get_data(file_path)

    if processed_leads is None or raw_leads is None:
        print("文件处理失败，请检查文件路径和格式")
        return

    print(f"成功处理 {len(processed_leads)} 个导联数据")
    print(f"导联列表: {list(processed_leads.keys())}")
    print()

    # 提取十二导联数据（按三行四列顺序）
    standard_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    beat_id = list(processed_leads.keys())[0].split('_')[1]  # 获取第一个心跳的ID
    lead_data = {}
    for lead in standard_leads:
        key = f"{lead}_{beat_id}"
        if key in raw_leads and key in processed_leads:
            lead_data[lead] = {
                'raw': raw_leads[key],
                'processed': processed_leads[key]
            }
        else:
            print(f"警告: 未找到导联 {lead} 的数据，可能影响可视化")

    # 保存处理结果
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    for lead_name, data in processed_leads.items():
        csv_filename = f"{output_dir}/{lead_name}_occo.csv"
        data.to_csv(csv_filename, index=False)
        print(f"数据已保存到: {csv_filename}")

    # 可视化三行四列布局的十二导联
    visualize_twelve_leads(lead_data, beat_id, output_dir)

    return processed_leads, raw_leads


def visualize_twelve_leads(lead_data, beat_id, output_dir):
    """三行四列布局的十二导联可视化（原始+处理后信号）"""
    # 创建3行4列的子图（临床常用布局）
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)  # 宽度20，高度12更适合三行布局
    fig.suptitle(f'十二导联ECG信号对比（原始 vs OCCO处理后）- 心跳ID: {beat_id}', fontsize=16)

    # 三行四列的导联顺序（按临床习惯排列）
    lead_order = [
        ['I', 'II', 'III', 'AVR'],  # 第一行：标准肢体导联+AVR
        ['AVL', 'AVF', 'V1', 'V2'],  # 第二行：加压肢体导联+胸导联V1-V2
        ['V3', 'V4', 'V5', 'V6']  # 第三行：胸导联V3-V6
    ]

    for row in range(3):
        for col in range(4):
            lead_name = lead_order[row][col]
            ax = axes[row, col]  # 当前子图

            # 处理缺失导联数据的情况
            if lead_name not in lead_data:
                ax.text(0.5, 0.5, f"无{lead_name}数据", ha='center', va='center', fontsize=10)
                ax.set_title(lead_name, fontsize=12)
                continue

            # 获取原始和处理后数据
            raw_data = lead_data[lead_name]['raw']
            processed_data = lead_data[lead_name]['processed']

            # 绘制原始信号（蓝色）
            # ax.plot(raw_data.index, raw_data.values, 'b-', linewidth=0.7, alpha=0.8, label='原始信号')

            # 处理后信号缩放+偏移（避免与原始信号重叠）
            raw_range = raw_data.max() - raw_data.min()
            offset = raw_range * 0.15  # 偏移量为原始信号范围的15%
            # 将处理后信号（归一化到[0,1]）缩放至原始信号范围，并下移偏移量
            processed_scaled = processed_data * raw_range + (raw_data.min() - offset)
            ax.plot(processed_data.index, processed_scaled, 'r-', linewidth=0.7, alpha=0.8, label='OCCO处理后')

            # 美化子图
            ax.set_title(lead_name, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(axis='both', labelsize=8)  # 缩小刻度字体

    # 统一设置x轴标签（仅底部行）
    for col in range(4):
        axes[2, col].set_xlabel('采样点', fontsize=10)

    # 调整布局（预留标题空间，避免子图重叠）
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # rect参数：[左, 下, 右, 上]，预留顶部5%给标题
    plt.show()

    # 保存图像
    img_path = f"{output_dir}/十二导联对比_三行四列_beat{beat_id}.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"十二导联图像已保存到: {img_path}")


def main():
    print("OCCO特征提取测试程序（三行四列十二导联可视化）")
    print("=" * 50)

    test_file_path = "梁定旭_denoised.json"  # 替换为实际文件路径

    if os.path.exists(test_file_path):
        test_single_file(test_file_path)
    else:
        print(f"测试文件不存在: {test_file_path}")


if __name__ == "__main__":
    main()