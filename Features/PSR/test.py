"""
PSR特征提取测试代码
使用PSR.py中的算法读取和处理单个ECG文件
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from PSR import PSR


def get_data(filename):
    """
    读取JSON文件并提取ECG数据
    
    Args:
        filename (str): JSON文件路径
    
    Returns:
        tuple: (processed_leads, raw_leads) 处理后的数据和原始数据
    """
    try:
        # 读取JSON文件
        with open(filename, 'r') as f:
            ecg_data = json.load(f)

        # 解析嵌套结构: beats -> 编号 -> 导联数据
        if "beats" not in ecg_data:
            print("错误: JSON文件中未找到外层的'beats'关键字")
            return None, None

        beat_content = ecg_data["beats"]

        # 初始化存储结果的字典
        raw_leads = {}
        processed_leads = {}

        # 十二导联的标准名称
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # 遍历beat下的所有编号（如001, 002等）
        for beat_id, beat_data in beat_content.items():
            # 处理当前心跳编号下的各导联数据
            for lead in leads:
                # 检查导联是否存在于当前beat数据中
                if lead in beat_data:
                    # 获取该导联的数据
                    lead_data = beat_data[lead]

                    # 处理数据格式
                    if isinstance(lead_data, str):
                        # 字符串格式: 浮点数间用空格分隔
                        data_values = list(map(float, lead_data.split()))
                    elif isinstance(lead_data, list):
                        # 列表格式: 直接转换为浮点数
                        data_values = [float(val) for val in lead_data]
                    else:
                        print(f"警告: 导联 {lead} 在beat {beat_id} 中的数据格式不支持")
                        continue

                    # 转换为pandas Series，使用"导联_心跳编号"作为键
                    key = f"{lead}_{beat_id}"
                    raw_series = pd.Series(data_values)
                    raw_leads[key] = raw_series

                    # 应用PSR处理
                    processed_series = PSR.ED(raw_series)
                    processed_leads[key] = processed_series
                else:
                    print(f"警告: 导联 {lead} 在beat {beat_id} 中未找到")

        if not raw_leads:
            print("警告: 未解析到任何有效数据")
            return None, None

        return processed_leads, raw_leads

    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到")
        return None, None
    except json.JSONDecodeError:
        print(f"错误: 文件 '{filename}' 不是有效的JSON格式")
        return None, None
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        return None, None


def test_single_file(file_path):
    """
    测试单个ECG文件的处理
    
    Args:
        file_path (str): JSON文件路径
    """
    print(f"正在处理文件: {file_path}")
    print("=" * 50)
    
    # 读取和处理数据
    processed_leads, raw_leads = get_data(file_path)
    
    if processed_leads is None or raw_leads is None:
        print("文件处理失败，请检查文件路径和格式")
        return
    
    print(f"成功处理 {len(processed_leads)} 个导联数据")
    print(f"导联列表: {list(processed_leads.keys())}")
    print()
    
    # 选择第一个导联进行详细分析
    first_lead_key = list(processed_leads.keys())[0]
    raw_data = raw_leads[first_lead_key]
    processed_data = processed_leads[first_lead_key]
    
    print(f"分析导联: {first_lead_key}")
    print(f"原始数据长度: {len(raw_data)}")
    print(f"处理后数据长度: {len(processed_data)}")
    print(f"原始数据范围: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"处理后数据范围: [{processed_data.min():.4f}, {processed_data.max():.4f}]")
    print()
    
    # 保存处理结果
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存处理后的数据为CSV
    for lead_name, data in processed_leads.items():
        csv_filename = f"{output_dir}/{lead_name}_psr.csv"
        data.to_csv(csv_filename, index=False)
        print(f"数据已保存到: {csv_filename}")
    
    # 可视化结果
    visualize_results(raw_data, processed_data, first_lead_key)
    
    return processed_leads, raw_leads


def visualize_results(raw_data, processed_data, lead_name):
    """
    可视化原始数据和处理后的数据
    
    Args:
        raw_data (pd.Series): 原始ECG数据
        processed_data (pd.Series): 处理后的PSR数据
        lead_name (str): 导联名称
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制原始ECG信号
    ax1.plot(raw_data.index, raw_data.values, 'b-', linewidth=0.8)
    ax1.set_title(f'原始ECG信号 - {lead_name}')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)
    
    # 绘制PSR处理后的信号
    ax2.plot(processed_data.index, processed_data.values, 'r-', linewidth=0.8)
    ax2.set_title(f'PSR处理后信号 - {lead_name}')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('欧氏距离')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存图像
    output_dir = "test_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/{lead_name}_comparison.png", dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_dir}/{lead_name}_comparison.png")





def main():
    """
    主测试函数
    """
    print("PSR特征提取测试程序")
    print("=" * 50)
    
    # 测试文件处理
    # 请将下面的文件路径替换为实际的JSON文件路径
    test_file_path = "..\\..\\梁定旭_denoised.json"  # 请修改为实际文件路径
    
    if os.path.exists(test_file_path):
        processed_leads, raw_leads = test_single_file(test_file_path)
        
        # 处理结果已在test_single_file函数中保存和可视化
        pass
    else:
        print(f"测试文件不存在: {test_file_path}")
        print("请修改test_file_path变量为实际的JSON文件路径")


if __name__ == "__main__":
    main()