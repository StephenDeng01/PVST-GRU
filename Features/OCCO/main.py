"""
OCCO特征提取批量处理程序
实现开闭闭开（OCCO）运算的批量处理
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import shutil
from OCCO import OCCO


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

                    # 应用OCCO处理
                    processed_series = OCCO.MC(raw_series)
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


def visualize_results_from_csv(csv_file_path, raw_data, lead_name, file_name, output_dir):
    """
    从CSV文件读取数据并可视化
    
    Args:
        csv_file_path (str): CSV文件路径
        raw_data (pd.Series): 原始ECG数据（用于对比）
        lead_name (str): 导联名称（包含心拍编号，如 'I_001'）
        file_name (str): 文件名
        output_dir (str): 输出目录
    """
    try:
        # 从CSV文件读取处理后的数据
        csv_data = pd.read_csv(csv_file_path)
        
        # 解析导联名称和心拍编号
        parts = lead_name.split('_')
        if len(parts) >= 2:
            base_lead = parts[0]  # 导联名称
            beat_id = parts[1]    # 心拍编号
        else:
            base_lead = lead_name
            beat_id = '001'
        
        # 构建列名
        column_name = f'{base_lead}_beat_{beat_id}'
        
        # 检查是否存在对应的列
        if column_name in csv_data.columns:
            processed_data = csv_data[column_name]
        else:
            # 如果找不到对应的列，使用第一列
            processed_data = csv_data.iloc[:, 0]
            print(f"警告: 未找到列 '{column_name}'，使用第一列数据")
        
        print(f"从CSV文件读取数据: {csv_file_path}")
        print(f"选择的导联: {base_lead}, 心拍: {beat_id}")
        print(f"查找的列名: {column_name}")
        print(f"CSV数据长度: {len(processed_data)}")
        print(f"CSV数据范围: [{processed_data.min():.4f}, {processed_data.max():.4f}]")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制原始ECG信号
        ax1.plot(raw_data.index, raw_data.values, 'b-', linewidth=0.8)
        ax1.set_title(f'原始ECG信号 - {file_name}_{lead_name}')
        ax1.set_ylabel('幅值')
        ax1.grid(True, alpha=0.3)
        
        # 绘制从CSV读取的OCCO处理后的信号
        ax2.plot(processed_data.index, processed_data.values, 'r-', linewidth=0.8)
        ax2.set_title(f'OCCO处理后信号 ({base_lead}_beat_{beat_id}) - {file_name}_{lead_name}')
        ax2.set_xlabel('采样点')
        ax2.set_ylabel('归一化幅值')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/{file_name}_{lead_name}_comparison.png", dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_dir}/{file_name}_{lead_name}_comparison.png")
        plt.close()  # 关闭图像以释放内存
        
    except Exception as e:
        print(f"从CSV文件读取数据时发生错误: {str(e)}")
        print(f"CSV文件路径: {csv_file_path}")


def process_folder_with_structure(input_folder, output_base_dir="output"):
    """
    处理总文件夹，按照元文件夹的子文件夹格式存放输出文件
    
    Args:
        input_folder (str): 输入总文件夹路径
        output_base_dir (str): 输出基础目录
    """
    print("OCCO特征提取文件夹处理程序")
    print("=" * 50)
    print(f"输入文件夹: {input_folder}")
    print(f"输出基础目录: {output_base_dir}")
    print()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 创建输出基础目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 统计信息
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    # 遍历输入文件夹中的所有子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 跳过根目录
        if root == input_folder:
            continue
            
        # 计算相对路径
        rel_path = os.path.relpath(root, input_folder)
        
        # 创建对应的输出目录
        output_dir = os.path.join(output_base_dir, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"处理子文件夹: {rel_path}")
        print(f"输出目录: {output_dir}")
        
        # 处理当前子文件夹中的所有JSON文件
        json_files = [f for f in files if f.endswith('.json')]
        
        if not json_files:
            print(f"  子文件夹 {rel_path} 中没有找到JSON文件")
            continue
        
        print(f"  找到 {len(json_files)} 个JSON文件")
        
        # 处理每个JSON文件
        for json_file in json_files:
            total_files += 1
            file_path = os.path.join(root, json_file)
            
            print(f"  处理文件: {json_file}")
            
            try:
                # 处理单个文件
                processed_leads, raw_leads = get_data(file_path)
                
                if processed_leads is None or raw_leads is None:
                    print(f"    处理失败: {json_file}")
                    failed_files += 1
                    continue
                
                # 获取文件名（不含扩展名）
                file_name = os.path.splitext(json_file)[0]
                
                # 将所有心拍和导联数据保存到一个CSV文件中，按层级组织
                all_data_list = []
                
                # 按心拍编号分组，然后按导联分组
                beat_data_dict = {}
                
                for lead_name, data in processed_leads.items():
                    # 解析心拍编号和导联名称
                    parts = lead_name.split('_')
                    if len(parts) >= 2:
                        base_lead = parts[0]  # 导联名称
                        beat_id = parts[1]    # 心拍编号
                    else:
                        base_lead = lead_name
                        beat_id = '001'
                    
                    # 初始化心拍字典
                    if beat_id not in beat_data_dict:
                        beat_data_dict[beat_id] = {}
                    
                    # 添加导联数据
                    data_with_name = data.copy()
                    data_with_name.name = f'{base_lead}_beat_{beat_id}'
                    beat_data_dict[beat_id][base_lead] = data_with_name
                
                # 将所有数据合并为一个DataFrame
                all_columns = []
                for beat_id in sorted(beat_data_dict.keys()):
                    for lead_name in sorted(beat_data_dict[beat_id].keys()):
                        all_columns.append(beat_data_dict[beat_id][lead_name])
                
                if all_columns:
                    combined_data = pd.concat(all_columns, axis=1)
                    csv_filename = f"{output_dir}/{file_name}_all_occo.csv"
                    combined_data.to_csv(csv_filename, index=False)
                    
                    # 统计信息
                    num_beats = len(beat_data_dict)
                    num_leads = len(set([col.split('_')[0] for col in combined_data.columns]))
                    print(f"    保存所有数据: {num_beats} 个心拍 × {num_leads} 个导联 = {len(combined_data.columns)} 列")
                    
                    # 显示列名结构
                    print(f"    列名结构: {list(combined_data.columns[:5])}...")
                
                # 随机选择一个心拍进行可视化
                if processed_leads:
                    import random
                    random_lead_key = random.choice(list(processed_leads.keys()))
                    raw_data = raw_leads[random_lead_key]
                    
                    # 从CSV文件读取数据并可视化
                    csv_filename = f"{output_dir}/{file_name}_all_occo.csv"
                    visualize_results_from_csv(csv_filename, raw_data, random_lead_key, file_name, output_dir)
                
                processed_files += 1
                print(f"    处理成功: {json_file}")
                
            except Exception as e:
                print(f"    处理文件时发生错误: {str(e)}")
                failed_files += 1
        
        print(f"  子文件夹 {rel_path} 处理完成")
        print()
    
    # 输出处理结果统计
    print("文件夹处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files} 个文件")
    print(f"处理失败: {failed_files} 个文件")
    print(f"输出基础目录: {output_base_dir}")


def main():
    """
    主函数 - 批量处理模式
    """
    print("OCCO特征提取批量处理程序")
    print("=" * 50)
    
    # 硬编码配置 - 批量处理
    folder_input_path = "F:\\extracted_beats_unified"  # 输入文件夹路径
    folder_output_dir = "folder_output"  # 文件夹处理输出目录
    
    print(f"输入文件夹: {folder_input_path}")
    print(f"输出目录: {folder_output_dir}")
    print()
    
    # 执行批量处理
    process_folder_with_structure(folder_input_path, folder_output_dir)


if __name__ == "__main__":
    main()

