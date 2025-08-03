"""
将 json 的数据写入 csv 中
"""

import json
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import shutil


def get_label_from_path(file_path: str) -> int:
    """根据文件路径中的positive/negative确定标签"""
    path_lower = file_path.lower()
    if "positive" in path_lower:
        return 1
    elif "negative" in path_lower:
        return 0
    else:
        print(f"警告: 文件路径 {file_path} 中未找到positive或negative，默认标签为0")
        return 0


def find_all_json_files(root_dir: str) -> List[str]:
    """查找根目录下所有JSON文件，包括子文件夹"""
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))
    return json_files


def process_single_json(json_path: str, output_root: str) -> Tuple[bool, str]:
    """
    处理单个JSON文件并保存为CSV

    Args:
        json_path: JSON文件路径
        output_root: 输出根目录

    Returns:
        处理结果和信息
    """
    try:
        # 1. 确定标签
        label = get_label_from_path(json_path)

        # 2. 读取并解析JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 3. 提取心拍数据
        if "beats" not in data:
            return False, f"未找到'beats'字段: {json_path}"

        beats = data["beats"]
        if not beats:
            return False, f"无有效心拍数据: {json_path}"

        # 4. 准备数据
        samples = []
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # 确定最大数据点数量
        max_points = 0
        for beat_data in beats.values():
            for lead in leads:
                if lead in beat_data:
                    if isinstance(beat_data[lead], str):
                        points = len(beat_data[lead].split())
                    else:
                        points = len(beat_data[lead])
                    max_points = max(max_points, points)

        if max_points == 0:
            return False, f"未找到有效数据点: {json_path}"

        # 生成样本
        for beat_id, beat_data in beats.items():
            sample = {"beat_id": beat_id}

            for lead in leads:
                if lead in beat_data:
                    # 解析导联数据
                    if isinstance(beat_data[lead], str):
                        lead_values = list(map(float, beat_data[lead].split()))
                    else:
                        lead_values = list(map(float, beat_data[lead]))

                    # 填充数据点
                    for i in range(max_points):
                        sample[f"{lead}_p{i + 1}"] = lead_values[i] if i < len(lead_values) else np.nan
                else:
                    # 导联缺失，填充NaN
                    for i in range(max_points):
                        sample[f"{lead}_p{i + 1}"] = np.nan

            # 添加标签
            sample["label"] = label
            samples.append(sample)

        # 5. 确定输出路径并创建文件夹
        relative_path = os.path.relpath(json_path, os.path.dirname(output_root))
        output_dir = os.path.dirname(os.path.join(output_root, relative_path))
        os.makedirs(output_dir, exist_ok=True)

        # 6. 保存为CSV
        csv_filename = os.path.splitext(os.path.basename(json_path))[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)

        pd.DataFrame(samples).fillna(0).to_csv(csv_path, index=False)
        return True, f"成功处理: {csv_path} (标签: {label}, 样本数: {len(samples)})"

    except Exception as e:
        return False, f"处理失败 {json_path}: {str(e)}"


def batch_process_ecg_data(input_root: str, output_root: str) -> None:
    """
    批量处理ECG数据，保持文件夹结构

    Args:
        input_root: 输入根目录
        output_root: 输出根目录
    """
    # 检查输入目录
    if not os.path.isdir(input_root):
        print(f"错误: 输入目录 {input_root} 不存在")
        return

    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    # 查找所有JSON文件
    json_files = find_all_json_files(input_root)
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    # 处理每个文件
    success_count = 0
    for i, json_file in enumerate(json_files, 1):
        success, message = process_single_json(json_file, output_root)
        print(f"[{i}/{len(json_files)}] {message}")
        if success:
            success_count += 1

    print(f"\n处理完成: 成功 {success_count}/{len(json_files)}")


if __name__ == "__main__":
    # 配置输入输出目录
    INPUT_DIRECTORY = "F:\extracted_beats_unified"  # 替换为你的输入根目录
    OUTPUT_DIRECTORY = "F:\heartbeat_csv"  # 替换为你的输出根目录

    # 执行批量处理
    batch_process_ecg_data(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
