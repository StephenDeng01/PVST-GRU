"""
NEO Smoothed-NEO (平滑)非线性能量算子
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from vmdpy import VMD


def NEO(x):
    # 计算这个元素前一个元素和后一个元素的乘积
    neo1 = x.shift(1) * x.shift(-1)
    # 第一个和最后一个元素直接用本身，因为缺乏乘子
    neo1.iloc[0] = x.iloc[0]
    neo1.iloc[-1] = x.iloc[-1]
    neo = x * x - neo1
    # 使用 Min-Max 归一化
    min_val = np.min(neo)
    max_val = np.max(neo)
    neo = (neo - min_val) / (max_val - min_val)
    return neo


def triangular_window(N):
    window = np.zeros(N)
    for n in range(N):
        window[n] = 2 / (N - 1) * ((N - 1) / 2 - np.abs(n - (N - 1) / 2))
    return pd.Series(window)


def SNEO(x):
    neo = NEO(x)
    trw = triangular_window(100)
    convolved_result = np.convolve(trw, neo, mode='valid')  # 将 trw 视为卷积核，neo 为输入信号
    # 在新的数据点位置上进行插值
    old_indices = np.arange(len(convolved_result))
    new_indices = np.linspace(0, len(convolved_result) - 1, 757)
    convolved_result = np.interp(new_indices, old_indices, convolved_result)
    # 使用 Min-Max 归一化
    min_val = np.min(convolved_result)
    max_val = np.max(convolved_result)
    convolved_result = (convolved_result - min_val) / (max_val - min_val)
    return pd.Series(convolved_result)


def get_data(filename):
    try:
        # 读取JSON文件
        with open(filename, 'r') as f:
            ecg_data = json.load(f)

        # 解析嵌套结构: data -> beat -> 编号 -> 导联数据
        # 检查各级关键字是否存在
        if "data" not in ecg_data:
            print("错误: JSON文件中未找到外层的'data'关键字")
            return None, None

        data_content = ecg_data["data"]

        if "beat" not in data_content:
            print("错误: data中未找到'beat'关键字")
            return None, None

        beat_content = data_content["beat"]

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
                    # 假设数据格式可能是字符串(空格分隔)或直接的数值列表
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

                    # 应用SNEO处理
                    processed_series = SNEO(raw_series)
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


def main():
    return None  # 占位
