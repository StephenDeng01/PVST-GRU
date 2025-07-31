"""
实现 开闭闭开（OCCO）运算的代码
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


class OCCO:

    """
    仅仅用作函数的容器，没有类的属性
    """

    @staticmethod
    def cor(heartbeat, g):
        """
        腐蚀操作
        :param heartbeat: 心拍信号
        :param g: 结构元素
        :return: 腐蚀操作的结果
        """
        g_len = len(g)
        result = []
        for i in range(len(heartbeat) - g_len):
            new_l = []
            for j in range(g_len):
                new_l.append(heartbeat[i + j] - g[j])
            result.append(min(new_l))
        # 创建插值函数
        # interp_func = interpolate.interp1d(np.arange(len(result), result, kind='linear', axis=0)
        # 在新的数据点位置上进行插值
        old_indices = np.arange(len(result))
        new_indices = np.linspace(0, len(result) - 1, 757)
        result = np.interp(new_indices, old_indices, result)
        # 使用 Min-Max 归一化
        min_val = np.min(result)
        max_val = np.max(result)
        result = (result - min_val) / (max_val - min_val)
        return result

    @staticmethod
    def exp(heartbeat, g):
        """
        膨胀操作
        :param heartbeat: 心拍信号
        :param g: 结构元素
        :return: 膨胀操作的结果
        """
        g_len = len(g)
        result = []
        for i in range(len(heartbeat) - g_len):
            new_l = []
            for j in range(g_len):
                new_l.append(heartbeat[i + j] - g[j])
            result.append(max(new_l))
        # 创建插值函数
        # interp_func = interpolate.interp1d(np.arange(len(result), result, kind='linear', axis=0)
        # 在新的数据点位置上进行插值
        old_indices = np.arange(len(result))
        new_indices = np.linspace(0, len(result) - 1, 757)
        result = np.interp(new_indices, old_indices, result)
        # 使用 Min-Max 归一化
        min_val = np.min(result)
        max_val = np.max(result)
        result = (result - min_val) / (max_val - min_val)
        # plt.plot(result)
        # plt.grid(True)
        # plt.show()
        return result

    @staticmethod
    def MC(heartbeat):
        """
        进行开闭闭开操作提取形态学特征
        :param heartbeat: 心拍信号
        :return: 特征（经过了插值）
        """
        scaler = MinMaxScaler()
        x = heartbeat.values
        # 使用 fit_transform 对数据进行归一化处理
        normalized_data = scaler.fit_transform(x.reshape(-1, 1))
        normalized_series = pd.Series(normalized_data.flatten())
        normalized_list = normalized_series.tolist()
        # 生成正半波正弦信号
        m = np.linspace(0, 1, 50)  # 从0到1秒，分成1000个点
        # 生成正半波正弦信号
        g = 1 * np.sin(np.pi * m)
        g = g.tolist()
        oc = OCCO.cor(normalized_list, g)
        oc = OCCO.exp(oc, g)  # 开运算
        oc = OCCO.exp(oc, g)
        OC = OCCO.cor(oc, g)
        co = OCCO.exp(normalized_list, g)
        co = OCCO.cor(co, g)
        co = OCCO.cor(co, g)
        CO = OCCO.exp(co, g)
        MC = normalized_list - 0.5 * (OC + CO)  # 修改过，之前写错OC-CO
        # 使用 Min-Max 归一化
        min_val = np.min(MC)
        max_val = np.max(MC)
        result = (MC - min_val) / (max_val - min_val)
        # 将归一化后的数据还原为 Series
        result_series = pd.Series(result)
        return result_series

