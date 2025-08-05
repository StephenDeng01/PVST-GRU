"""
香农包络线的提取
"""

import numpy as np
import pandas as pd


class PSR:
    """
    实现 PSR + DEM(坐标延迟重构 Delay Embedding Method) 的结合提取特征
    """

    @staticmethod
    def time_delay_embedding(signal, lag, dimension):
        """
        实现 相空间重构 的功能
        :param signal: 信号
        :param lag: 延迟参数 t
        :param dimension: 嵌入维度
        :return: 重构结果
        """
        N = len(signal)
        embedded_series = np.zeros((N - (dimension - 1) * lag, dimension))
        for i in range(dimension):
            embedded_series[:, i] = signal[i * lag:i * lag + len(embedded_series)]
        return embedded_series

    @staticmethod
    def euclidean_distance_sequence(embedded_series):
        """
        计算相邻时间点在相空间中的欧式距离
        :param embedded_series: 时间延迟嵌入之后的序列
        :return: 距离序列，反映了信号在空间中的演化速度
        """

        N = len(embedded_series)
        distance_sequence = np.zeros(N - 1)
        for i in range(N - 1):
            # distance_sequence[i] = euclidean(embedded_series[i], embedded_series[i + 1])
            distance_sequence[i] = np.linalg.norm(embedded_series[i] - embedded_series[i + 1])
        return distance_sequence

    @staticmethod
    def ED(x):
        x = x.values
        # 设置延迟和维度
        lag = 22  # 绘制ECG_v5时间序列的自相关函数图估计出的合适值
        dimension = 20  # 根据最小平均互信息法找到合适的嵌入维度
        # 相空间重构
        embedded_series = PSR.time_delay_embedding(x, lag, dimension)
        # 计算欧氏距离序列
        distance_sequence = PSR.euclidean_distance_sequence(embedded_series)
        '''插值'''
        # 在新的数据点位置上进行插值
        old_indices = np.arange(len(distance_sequence))
        new_indices = np.linspace(0, len(distance_sequence) - 1, 757)
        result = np.interp(new_indices, old_indices, distance_sequence)
        return pd.Series(result)

