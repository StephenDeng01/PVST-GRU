from vmdpy import VMD
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA


class WVD:
    """
    这里实现 VMD + Shannon 特征提取，每段长度调整为757
    """

    @staticmethod
    def funVMD(x):
        x = x.values
        '''初始化VMD对象'''
        alpha = 1000  # 惩罚系数
        tau = 0.02  # 噪声容限
        K = 4  # 分解模态（IMF）个数
        DC = 0  # 是否是否包括直流分量
        init = 1  # 中心频率初始化方式
        tol = 1e-7  # 控制误差大小常量
        # 进行VMD分解
        imfs_v, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)
        imfs_v4 = []
        for imf in imfs_v:
            imfs_v4.extend(imf)
        return pd.Series(imfs_v4)

    @staticmethod
    def SEE(x):
        x = x.values
        # 按757长度分段（总长度应为757*4=3028）
        x1 = x[:757]
        x2 = x[757:1514]  # 757*2=1514
        x3 = x[1514:2271]  # 757*3=2271
        x4 = x[2271:3028]  # 757*4=3028

        # 使用Min-Max归一化
        epsilon = 1e-10  # 避免除零和对数计算错误
        # 处理第一段
        min_val = np.min(x1)
        max_val = np.max(x1)
        x1 = (x1 - min_val + epsilon) / (max_val - min_val)
        # 处理第二段
        min_val = np.min(x2)
        max_val = np.max(x2)
        x2 = (x2 - min_val + epsilon) / (max_val - min_val)
        # 处理第三段
        min_val = np.min(x3)
        max_val = np.max(x3)
        x3 = (x3 - min_val + epsilon) / (max_val - min_val)
        # 处理第四段
        min_val = np.min(x4)
        max_val = np.max(x4)
        x4 = (x4 - min_val + epsilon) / (max_val - min_val)

        # 计算香农能量SE
        SE1 = -(x1 ** 2) * (np.log10(x1 ** 2))
        SE2 = -(x2 ** 2) * (np.log10(x2 ** 2))
        SE3 = -(x3 ** 2) * (np.log10(x3 ** 2))
        SE4 = -(x4 ** 2) * (np.log10(x4 ** 2))

        # 计算香农能量包络SEE
        Ea1, a_std1 = np.mean(SE1), np.std(SE1)
        Ea2, a_std2 = np.mean(SE2), np.std(SE2)
        Ea3, a_std3 = np.mean(SE3), np.std(SE3)
        Ea4, a_std4 = np.mean(SE4), np.std(SE4)

        SEE1 = (SE1 - Ea1) / a_std1
        SEE2 = (SE2 - Ea2) / a_std2
        SEE3 = (SE3 - Ea3) / a_std3
        SEE4 = (SE4 - Ea4) / a_std4

        # 拼接所有段的SEE结果
        SEEs4 = []
        SEEs4.extend(SEE1)
        SEEs4.extend(SEE2)
        SEEs4.extend(SEE3)
        SEEs4.extend(SEE4)
        return pd.Series(SEEs4)

    @staticmethod
    def funPCA(x):
        # 重塑为4行（对应4个分段），每行长度为757
        x = pd.DataFrame(x.values.reshape(4, -1))
        # 数据标准化
        scaler = StandardScaler()
        x = scaler.fit_transform(x.T)
        # PCA降维到1个主成分
        pca = PCA(n_components=1)
        reduced_data = pca.fit_transform(x)
        return pd.Series(reduced_data.ravel())

    @staticmethod
    def kpca(x):
        # 重塑为4行（对应4个分段），每行长度为757
        x = pd.DataFrame(x.values.reshape(4, -1))
        X = x.values
        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X.T)
        # 使用RBF核的KPCA降维到1个主成分
        kpca = KernelPCA(n_components=1, kernel='rbf')
        X_kpca = kpca.fit_transform(X)
        result = X_kpca.reshape((-1))
        return pd.Series(result)
