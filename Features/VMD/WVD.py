from vmdpy import VMD
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA


class WVD:
    """
    这里实现 VMD + Shannon 特征提取
    """

    @staticmethod
    def funVMD(x):
        # global id
        # id += 1
        # print(i+1, id, sum_id)
        x = x.values
        '''初始化VMD对象'''
        # alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
        # 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
        # a值越大，各IMF分量的带宽越小，过小的带宽是使得被分解的信号中某些信号丢失该系数常见取值范围为1000~3000
        alpha = 1000
        tau = 0.02  # tau 噪声容限，即允许重构后的信号与原始信号有差别。
        K = 4  # K 分解模态（IMF）个数
        DC = 0  # 是否包括直流分量
        init = 1  # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
        tol = 1e-7  # 控制误差大小常量，决定精度与迭代次数
        imfs_v, u_hat, omega = VMD(x, alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
        imfs_v4 = []
        for imf in imfs_v:
            imfs_v4.extend(imf)
        return pd.Series(imfs_v4)

    @staticmethod
    def SEE(x):
        x = x.values
        x1 = x[:500]
        x2 = x[500:1000]
        x3 = x[1000:1500]
        x4 = x[1500:2000]
        # 使用 Min-Max 归一化
        epsilon = 1e-10  # 设置一个小的非零最小值
        min_val = np.min(x1)
        max_val = np.max(x1)
        x1 = (x1 - min_val + epsilon) / (max_val - min_val)
        min_val = np.min(x2)
        max_val = np.max(x2)
        x2 = (x2 - min_val + epsilon) / (max_val - min_val)
        min_val = np.min(x3)
        max_val = np.max(x3)
        x3 = (x3 - min_val + epsilon) / (max_val - min_val)
        min_val = np.min(x4)
        max_val = np.max(x4)
        x4 = (x4 - min_val + epsilon) / (max_val - min_val)
        # 香农能量SE
        SE1 = -(x1 ** 2) * (np.log10(x1 ** 2))
        SE2 = -(x2 ** 2) * (np.log10(x2 ** 2))
        SE3 = -(x3 ** 2) * (np.log10(x3 ** 2))
        SE4 = -(x4 ** 2) * (np.log10(x4 ** 2))
        # 香农能量包络SEE
        Ea1 = np.mean(SE1)  # 均值
        Ea2 = np.mean(SE2)
        Ea3 = np.mean(SE3)
        Ea4 = np.mean(SE4)
        a_std1 = np.std(SE1)  # 标准差
        a_std2 = np.std(SE2)
        a_std3 = np.std(SE3)
        a_std4 = np.std(SE4)
        SEE1 = (SE1 - Ea1) / a_std1
        SEE2 = (SE2 - Ea2) / a_std2
        SEE3 = (SE3 - Ea3) / a_std3
        SEE4 = (SE4 - Ea4) / a_std4
        SEEs4 = []
        SEEs4.extend(SEE1)
        SEEs4.extend(SEE2)
        SEEs4.extend(SEE3)
        SEEs4.extend(SEE4)
        return pd.Series(SEEs4)

    @staticmethod
    def funPCA(x):
        x = pd.DataFrame(x.values.reshape(4, -1))
        # 数据标准化
        scaler = StandardScaler()
        x = scaler.fit_transform(x.T)
        # 创建PCA对象，设置降维后的维度为1
        pca = PCA(n_components=1)
        # 使用fit_transform方法对数据进行降维
        reduced_data = pca.fit_transform(x)
        # print(pd.Series(reduced_data.ravel()))
        return pd.Series(reduced_data.ravel())

    @staticmethod
    def kpca(x):
        x = pd.DataFrame(x.values.reshape(4, -1))
        # 假设 X 是特征
        X = x.values
        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X.T)
        # print(X.shape)
        # 创建KPCA模型并拟合数据
        kernel = 'rbf'  # 高斯核函数，也称为RBF核,非线性核函数，能够处理复杂的非线性关系。
        kpca = KernelPCA(n_components=1, kernel=kernel)
        # kpca = PCA(n_components=1)
        X_kpca = kpca.fit_transform(X)
        # print(X_kpca.shape)  # X_kpca包含降维后的数据，每行对应一个样本，每列对应一个主成分
        result = X_kpca.reshape((-1))
        # x = X.T
        return pd.Series(result)
