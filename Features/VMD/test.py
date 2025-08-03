import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class WVD:
    """VMD + Shannon特征提取类（全程使用线性插值保持数据长度）"""

    @staticmethod
    def funVMD(x):
        # 确保输入是float类型的numpy数组，且通过插值调整为757点
        x = np.asarray(x, dtype=np.float64)
        if len(x) != 757:
            # 使用线性插值调整长度为757
            x = np.interp(np.linspace(0, len(x) - 1, 757), np.arange(len(x)), x)

        alpha = 1000
        tau = 0.02
        K = 4  # 4个成分
        DC = 0
        init = 1
        tol = 1e-7
        imfs_v, _, _ = VMD(x, alpha, tau, K, DC, init, tol)

        # 确保每个VMD成分都通过插值调整为757点
        imfs_flat = []
        for imf in imfs_v:
            if len(imf) != 757:
                imf = np.interp(np.linspace(0, len(imf) - 1, 757), np.arange(len(imf)), imf)
            imfs_flat.extend(imf)
        return pd.Series(imfs_flat)

    @staticmethod
    def SEE(x):
        x = x.values
        target_len = 4 * 757  # 3028点
        # 使用线性插值调整为目标长度
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x1 = x[:757]
        x2 = x[757:1514]
        x3 = x[1514:2271]
        x4 = x[2271:3028]

        epsilon = 1e-10
        # 归一化
        x1 = (x1 - np.min(x1) + epsilon) / (np.max(x1) - np.min(x1) + epsilon * 2)
        x2 = (x2 - np.min(x2) + epsilon) / (np.max(x2) - np.min(x2) + epsilon * 2)
        x3 = (x3 - np.min(x3) + epsilon) / (np.max(x3) - np.min(x3) + epsilon * 2)
        x4 = (x4 - np.min(x4) + epsilon) / (np.max(x4) - np.min(x4) + epsilon * 2)

        # 计算香农能量
        SE1 = -(x1 ** 2) * np.log10(x1 ** 2)
        SE2 = -(x2 ** 2) * np.log10(x2 ** 2)
        SE3 = -(x3 ** 2) * np.log10(x3 ** 2)
        SE4 = -(x4 ** 2) * np.log10(x4 ** 2)

        # 处理可能的NaN值
        SE1 = np.nan_to_num(SE1)
        SE2 = np.nan_to_num(SE2)
        SE3 = np.nan_to_num(SE3)
        SE4 = np.nan_to_num(SE4)

        # 标准化SEE
        SEE1 = (SE1 - np.mean(SE1)) / (np.std(SE1) + epsilon)
        SEE2 = (SE2 - np.mean(SE2)) / (np.std(SE2) + epsilon)
        SEE3 = (SE3 - np.mean(SE3)) / (np.std(SE3) + epsilon)
        SEE4 = (SE4 - np.mean(SE4)) / (np.std(SE4) + epsilon)

        return pd.Series(np.concatenate([SEE1, SEE2, SEE3, SEE4]))

    @staticmethod
    def funPCA(x):
        x = x.values
        target_len = 4 * 757  # 3028点
        # 插值调整长度
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x_reshaped = x.reshape(4, 757)
        x_scaled = StandardScaler().fit_transform(x_reshaped.T)
        pca = PCA(n_components=1)
        return pd.Series(pca.fit_transform(x_scaled).ravel())

    @staticmethod
    def kpca(x):
        x = x.values
        target_len = 4 * 757  # 3028点
        # 插值调整长度
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x_reshaped = x.reshape(4, 757)
        x_scaled = StandardScaler().fit_transform(x_reshaped.T)
        kpca = KernelPCA(n_components=1, kernel='rbf')
        return pd.Series(kpca.fit_transform(x_scaled).ravel())


def visualize_reduction_results(csv_path, sample_idx=0):
    """可视化降维后的最终结果，全程使用插值保持数据长度"""
    # 读取数据并处理类型
    try:
        df = pd.read_csv(csv_path, header=None, dtype=np.float64)
    except ValueError:
        df = pd.read_csv(csv_path, header=None)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.mean())

    beat_id = df.iloc[sample_idx, 0]
    signal_data = df.iloc[sample_idx, 1:]

    # 准备画布
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'心拍编号 {beat_id:.0f} 的降维结果可视化', fontsize=16)

    # 遍历12个导联
    for lead_idx in range(12):
        row, col = lead_idx // 4, lead_idx % 4
        ax = axes[row, col]

        # 提取导联信号并通过插值确保757点
        start_col = lead_idx * 757
        end_col = start_col + 757
        signal = signal_data.iloc[start_col:end_col].values.astype(np.float64)

        # 关键：使用线性插值调整为757点
        if len(signal) != 757:
            signal = np.interp(np.linspace(0, len(signal) - 1, 757), np.arange(len(signal)), signal)

        signal = pd.Series(signal)

        # 处理特征
        vmd_components = WVD.funVMD(signal)
        see_features = WVD.SEE(vmd_components)
        pca_result = WVD.funPCA(see_features)
        kpca_result = WVD.kpca(see_features)

        # 绘制
        ax.plot(pca_result, label='PCA降维结果', color='blue', alpha=0.7)
        ax.plot(kpca_result, label='KPCA降维结果', color='red', alpha=0.7)
        ax.set_title(f'导联 {lead_idx + 1}', fontsize=12)
        ax.set_xlabel('样本点')
        ax.set_ylabel('特征值')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    ecg_data_path = "卢显超_denoised.csv"  # 替换为你的CSV路径
    visualize_reduction_results(ecg_data_path, sample_idx=0)


