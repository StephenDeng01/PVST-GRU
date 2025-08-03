import os
import pandas as pd
import numpy as np
from vmdpy import VMD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class WVD:
    """VMD + Shannon特征提取类（支持子文件夹处理）"""

    @staticmethod
    def funVMD(x):
        x = np.asarray(x, dtype=np.float64)
        if len(x) != 757:
            x = np.interp(np.linspace(0, len(x) - 1, 757), np.arange(len(x)), x)

        alpha = 1000
        tau = 0.02
        K = 4
        DC = 0
        init = 1
        tol = 1e-7
        imfs_v, _, _ = VMD(x, alpha, tau, K, DC, init, tol)

        imfs_flat = []
        for imf in imfs_v:
            if len(imf) != 757:
                imf = np.interp(np.linspace(0, len(imf) - 1, 757), np.arange(len(imf)), imf)
            imfs_flat.extend(imf)
        return pd.Series(imfs_flat)

    @staticmethod
    def SEE(x):
        x = x.values
        target_len = 4 * 757
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x1 = x[:757]
        x2 = x[757:1514]
        x3 = x[1514:2271]
        x4 = x[2271:3028]

        epsilon = 1e-10
        x1 = (x1 - np.min(x1) + epsilon) / (np.max(x1) - np.min(x1) + epsilon * 2)
        x2 = (x2 - np.min(x2) + epsilon) / (np.max(x2) - np.min(x2) + epsilon * 2)
        x3 = (x3 - np.min(x3) + epsilon) / (np.max(x3) - np.min(x3) + epsilon * 2)
        x4 = (x4 - np.min(x4) + epsilon) / (np.max(x4) - np.min(x4) + epsilon * 2)

        SE1 = -(x1 ** 2) * np.log10(x1 ** 2)
        SE2 = -(x2 ** 2) * np.log10(x2 ** 2)
        SE3 = -(x3 ** 2) * np.log10(x3 ** 2)
        SE4 = -(x4 ** 2) * np.log10(x4 ** 2)

        SE1 = np.nan_to_num(SE1)
        SE2 = np.nan_to_num(SE2)
        SE3 = np.nan_to_num(SE3)
        SE4 = np.nan_to_num(SE4)

        SEE1 = (SE1 - np.mean(SE1)) / (np.std(SE1) + epsilon)
        SEE2 = (SE2 - np.mean(SE2)) / (np.std(SE2) + epsilon)
        SEE3 = (SE3 - np.mean(SE3)) / (np.std(SE3) + epsilon)
        SEE4 = (SE4 - np.mean(SE4)) / (np.std(SE4) + epsilon)

        return pd.Series(np.concatenate([SEE1, SEE2, SEE3, SEE4]))

    @staticmethod
    def funPCA(x):
        x = x.values
        target_len = 4 * 757
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x_reshaped = x.reshape(4, 757)
        x_scaled = StandardScaler().fit_transform(x_reshaped.T)
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(x_scaled).ravel()

        # PCA结果归一化
        epsilon = 1e-10
        pca_norm = (pca_result - np.min(pca_result)) / (np.max(pca_result) - np.min(pca_result) + epsilon)
        return pd.Series(pca_norm)


def process_single_file(input_path, output_path):
    """处理单个CSV文件，保持格式一致"""
    try:
        df = pd.read_csv(input_path, header=None, dtype=np.float64)
    except ValueError:
        df = pd.read_csv(input_path, header=None)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.mean())

    processed_samples = []
    for idx, row in df.iterrows():
        beat_id = row[0]
        signal_data = row[1:]
        lead_features = []

        for lead_idx in range(12):
            start_col = lead_idx * 757
            end_col = start_col + 757
            signal = signal_data[start_col:end_col].values.astype(np.float64)

            if len(signal) != 757:
                signal = np.interp(np.linspace(0, len(signal) - 1, 757), np.arange(len(signal)), signal)

            vmd_components = WVD.funVMD(pd.Series(signal))
            see_features = WVD.SEE(vmd_components)
            pca_norm = WVD.funPCA(see_features)
            lead_features.extend(pca_norm.values)

        sample_result = [beat_id] + lead_features
        processed_samples.append(sample_result)

    result_df = pd.DataFrame(processed_samples)
    result_df.to_csv(output_path, header=False, index=False)
    print(f"已处理：{input_path} -> {output_path}")


def batch_process_recursive(input_root, output_root):
    """递归处理所有子文件夹中的CSV文件，保持文件夹结构"""
    # 遍历输入根目录下的所有文件和子文件夹
    for root, dirs, files in os.walk(input_root):
        # 计算当前目录相对于输入根目录的相对路径（用于复刻结构）
        relative_path = os.path.relpath(root, input_root)
        # 构建输出目录路径
        output_dir = os.path.join(output_root, relative_path)
        # 创建输出目录（包括所有必要的父目录）
        os.makedirs(output_dir, exist_ok=True)

        # 处理当前目录下的所有CSV文件
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file)
                # 处理单个文件
                process_single_file(input_file_path, output_file_path)

    print(f"\n所有文件处理完成！结果保存在：{output_root}")


if __name__ == "__main__":
    # 输入根文件夹（包含子文件夹和CSV文件）
    input_root_folder = "F:\heartbeat_csv\extracted_beats_unified"
    # 输出根文件夹（将复刻输入的子文件夹结构）
    output_root_folder = "F:\heartbeat_csv\SEE"

    # 开始递归批处理
    batch_process_recursive(input_root_folder, output_root_folder)