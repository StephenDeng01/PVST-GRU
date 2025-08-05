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
        target_len = 4 * 757  # 4个IMF分量，每个757点
        if len(x) != target_len:
            x = np.interp(np.linspace(0, len(x) - 1, target_len), np.arange(len(x)), x)

        x1 = x[:757]
        x2 = x[757:1514]
        x3 = x[1514:2271]
        x4 = x[2271:3028]

        epsilon = 1e-10
        # 归一化处理
        x1 = (x1 - np.min(x1) + epsilon) / (np.max(x1) - np.min(x1) + epsilon * 2)
        x2 = (x2 - np.min(x2) + epsilon) / (np.max(x2) - np.min(x2) + epsilon * 2)
        x3 = (x3 - np.min(x3) + epsilon) / (np.max(x3) - np.min(x3) + epsilon * 2)
        x4 = (x4 - np.min(x4) + epsilon) / (np.max(x4) - np.min(x4) + epsilon * 2)

        # 香农能量计算
        SE1 = -(x1 **2) * np.log10(x1** 2 + epsilon)
        SE2 = -(x2 **2) * np.log10(x2** 2 + epsilon)
        SE3 = -(x3 **2) * np.log10(x3** 2 + epsilon)
        SE4 = -(x4 **2) * np.log10(x4** 2 + epsilon)

        # 处理异常值
        SE1 = np.nan_to_num(SE1)
        SE2 = np.nan_to_num(SE2)
        SE3 = np.nan_to_num(SE3)
        SE4 = np.nan_to_num(SE4)

        # 标准化得到SEE
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
    """处理单个CSV文件，保持与JSON转CSV相同的输出格式（含表头、beat_id、label）"""
    try:
        # 读取带表头的CSV（与JSON转CSV的输出格式匹配）
        df = pd.read_csv(input_path)
        # 确认必要字段存在
        if 'beat_id' not in df.columns or 'label' not in df.columns:
            raise ValueError("输入CSV缺少'beat_id'或'label'字段")
    except Exception as e:
        print(f"读取文件失败 {input_path}: {str(e)}")
        return

    # 定义12导联名称（与上一段代码保持一致）
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # 存储处理后的样本
    processed_samples = []
    # 生成特征列名（格式：导联_f序号，如I_f1, I_f2...）
    feature_columns = []
    for lead in leads:
        for i in range(757):  # 每个导联的PCA特征长度与原始信号一致（757点）
            feature_columns.append(f"{lead}_p{i+1}")

    # 处理每个样本
    for _, row in df.iterrows():
        try:
            beat_id = row['beat_id']
            label = row['label']
            # 提取当前样本的12导联数据（从导联_pXXX列）
            lead_data_list = []
            for lead in leads:
                # 提取该导联的所有数据点列（如I_p1到I_p757）
                lead_cols = [col for col in df.columns if col.startswith(f"{lead}_p")]
                if not lead_cols:
                    raise ValueError(f"未找到导联{lead}的数据列（格式应为{lead}_pXXX）")
                lead_cols.sort(key=lambda x: int(x.split('_p')[1]))  # 按点序号排序
                lead_data = row[lead_cols].values.astype(np.float64)  # 提取该导联数据

                # 确保长度为757点
                if len(lead_data) != 757:
                    lead_data = np.interp(np.linspace(0, len(lead_data)-1, 757),
                                         np.arange(len(lead_data)),
                                         lead_data)
                lead_data_list.append(lead_data)

            # 提取特征
            all_features = []
            for lead_idx, lead_signal in enumerate(lead_data_list):
                vmd_components = WVD.funVMD(pd.Series(lead_signal))
                see_features = WVD.SEE(vmd_components)
                pca_features = WVD.funPCA(see_features)
                all_features.extend(pca_features.values)

            # 构建样本（beat_id + 特征 + label）
            sample = {
                'beat_id': beat_id,
                **dict(zip(feature_columns, all_features)),
                'label': label
            }
            processed_samples.append(sample)

        except Exception as e:
            print(f"处理样本失败（beat_id: {row.get('beat_id', '未知')}）: {str(e)}")
            continue

    # 保存为CSV（含表头，与输入格式一致）
    if processed_samples:
        result_df = pd.DataFrame(processed_samples)
        result_df.to_csv(output_path, index=False)  # 保留表头
        print(f"已处理：{input_path} -> {output_path}（样本数：{len(processed_samples)}）")
    else:
        print(f"无有效样本：{input_path}")


def batch_process_recursive(input_root, output_root):
    """递归处理所有子文件夹中的CSV文件，保持文件夹结构和输出格式"""
    for root, dirs, files in os.walk(input_root):
        # 复刻文件夹结构
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # 处理当前目录下的CSV文件
        for file in files:
            if file.lower().endswith('.csv'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file)
                process_single_file(input_file, output_file)

    print(f"\n所有文件处理完成！结果保存在：{output_root}")


if __name__ == "__main__":
    # 输入根文件夹（JSON转CSV的输出目录）
    input_root_folder = "F:/heartbeat_csv/extracted_beats_unified"
    # 输出根文件夹（特征提取结果，保持相同结构）
    output_root_folder = "F:/heartbeat_csv/SEE"

    batch_process_recursive(input_root_folder, output_root_folder)
