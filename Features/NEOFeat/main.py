import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def NEO(x):
    neo1 = x.shift(1) * x.shift(-1)
    neo1.iloc[0] = x.iloc[0]
    neo1.iloc[-1] = x.iloc[-1]
    neo = x*x-neo1
    # 使用 Min-Max 归一化
    min_val = np.min(neo)
    max_val = np.max(neo)
    if max_val == min_val:
        return pd.Series(np.zeros_like(neo))
    neo = (neo - min_val) / (max_val - min_val)
    return neo


def triangular_window(N):
    window = np.zeros(N)
    for n in range(N):
        window[n] = 2 / (N-1) * ((N - 1) / 2 - np.abs(n - (N - 1) / 2))
    return pd.Series(window)


def SNEO(x):
    neo = NEO(x)
    trw = triangular_window(100)
    convolved_result = np.convolve(trw, neo, mode='same')  # 使用'same'模式保持长度
    # 使用线性插值调整长度为757点
    if len(convolved_result) != 757:
        convolved_result = np.interp(np.linspace(0, len(convolved_result) - 1, 757), 
                                   np.arange(len(convolved_result)), convolved_result)
    # 使用 Min-Max 归一化
    min_val = np.min(convolved_result)
    max_val = np.max(convolved_result)
    if max_val == min_val:
        return pd.Series(np.zeros_like(convolved_result))
    convolved_result = (convolved_result - min_val) / (max_val - min_val)
    return pd.Series(convolved_result)


def min_max_normalize(arr):
    """对1维数组做Min-Max归一化"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def extract_neo_and_normalize(input_csv_path):
    """
    读取CSV，提取每个导联SNEO特征并归一化（导联内归一化），返回所有样本合成的DataFrame
    """
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"读取文件失败 {input_csv_path}: {str(e)}")
        return None

    if 'beat_id' not in df.columns or 'label' not in df.columns:
        print(f"文件缺少 beat_id 或 label 列，跳过: {input_csv_path}")
        return None

    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    processed_samples = []
    sneo_length = None  # 记录SNEO序列长度（通常757）

    for _, row in df.iterrows():
        sample_dict = {}
        sample_dict['beat_id'] = row['beat_id']
        
        for lead in leads:
            # 查找该导联的所有数据点列（格式：{lead}_p{point_number}）
            lead_cols = [col for col in df.columns if col.startswith(f"{lead}_p")]
            if not lead_cols:
                print(f"警告: 导联{lead}数据列缺失于文件 {input_csv_path}")
                continue
                
            # 按点序号排序
            lead_cols.sort(key=lambda x: int(x.split('_p')[1]))
            lead_data = row[lead_cols].values.astype(float)

            # 使用线性插值确保数据长度为757点
            if len(lead_data) != 757:
                lead_data = np.interp(np.linspace(0, len(lead_data) - 1, 757), 
                                    np.arange(len(lead_data)), lead_data)

            raw_series = pd.Series(lead_data)
            
            # 只进行SNEO处理
            sneo_result = SNEO(raw_series)

            if sneo_length is None:
                sneo_length = len(sneo_result)

            # 导联内归一化
            normalized_sneo = min_max_normalize(sneo_result.values)

            # 只拼接SNEO特征，列名形如 I_sneo_1, I_sneo_2 ...
            for i, val in enumerate(normalized_sneo):
                col_name = f"{lead}_p{i+1}"
                sample_dict[col_name] = val

        # label最后放
        sample_dict['label'] = row['label']

        processed_samples.append(sample_dict)

    return pd.DataFrame(processed_samples)


def batch_process_neo_norm(input_root, output_root):
    """批量处理SNEO特征提取和归一化"""
    for root, _, files in os.walk(input_root):
        relative = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.csv'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                print(f"处理文件: {input_path}")
                processed_df = extract_neo_and_normalize(input_path)
                if processed_df is not None:
                    processed_df.to_csv(output_path, index=False)
                    print(f"已保存归一化SNEO特征: {output_path}")
                else:
                    print(f"跳过文件: {input_path}")


def visualize_results_from_csv(csv_path, sample_idx=0):
    """从CSV文件读取数据并可视化NEO/SNEO处理结果"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        return

    if sample_idx >= len(df):
        print(f"样本索引 {sample_idx} 超出范围，文件只有 {len(df)} 个样本")
        return

    # 获取样本数据
    sample = df.iloc[sample_idx]
    beat_id = sample['beat_id']
    
    # 选择第一个导联进行可视化
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
             'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    lead = leads[0]  # 使用第一个导联
    
    # 提取该导联的所有特征
    sneo_cols = [col for col in df.columns if col.startswith(f"{lead}_sneo_")]
    
    if not sneo_cols:
        print(f"未找到导联 {lead} 的SNEO特征")
        return
    
    # 提取特征数据
    sneo_data = sample[sneo_cols].values
    
    # 可视化
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(f'心拍 {beat_id:.0f} - 导联 {lead} 的SNEO特征可视化', fontsize=16)
    
    # SNEO特征
    axes.plot(sneo_data, 'r-', linewidth=0.8)
    axes.set_title('SNEO特征')
    axes.set_xlabel('采样点')
    axes.set_ylabel('幅值')
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存图像
    output_dir = "neo_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/{lead}_beat_{beat_id:.0f}_sneo_features.png", 
                dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_dir}/{lead}_beat_{beat_id:.0f}_sneo_features.png")


if __name__ == "__main__":
    input_folder = r"F:\heartbeat_csv\extracted_beats_unified"  # 原始CSV根目录
    output_folder = r"F:\heartbeat_csv\NEO"                # 归一化NEO/SNEO结果输出目录
    batch_process_neo_norm(input_folder, output_folder)
    print("✅ 所有文件处理完成！")
