import os
import pandas as pd
import numpy as np
from PSR import PSR  # 确保你的PSR模块中有ED函数


def min_max_normalize(arr):
    """对1维数组做Min-Max归一化"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def extract_psr_and_normalize(input_csv_path):
    """
    读取CSV，提取每个导联PSR特征并归一化（导联内归一化），返回所有样本合成的DataFrame
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
    psr_length = None  # 记录PSR序列长度（通常757）

    for _, row in df.iterrows():
        sample_dict = {}
        sample_dict['beat_id'] = row['beat_id']
        for lead in leads:
            lead_cols = [col for col in df.columns if col.startswith(f"{lead}_p")]
            if not lead_cols:
                raise ValueError(f"导联{lead}数据列缺失于文件 {input_csv_path}")
            lead_cols.sort(key=lambda x: int(x.split('_p')[1]))
            lead_data = row[lead_cols].values.astype(float)

            raw_series = pd.Series(lead_data)
            processed_series = PSR.ED(raw_series)  # PSR特征序列

            if psr_length is None:
                psr_length = len(processed_series)

            # 导联内归一化
            normalized = min_max_normalize(processed_series.values)

            # 拼接所有导联的归一化特征，列名形如 I_p1, I_p2 ...
            for i, val in enumerate(normalized):
                col_name = f"{lead}_p{i+1}"
                sample_dict[col_name] = val

        # label最后放
        sample_dict['label'] = row['label']

        processed_samples.append(sample_dict)

    return pd.DataFrame(processed_samples)


def batch_process_psr_norm(input_root, output_root):
    for root, _, files in os.walk(input_root):
        relative = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith('.csv'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                print(f"处理文件: {input_path}")
                processed_df = extract_psr_and_normalize(input_path)
                if processed_df is not None:
                    processed_df.to_csv(output_path, index=False)
                    print(f"已保存归一化PSR特征: {output_path}")
                else:
                    print(f"跳过文件: {input_path}")


if __name__ == "__main__":
    input_folder = r"F:\heartbeat_csv\extracted_beats_unified"  # 原始CSV根目录
    output_folder = r"F:\heartbeat_csv\PSR"                # 归一化PSR结果输出目录
    batch_process_psr_norm(input_folder, output_folder)
    print("✅ 所有文件处理完成！")
