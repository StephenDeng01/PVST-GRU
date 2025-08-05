import os
import pandas as pd
import numpy as np
from OCCO import OCCO

def normalize_lead(lead_data):
    """对每个导联数据点进行样本内归一化（归一化到0-1）"""
    min_val = np.min(lead_data)
    max_val = np.max(lead_data)
    epsilon = 1e-10
    return (lead_data - min_val) / (max_val - min_val + epsilon)

def process_occ_mc_file(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        if 'beat_id' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{input_path} 缺少必要列 'beat_id' 或 'label'")
    except Exception as e:
        print(f"读取失败: {input_path} -> {e}")
        return

    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    processed_samples = []

    # 提取导联列名模板
    lead_template = {lead: [col for col in df.columns if col.startswith(f"{lead}_p")] for lead in leads}
    for lead, cols in lead_template.items():
        lead_template[lead] = sorted(cols, key=lambda x: int(x.split("_p")[1]))

    for _, row in df.iterrows():
        beat_id = row['beat_id']
        label = row['label']
        processed_row = {'beat_id': beat_id}

        for lead in leads:
            lead_cols = lead_template[lead]
            if len(lead_cols) == 0:
                continue

            signal = row[lead_cols].values.astype(np.float64)
            signal = normalize_lead(signal)  # 每个导联在样本内归一化
            signal_series = pd.Series(signal)

            try:
                occo_result = OCCO.MC(signal_series)
                # 输出列名格式：如 I_p1, I_p2, ...
                for i, val in enumerate(occo_result.values):
                    processed_row[f"{lead}_p{i+1}"] = val
            except Exception as e:
                print(f"导联{lead} 处理失败，跳过（beat_id={beat_id}）: {e}")

        # 最后添加 label 保证它是最后一列
        processed_row['label'] = label
        processed_samples.append(processed_row)

    if processed_samples:
        df_out = pd.DataFrame(processed_samples)
        # 明确重新排列列顺序，确保 label 是最后一列
        cols = [col for col in df_out.columns if col != 'label'] + ['label']
        df_out = df_out[cols]
        df_out.to_csv(output_path, index=False)
        print(f"完成: {input_path} -> {output_path}（共{len(processed_samples)}个样本）")
    else:
        print(f"无有效样本: {input_path}")

def batch_occ_mc(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(out_path, exist_ok=True)

        for file in files:
            if file.lower().endswith('.csv'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(out_path, file)
                process_occ_mc_file(input_file, output_file)

    print(f"\n✅ 所有CSV已处理完成，保存到: {output_dir}")

if __name__ == "__main__":
    input_root = "F:/heartbeat_csv/extracted_beats_unified"  # 原始CSV目录
    output_root = "F:/heartbeat_csv/OCCO_MC"                 # 保存MC结果目录
    batch_occ_mc(input_root, output_root)
