import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PSR import PSR


def get_data(filename):
    """
    读取CSV文件（与前文JSON转CSV输出格式一致）并提取ECG数据

    Args:
        filename (str): CSV文件路径（包含beat_id、导联_pXXX列、label）

    Returns:
        tuple: (processed_leads, raw_leads) 处理后的数据和原始数据
    """
    try:
        # 读取CSV文件（包含表头：beat_id、I_p1...I_p757、II_p1...、label等）
        df = pd.read_csv(filename)

        # 验证必要字段
        if 'beat_id' not in df.columns:
            print("错误: CSV文件中未找到'beat_id'字段")
            return None, None
        if 'label' not in df.columns:
            print("错误: CSV文件中未找到'label'字段")
            return None, None

        # 十二导联标准名称
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # 初始化存储结果的字典（键格式："导联_beat_id"）
        raw_leads = {}
        processed_leads = {}

        # 逐行处理每个心拍（每行对应一个beat）
        for idx, row in df.iterrows():
            beat_id = row['beat_id']  # 获取当前心拍ID

            # 处理每个导联
            for lead in leads:
                # 提取该导联的所有数据点列（格式：lead_p1, lead_p2...lead_p757）
                lead_cols = [col for col in df.columns if col.startswith(f"{lead}_p")]
                if not lead_cols:
                    print(f"警告: 导联 {lead} 在beat {beat_id} 中未找到对应列（需格式为{lead}_pXXX）")
                    continue

                # 按点序号排序（确保时间序列顺序正确）
                lead_cols.sort(key=lambda x: int(x.split('_p')[1]))

                # 提取该导联的数据值（转换为浮点数序列）
                lead_data = row[lead_cols].values.astype(float)

                # 生成唯一键（导联_心拍ID）
                key = f"{lead}_{beat_id}"

                # 存储原始数据（pandas Series格式）
                raw_series = pd.Series(lead_data, name=key)
                raw_leads[key] = raw_series

                # 应用PSR处理
                processed_series = PSR.ED(raw_series)  # 调用PSR中的ED方法
                processed_leads[key] = processed_series

        if not raw_leads:
            print("警告: 未解析到任何有效导联数据")
            return None, None

        return processed_leads, raw_leads

    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到")
        return None, None
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        return None, None


def test_single_file(file_path):
    """
    测试单个CSV文件的处理（适配前文输出的CSV格式）

    Args:
        file_path (str): CSV文件路径
    """
    print(f"正在处理文件: {file_path}")
    print("=" * 50)

    # 读取和处理数据（从CSV中提取）
    processed_leads, raw_leads = get_data(file_path)

    if processed_leads is None or raw_leads is None:
        print("文件处理失败，请检查文件路径和格式")
        return

    print(f"成功处理 {len(processed_leads)} 个导联数据")
    print(f"导联列表: {list(processed_leads.keys())[:5]}...")  # 显示前5个示例
    print()

    # 选择第一个导联进行详细分析
    first_lead_key = list(processed_leads.keys())[0]
    raw_data = raw_leads[first_lead_key]
    processed_data = processed_leads[first_lead_key]

    print(f"分析导联: {first_lead_key}")
    print(f"原始数据长度: {len(raw_data)}（应对应757个点）")
    print(f"处理后数据长度: {len(processed_data)}")
    print(f"原始数据范围: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"处理后数据范围: [{processed_data.min():.4f}, {processed_data.max():.4f}]")
    print()

    # 保存处理结果
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存处理后的数据为CSV
    for lead_name, data in processed_leads.items():
        csv_filename = f"{output_dir}/{lead_name}_psr.csv"
        data.to_csv(csv_filename, index=False, header=False)
        print(f"数据已保存到: {csv_filename}")

    # 可视化结果
    visualize_results(raw_data, processed_data, first_lead_key)

    return processed_leads, raw_leads


def visualize_results(raw_data, processed_data, lead_name):
    """
    可视化原始数据和PSR处理后的数据

    Args:
        raw_data (pd.Series): 原始ECG数据（导联的_pXXX列）
        processed_data (pd.Series): PSR处理后的数据
        lead_name (str): 导联名称（格式：导联_beat_id）
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制原始ECG信号（来自CSV的导联_pXXX列）
    ax1.plot(raw_data.index, raw_data.values, 'b-', linewidth=0.8)
    ax1.set_title(f'原始ECG信号 - {lead_name}')
    ax1.set_ylabel('幅值')
    ax1.grid(True, alpha=0.3)

    # 绘制PSR处理后的信号
    ax2.plot(processed_data.index, processed_data.values, 'r-', linewidth=0.8)
    ax2.set_title(f'PSR处理后信号 - {lead_name}')
    ax2.set_xlabel('采样点序号')
    ax2.set_ylabel('欧氏距离')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    img_path = f"{output_dir}/{lead_name}_comparison.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {img_path}")

    plt.show()


def main():
    """主测试函数"""
    print("PSR特征提取测试程序（适配CSV格式）")
    print("=" * 50)

    # 测试文件路径（请替换为实际的CSV文件路径，与前文JSON转CSV输出格式一致）
    test_file_path = "卢显超_denoised.csv"  # 示例：前文生成的CSV文件

    if os.path.exists(test_file_path):
        processed_leads, raw_leads = test_single_file(test_file_path)
    else:
        print(f"测试文件不存在: {test_file_path}")
        print("请修改test_file_path变量为实际的CSV文件路径（需符合前文输出格式）")


if __name__ == "__main__":
    main()