"""
把第一行第一列改成 0
"""

import os
import pandas as pd


def modify_first_cell(root_dir):
    """
    遍历根目录下所有CSV文件（包括子文件夹），将每个文件的第一行第一列修改为0

    参数:
        root_dir: 根目录路径
    """
    # 遍历所有文件和子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 筛选CSV文件
            if filename.lower().endswith('.csv'):
                csv_path = os.path.join(dirpath, filename)
                try:
                    # 读取CSV文件（保留索引，方便修改后写回）
                    df = pd.read_csv(csv_path)

                    # 修改第一行第一列（索引0, 列0）
                    if not df.empty:  # 确保文件非空
                        first_col = df.columns[0]  # 获取第一列列名
                        df.at[0, first_col] = 0  # 修改第一行第一列的值为0

                        # 写回CSV文件（覆盖原文件，不保留索引）
                        df.to_csv(csv_path, index=False)
                        print(f"已处理: {csv_path}")
                    else:
                        print(f"跳过空文件: {csv_path}")

                except Exception as e:
                    print(f"处理失败 {csv_path}: {str(e)}")


if __name__ == "__main__":
    # 请替换为你的根目录路径
    ROOT_DIRECTORY = "F:\heartbeat_csv\SEE"  # 例如: "E:/data/csv_files"

    if not os.path.exists(ROOT_DIRECTORY):
        print(f"错误: 目录 {ROOT_DIRECTORY} 不存在")
    else:
        print(f"开始处理目录: {ROOT_DIRECTORY}...")
        modify_first_cell(ROOT_DIRECTORY)
        print("所有CSV文件处理完成")