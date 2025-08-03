import os
import shutil
import glob
from sklearn.model_selection import train_test_split


def extract_patient_id(file_path):
    """从文件名中提取患者ID（根据实际文件名格式调整）"""
    file_name = os.path.basename(file_path)
    # 假设文件名格式为“患者名_denoised.csv”，例如“万明成_denoised.csv”
    patient_id = file_name.split("_denoised.csv")[0]
    return patient_id


def get_all_patients(input_dir):
    """获取所有唯一患者ID及其对应的对应的文件路径"""
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        return None, "未找到任何CSV文件"

    patient_files = {}
    for file in csv_files:
        patient_id = extract_patient_id(file)
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file)

    return patient_files, None


def split_and_simplify_structure(input_dir, output_dir, test_size=0.2, random_state=42):
    """
    按患者划分数据集，保留子文件夹结构但不包含原始根文件夹

    Args:
        input_dir: 输入CSV文件的根目录
        output_dir: 输出目录
        test_size: 测试集比例
        random_state: 随机种子，保证结果可复现
    """
    # 获取所有患者及其文件
    patient_files, error = get_all_patients(input_dir)
    if error:
        print(f"错误: {error}")
        return

    total_patients = len(patient_files)
    print(f"找到 {total_patients} 个唯一患者")

    # 划分患者为训练集和测试集
    all_patients = list(patient_files.keys())
    train_patients, test_patients = train_test_split(
        all_patients,
        test_size=test_size,
        random_state=random_state
    )

    print(f"训练集: {len(train_patients)} 个患者")
    print(f"测试集: {len(test_patients)} 个患者")

    # 创建输出目录
    train_root = os.path.join(output_dir, 'train')
    test_root = os.path.join(output_dir, 'test')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    # 处理训练集患者文件
    train_file_count = 0
    for patient in train_patients:
        for file_path in patient_files[patient]:
            # 计算相对于输入根目录的路径（不含根目录本身）
            rel_path = os.path.relpath(file_path, input_dir)
            # 构建目标路径（直接放在train/test下，不包含原始根目录）
            target_path = os.path.join(train_root, rel_path)
            # 创建目标目录
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # 复制文件
            shutil.copy2(file_path, target_path)
            train_file_count += 1

    # 处理测试集患者文件
    test_file_count = 0
    for patient in test_patients:
        for file_path in patient_files[patient]:
            # 计算相对于输入根目录的路径（不含根目录本身）
            rel_path = os.path.relpath(file_path, input_dir)
            # 构建目标路径
            target_path = os.path.join(test_root, rel_path)
            # 创建目标目录
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # 复制文件
            shutil.copy2(file_path, target_path)
            test_file_count += 1

    print(f"处理完成:")
    print(f"训练集包含 {train_file_count} 个文件，保存至: {train_root}")
    print(f"测试集包含 {test_file_count} 个文件，保存至: {test_root}")


if __name__ == "__main__":
    # 配置路径
    INPUT_DIRECTORY = "F:\heartbeat_csv"  # 输入CSV文件的根目录
    OUTPUT_DIRECTORY = "F:\heartbeat_csv_dataset"  # 输出目录

    # 执行划分
    split_and_simplify_structure(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
